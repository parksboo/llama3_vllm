import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ========================================
# 0. 설정
# ========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
FINETUNED_DIR = "/home/supa/vllm_practice/outputs_llama3_lora_offload/best_model"

# label mapping (evasion-based clarity 9-class)
mapping_labels = {
    "Explicit": 0,
    "Implicit": 1,
    "Dodging": 2,
    "Deflection": 3,
    "Partial/half-answer": 4,
    "General": 5,
    "Declining to answer": 6,
    "Claims ignorance": 7,
    "Clarification": 8,
}
id2label = {v: k for k, v in mapping_labels.items()}


# ========================================
# 1. 모델 + 토크나이저 로드
# ========================================
def load_model():
    # tokenizer는 반드시 base model에서!
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="cpu_offload",
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        FINETUNED_DIR,
        local_files_only=True
    )
    model.eval()
    return tokenizer, model



# ========================================
# 2. Prompt 생성 함수
# ========================================
INSTRUCTION_HEADER = """You are an expert annotator for political interview clarity classification.

You are given an interview question and an answer from a politician.
Your task is to classify how the answer addresses the question,
using exactly one of the following labels:

1. Explicit – A clear, direct answer to the question.
2. Implicit – The answer implies a stance but does not explicitly state it.
3. Dodging – The speaker avoids answering the question and shifts to something else.
4. Deflection – The speaker redirects the focus to another person, issue, or question.
5. Partial/half-answer – The answer addresses only part of the question.
6. General – The answer is very vague or generic and does not address specifics.
7. Declining to answer – The speaker explicitly refuses to answer.
8. Claims ignorance – The speaker claims not to know the answer.
9. Clarification – The speaker asks for clarification or reinterprets the question.

Read the following interview question, full answer, and the annotated sub-answer segment.
Then output the label in the format: "Label: <LABEL>".

"""


def build_prompt(question, answer, subanswer):
    system_msg = INSTRUCTION_HEADER

    user_msg = (
        f"Question:\n{question}\n\n"
        f"Full answer:\n{answer}\n\n"
        f"Sub-answer segment:\n{subanswer}\n\n"
    )

    assistant_prefix = "Label:"

    # ChatML format
    prompt = (
        "<|system|>\n" + system_msg + "\n"
        "<|user|>\n" + user_msg + "\n"
        "<|assistant|>\n" + assistant_prefix
    )

    return prompt

# ========================================
# 3. 단일 샘플 inference
# ========================================
def predict(tokenizer, model, question, answer, subanswer):
    prompt = build_prompt(question, answer, subanswer)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Label parsing
    if "Label:" in decoded:
        after = decoded.split("Label:")[-1].strip()
        pred_label = after.split()[0]
    else:
        pred_label = "UNKNOWN"

    # unknown label → fallback
    if pred_label not in mapping_labels:
        pred_label = "Explicit"

    return pred_label, decoded



# ========================================
# 4. 전체 test set inference
# ========================================
def run_test_inference():
    tokenizer, model = load_model()
    dataset = load_dataset("ailsntua/QEvasion")

    test_rows = dataset["test"]

    results = []

    print(f"\nRunning inference on {len(test_rows)} test samples...")

    for i, row in enumerate(test_rows):
        q = row["interview_question"]
        a = row["interview_answer"]
        sub = row["question"]
        gold = row["evasion_label"]

        pred, full_out = predict(tokenizer, model, q, a, sub)

        results.append({
            "index": i,
            "gold": gold,
            "pred": pred,
            "question": q,
            "answer": a,
            "subanswer": sub,
            "model_output": full_out
        })

        # 간단한 진행 표시
        if i % 50 == 0:
            print(f"Processed {i}/{len(test_rows)}...")

    return results



# ========================================
# 5. 실행 및 출력
# ========================================
if __name__ == "__main__":
    results = run_test_inference()

    print("\n==================== SAMPLE RESULTS ====================")
    for r in results[:5]:   # 처음 5개만 preview
        print(f"\nIndex: {r['index']}")
        print(f"Gold Label: {r['gold']}")
        print(f"Pred Label: {r['pred']}")
        print("Question:", r["question"])
        print("Answer:", r["answer"])
        print("Subanswer:", r["subanswer"])
        print("Model Output:", r["model_output"])
        
    # 결과를 파일로 저장
    output_file = "inference_results.txt"
    with open(output_file, "w") as f:
        for r in results:
            f.write(f"Index: {r['index']}\n")
            f.write(f"Pred Label: {r['pred']}\n")
            f.write("Question:\n" + r["question"] + "\n")
            f.write("Answer:\n" + r["answer"] + "\n")
            f.write("Subanswer:\n" + r["subanswer"] + "\n")
            f.write("Model Output:\n" + r["model_output"] + "\n")
            f.write(f"")
            f.write("--------------------------------------------------\n")