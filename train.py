import os
import argparse
import time
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# 파편화 완화
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 여기서는 device를 "cuda" 문자열로만 사용 (모델은 device_map이 관리)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

errors = []  # 디코딩/파싱 에러 기록용
# -------------------------
# 1. Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Llama3 LoRA fine-tuning for QEvasion (CPU offload, no quant)")
parser.add_argument("--model_name", type=str, required=True,
                    help="e.g. meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--experiment", type=str, required=True,
                    choices=["evasion_based_clarity", "direct_clarity"])
parser.add_argument("--batch_size", type=int, default=2)        # 안전하게 1로 시작
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_length", type=int, default=1024)      # 1024 → 512로 줄임
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--output_dir", type=str, default="./outputs_llama3_lora_offload")
args = parser.parse_args()

model_name = args.model_name
experiment = args.experiment
batch_size = args.batch_size
num_epochs = args.num_epochs
max_length = args.max_length
lr = args.lr
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


# -------------------------
# 2. Label mapping
# -------------------------
if experiment == "evasion_based_clarity":
    mapping_labels = {
        "Explicit": 0,
        "Implicit": 1,
        "Dodging": 2,
        "General": 3,
        "Deflection": 4,
        "Partial/half-answer": 5,
        "Declining to answer": 6,
        "Claims ignorance": 7,
        "Clarification": 8,
    }
    label_field = "evasion_label"
elif experiment == "direct_clarity":
    mapping_labels = {
        "Clear Reply": 0,
        "Ambivalent": 1,
        "Clear Non-Reply": 2,
    }
    label_field = "clarity_label"

id2label = {v: k for k, v in mapping_labels.items()}
num_labels = len(mapping_labels)


# -------------------------
# 3. Instruction header
# -------------------------
INSTRUCTION_HEADER = """You are an expert annotator for political interview clarity classification.

You are given an interview question and an answer from a politician.
Your task is to classify how the answer addresses the question,
using exactly one of the following labels:

"""

if experiment == "evasion_based_clarity":
    INSTRUCTION_HEADER += """1. Explicit – A clear, direct answer to the question.
2. Implicit – The answer implies a stance but does not explicitly state it.
3. Dodging – The speaker avoids answering the question and shifts to something else.
4. Deflection – The speaker redirects the focus to another person, issue, or question.
5. Partial/half-answer – The answer addresses only part of the question.
6. General – The answer is very vague or generic and does not address specifics.
7. Declining to answer – The speaker explicitly refuses to answer.
8. Claims ignorance – The speaker claims not to know the answer.
9. Clarification – The speaker asks for clarification or reinterprets the question.
"""
else:
    INSTRUCTION_HEADER += """1. Clear Reply – A clear, direct answer to the question.
2. Ambivalent – The answer is partially addressing the question or is ambiguous.
3. Clear Non-Reply – The answer does not address the question at all.
"""

INSTRUCTION_HEADER += """

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

# -------------------------
# 4. Dataset class
# -------------------------
class LlamaQEDataset(Dataset):
    """
    mode = 'train' : prompt + 정답 라벨까지 이어 붙여서 LM 학습
    mode = 'val'   : prompt만 저장, gold label index는 따로 저장 (generate로 평가)
    """

    def __init__(self, rows, tokenizer, mapping_labels, label_field, max_length=512, mode="train"):
        self.tokenizer = tokenizer
        self.mapping_labels = mapping_labels
        self.label_field = label_field
        self.max_length = max_length
        self.mode = mode
        self.rows = rows

        self.texts = []
        self.labels = []

        for row in rows:
            label_str = row[label_field]
            if label_str not in mapping_labels:
                continue
            label_idx = mapping_labels[label_str]
            prompt = build_prompt(row["interview_question"], row["interview_answer"], row["question"])

            if mode == "train":
                full_text = prompt + " " + label_str
                self.texts.append(full_text)
                self.labels.append(label_idx)
            else:
                self.texts.append(prompt)
                self.labels.append(label_idx)

        print(f"[{mode}] num_samples = {len(self.texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]   # prompt + " " + label
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)

        # --------------------------
        # 🎯 Loss Mask 만들기
        # --------------------------
        # prompt 길이와 label 길이를 구해야 함
        prompt = build_prompt(self.rows[idx]["interview_question"],
                              self.rows[idx]["interview_answer"],
                              self.rows[idx]["question"])
        label_str = id2label[self.labels[idx]]
        full_text = prompt + " " + label_str

        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        prompt_len = prompt_enc["input_ids"].shape[1]

        # 모든 토큰은 기본적으로 mask = -100
        labels = input_ids.clone()
        labels[:prompt_len] = -100   # prompt 부분은 학습 제외

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
            "gold_label_idx": torch.tensor(self.mapping_labels[self.labels[idx]], dtype=torch.long),
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "gold_label_idxs": gold_label_idxs,
    }

# -------------------------
# 5. Load dataset & split
# -------------------------
dataset = load_dataset("ailsntua/QEvasion")
rows = [row for row in dataset["train"]]

all_labels = [mapping_labels[row[label_field]] for row in rows if row[label_field] in mapping_labels]
label_counts = Counter(all_labels)
print("Label counts:", label_counts)

train_rows, val_rows = train_test_split(
    rows,
    test_size=0.1,
    random_state=1,
    stratify=[row[label_field] for row in rows],
)

# -------------------------
# 6. Tokenizer & Model (Llama3 + LoRA + CPU offload, no quantization)
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# prepare offload folder and dtype based on available hardware
offload_folder = os.path.join(output_dir, "cpu_offload")
os.makedirs(offload_folder, exist_ok=True)
torch_dtype = torch.float16 if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch_dtype,
    device_map="auto",
    offload_folder=offload_folder,
    offload_state_dict=True,
    low_cpu_mem_usage=True,
)

# gradient checkpointing & use_cache 비활성화 → 메모리 절약
model.gradient_checkpointing_enable()
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("Device map:", model.hf_device_map)
model.print_trainable_parameters()
# ⚠️ 여기서 model.to(device) 절대 호출하지 않음!
# device_map + offload가 이미 모델의 배치를 관리함


# -------------------------
# 7. Build Datasets & Dataloaders
# -------------------------
train_dataset = LlamaQEDataset(
    train_rows, tokenizer, mapping_labels, label_field, max_length=max_length, mode="train"
)
val_dataset = LlamaQEDataset(
    val_rows, tokenizer, mapping_labels, label_field, max_length=max_length, mode="val"
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
)

# -------------------------
# 8. Optimizer & Scheduler
# -------------------------
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)
# -------------------------
# 9. Zero-shot case test
# -------------------------
def predict(tokenizer, model, question, answer, subanswer):
    prompt = build_prompt(question, answer, subanswer)

    enc = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Move inputs to CUDA only when CUDA is available. For device-mapped/offloaded models
    # keep inputs on CPU otherwise and let the model manage device placement.
    if torch.cuda.is_available():
        input_ids = enc["input_ids"].cuda()
        attention_mask = enc["attention_mask"].cuda()
    else:
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Label parsing
    try:
        if "Label:" in decoded:
            after = decoded.split("Label:")[-1].strip()
            pred_label = after.split()[0]  # 여기서 IndexError 가능
        else:
            pred_label = "UNKNOWN"
        # unknown case
        if pred_label not in mapping_labels:
            raise ValueError(f"Unknown label: {pred_label}")
    except Exception as e:
        # 에러 발생 시 기록
        errors.append({
            "error": str(e),
            "decoded": decoded,
            "raw_prompt": prompt if "prompt" in locals() else "N/A"
        })
    # fallback label
        pred_label = "Explicit"
    return pred_label, decoded

# # zero-shot test on val set
# print("Running zero-shot test...")
# zero_shot_results = []
# first_token_times = []

# for row in val_rows:
#     question = row["interview_question"]
#     answer = row["interview_answer"]
#     subanswer = row["question"]
#     gold_label = row[label_field]
#     # Measure time-to-first-token using CUDA events only if CUDA is available.
#     if torch.cuda.is_available():
#         first_token_start = torch.cuda.Event(enable_timing=True)
#         first_token_end = torch.cuda.Event(enable_timing=True)
#         first_token_start.record()
#         pred_label, model_output = predict(tokenizer, model, question, answer, subanswer)
#         first_token_end.record()
#         torch.cuda.synchronize()
#         elapsed_time = first_token_start.elapsed_time(first_token_end) / 1000.0  # ms -> s
#     else:
#         t0 = time.time()
#         pred_label, model_output = predict(tokenizer, model, question, answer, subanswer)
#         elapsed_time = time.time() - t0

#     zero_shot_results.append({
#         "gold": gold_label,
#         "pred": pred_label,
#         "question": question,
#         "answer": answer,
#         "subanswer": subanswer,
#         "model_output": model_output,
#     })
#     first_token_times.append(elapsed_time)
    
# zero_shot_acc = sum(1 for r in zero_shot_results if r["gold"] == r["pred"]) / len(zero_shot_results)
# zero_shot_macro_f1 = f1_score(
#     [mapping_labels[r["gold"]] for r in zero_shot_results],
#     [mapping_labels[r["pred"]] for r in zero_shot_results],
#     average="macro",
# )
# with open(os.path.join(output_dir, "zero_shot_results.txt"), "w") as f:
#     for r in zero_shot_results:
#         f.write(f"Gold Label: {r['gold']}\n")
#         f.write(f"Pred Label: {r['pred']}\n")
#         f.write("Question:\n" + r["question"] + "\n")
#         f.write("Answer:\n" + r["answer"] + "\n")
#         f.write("Subanswer:\n" + r["subanswer"] + "\n")
#         f.write("Model Output:\n")
#         f.write("--------------------------------------------------\n")
#         f.write(r["model_output"] + "\n")
#         f.write("--------------------------------------------------\n")
# # TTFT computation
# time_to_first_token = sum(first_token_times) / len(first_token_times)
# print(f"Zero-shot Accuracy: {zero_shot_acc:.4f}")
# print(f"Zero-shot Macro F1: {zero_shot_macro_f1:.4f}")
# print(f"Time to first token: {time_to_first_token:.4f} seconds")

# with open(os.path.join(output_dir, "errors.txt"), "w") as f:
#     for error in errors:
#         f.write(f"Error: {error['error']}\n")
#         f.write(f"Decoded: {error['decoded']}\n")
#         f.write(f"Raw Prompt: {error['raw_prompt']}\n")
#         f.write("--------------------------------------------------\n")
# -------------------------
# -------------------------
# 10. Training & Validation
# -------------------------
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(num_epochs):
    print(f"\n========== Epoch {epoch+1}/{num_epochs} ==========")
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training"):
        # 입력만 GPU로 보냄 (모델은 자동으로 샤딩/오프로딩)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch["labels"],
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

    # -------------------------
    # Validation (generate → label 파싱 → F1)
    # -------------------------
    model.eval()
    val_losses = 0.0
    all_preds = []
    all_golds = []
    prev_best_f1 = 0.0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} - Validation"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch["labels"],
                )
                val_losses += outputs.loss.item()

                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=16,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            # Label parsing
            try:
                if "Label:" in decoded:
                    after_label = decoded.split("Label:")[-1].strip()
                    pred_token = after_label.split()[0]
                else:
                    pred_token = "Explicit"  # fallback
            except Exception as e:
                errors.append({
                    "error": str(e),
                    "decoded": decoded,
                    "raw_prompt": prompt if "prompt" in locals() else "N/A"
                })
                pred_token = "Explicit"

            if pred_token not in mapping_labels:
                pred_idx = Counter(all_labels).most_common(1)[0][0]
            else:
                pred_idx = mapping_labels[pred_token]

            all_preds.append(pred_idx)
            all_golds.extend(batch["gold_label_idxs"].cpu().numpy().tolist())

    avg_val_loss = val_losses / len(val_dataloader)
    acc = accuracy_score(all_golds, all_preds)
    macro_f1 = f1_score(all_golds, all_preds, average="macro")

    print(
        f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} - "
        f"Acc: {acc*100:.2f}% - Macro F1: {macro_f1:.4f}"
    )
    if macro_f1 > prev_best_f1:
        ckpt_dir = os.path.join(output_dir, "best_model")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"New best model saved to {ckpt_dir}")
        prev_best_f1 = macro_f1

with open(os.path.join(output_dir, "errors.txt"), "a") as f:
    for error in errors:
        f.write(f"Error: {error['error']}\n")
        f.write(f"Decoded: {error['decoded']}\n")
        f.write(f"Raw Prompt: {error['raw_prompt']}\n")
        f.write("--------------------------------------------------\n")
# -------------------------
final_dir = os.path.join(output_dir, f"llama3_qevasion_{experiment}")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print("Training finished. Model saved to:", final_dir)