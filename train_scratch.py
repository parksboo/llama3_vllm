import os
import argparse
import torch
from torch import bfloat16
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from peft import LoraConfig



# -------------------------
# 1. Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Llama3 LoRA fine-tuning for QEvasion (CPU offload, no quant)")
parser.add_argument("--model_name", type=str, default="e.g. meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--experiment", type=str, default="evasion_based_clarity", choices=["evasion_based_clarity", "direct_clarity"])
parser.add_argument("--batch_size", type=int, default=2)        
parser.add_argument("--num_epochs", type=int, default=4)
parser.add_argument("--max_length", type=int, default=None)
parser.add_argument("--lr", type=float, default=1e-4)
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
    INSTRUCTION_HEADER += """1. Explicit – The information requested is explicitly stated (in the requested form).
2. Implicit – The information requested is given, but without being explicitly stated (not in the expected form).
3. Dodging – Ignoring the question altogether.
4. Deflection – Starts on topic but shifts the focus and makes a different point than what is asked.
5. Partial/half-answer – Offers only a specific component of the requested information.
6. General – The information provided is too general/lacks the requested specificity.
7. Declining to answer – Acknowledge the question but directly or indirectly refusing to answer at the moment.
8. Claims ignorance – The answerer claims/admits not to know the answer themselves.
9. Clarification – Does not provide the requested information and asks for clarification.
"""
else:
    INSTRUCTION_HEADER += """1. Clear Reply – A clear, direct answer to the question.
2. Ambivalent – The answer is partially addressing the question or is ambiguous.
3. Clear Non-Reply – The answer does not address the question at all.
"""

INSTRUCTION_HEADER += """

Read the following interview question and answer segment.
Then output the label in the format: "Label: <LABEL>".

"""

def build_prompt(data):
    system_prompt = INSTRUCTION_HEADER
    user_prompt = f"Interview Question: {data['interview_question']}\n\nFull Answer: {data['interview_answer']}\n\nLabel:"
    assistant_response = f"{data[label_field]}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]
def build_messages_dataset(raw_dataset):
    messages_list = []
    for row in raw_dataset:
        msgs = build_prompt(row)
        messages_list.append({"messages": msgs})
    return messages_list

# -------------------------
# 4. Load dataset
# -------------------------

raw_dataset = load_dataset("ailsntua/QEvasion", split="train")
rows = [row for row in raw_dataset]
labels = [row[label_field] for row in raw_dataset]

# 2) stratified split
train_rows, eval_rows = train_test_split(
    rows,
    test_size=0.1,
    random_state=3407,
    stratify=labels,
)

dataset = build_messages_dataset(train_rows)
train_dataset = Dataset.from_list(dataset)
eval_dataset  = Dataset.from_list(build_messages_dataset(eval_rows))

LLAMA3_MASKING_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
        "{% if loop.index0 == 0 %}"
            "{% set start_token = bos_token %}"
        "{% else %}"
            "{% set start_token = '' %}"
        "{% endif %}"

        "{% if message['role'] == 'assistant' %}"
            "{{ start_token + '<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n\n' }}"
            "{% generation %}"
            "{{ message['content'] | trim + '<|eot_id|>' }}"
            "{% endgeneration %}"
        "{% else %}"
            "{{ start_token + '<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>' + '\n\n' }}"
    "{% endif %}"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.chat_template = LLAMA3_MASKING_TEMPLATE
lora_args = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_args = SFTConfig(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = 8,
        max_steps = len(dataset) * num_epochs // (batch_size * 8),
        warmup_steps=len(dataset) * num_epochs // (batch_size * 8*20),
        learning_rate = lr,
        logging_steps = 10,
        optim = "adamw_torch",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        
        batch_eval_metrics=True,
        
        dataset_text_field=None,     
        packing=False, 
        assistant_only_loss=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=bfloat16,
    device_map="auto",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
        peft_config=lora_args,
        processing_class=tokenizer
    )

trainer.train()

model.eval()

tokenizer.padding_side = "left"

def format_prompt_for_inference(example):

    input_messages = example["messages"][:-1] 
    
    if input_messages[-1]["role"] != "user":
        pass

    prompt = tokenizer.apply_chat_template(
        input_messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )
    return prompt

ground_truths = [ex["messages"][-1]["content"] for ex in eval_dataset]

# ==========================================
# 3. 배치 추론 (Batch Inference) 수행
# ==========================================

eval_batch_size = 8 
generated_texts = []

print("Starting Inference...")

torch.cuda.empty_cache()

for i in tqdm(range(0, len(eval_dataset), eval_batch_size)):
    batch_indices = range(i, min(i + eval_batch_size, len(eval_dataset)))
    batch_examples = [eval_dataset[idx] for idx in batch_indices]
    
    prompts = [format_prompt_for_inference(ex) for ex in batch_examples]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,     
            do_sample=False,       
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    input_len = inputs.input_ids.shape[1]
    decoded_output = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    generated_texts.extend(decoded_output)

# ==========================================
# 4. 결과 파싱 및 F1-Score 계산
# ==========================================

def parse_prediction(text):
    text = text.strip().lower()
    for label_name, label_id in mapping_labels.items():
        if text.startswith(label_name.lower()):
            return label_id
    return -1 

y_pred = [parse_prediction(text) for text in generated_texts]
y_true = []

for gt in ground_truths:
    if gt in mapping_labels:
        y_true.append(mapping_labels[gt])
    else:
        y_true.append(-1)

macro_f1 = f1_score(y_true, y_pred, average="macro")

print(f"\nEvaluation Result:")
print(f"Macro F1-Score: {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(mapping_labels.keys()), labels=list(mapping_labels.values())))

# -------------------------
# 5. Save the fine-tuned model and results

trainer.save_model(output_dir+"/new_model")

with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
    f.write(f"model output - {experiment}\n")
    for text in generated_texts:
        f.write(text + "\n")
        f.write("===========================================\n")
        f.write(f"answer - {ground_truths[generated_texts.index(text)]}\n")
        f.write("===========================================\n\n")
        
  
    