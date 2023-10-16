import pickle
import torch
import transformers
import copy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

import boto3


def tokenize_dataset(data_point, tokenizer, ignore_index=-100):
    features = []

    for data in data_point:
      instruction_text = ""
      # if data['instruction'] != "":
      #    instruction_text = data['instruction'] + "\n"
      prompt_full = f"[INST]\n{instruction_text}[/INST]\n{data['other']}{data['andy']}{tokenizer.eos_token}"
      prompt_no_output = f"[INST]\n{instruction_text}[/INST]\n{data['other']}"

      #ignore indexに指定するほう
      if len(tokenizer.encode(prompt_full)) >= 2048:
          continue
      tokenized_full = tokenizer(prompt_full, padding='longest', truncation=True, max_length=2048, return_tensors='pt')
      tokenized_no_output = tokenizer(prompt_no_output, padding='longest', truncation=True, max_length=2048, return_tensors='pt')

      input_ids = tokenized_full['input_ids'][0]
      labels = copy.deepcopy(input_ids)
      source_len = len(tokenized_no_output['input_ids'])

      labels[:source_len] = ignore_index
      
      features.append({
          'input_ids': input_ids,
          'labels': labels
      })
    return features


model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
peft_name = "lorappo-rinna-3.6b"
output_dir = "lorappo-rinna-3.6b-results"

from transformers import AutoTokenizer, AutoModelForCausalLM

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

CUTOFF_LEN = 256  # コンテキスト長

import json

with open("outputtest.json", "r", encoding='utf-8') as f:
    data = json.load(f)


print("データ数:", len(data))


class InstructCollator():
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, examples):
        input_batch = []
        label_batch = []
        for example in examples:
            input_batch.append(example['input_ids'])
            label_batch.append(example['labels'])
        input_ids = pad_sequence(
            input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # labelsのpaddingトークンは先程と同様にignore_indexである-100で埋める
        labels = pad_sequence(
            label_batch, batch_first=True, padding_value=self.ignore_index
        )
        # attention_maskはbool値でもいいらしい
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
config = AutoConfig.from_pretrained(model_name,use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    load_in_8bit=True
)

VAL_SET_SIZE = int(len(data) * 0.05)

new_dataset = Dataset.from_dict({k: [dic[k] for dic in data] for k in data[0]})
print(f"データセットの件数 = {len(new_dataset)}")
# 学習データと検証データの準備
train_val = new_dataset.train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=1990
)
train_data = train_val["train"]
val_data = train_val["test"]
tokenized_train = tokenize_dataset(train_data, tokenizer)
tokenized_val = tokenize_dataset(val_data, tokenizer)
collator = InstructCollator(tokenizer)
loader = DataLoader(tokenized_train, collate_fn=collator, batch_size=8, shuffle=True)
batch = next(iter(loader))

#学習パラメータの指定
# MICRO_BATCH_SIZE = 2
BATCH_SIZE = 32
from transformers import AutoModelForCausalLM

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

# LoRAのパラメータ
lora_config = LoraConfig(
    r= 8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# モデルの前処理
model = prepare_model_for_int8_training(model)

# LoRAモデルの準備
model = get_peft_model(model, lora_config)

# 学習可能パラメータの確認
model.print_trainable_parameters()

eval_steps = 200
save_steps = 200
logging_steps = 20

# トレーナーの準備
trainer = transformers.Trainer(
    #model=model.to(torch.bfloat16),
    model = model,
    data_collator=collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    args=transformers.TrainingArguments(
        num_train_epochs=4,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        # per_device_train_batch_size=MICRO_BATCH_SIZE,
        # per_device_eval_batch_size=MICRO_BATCH_SIZE,
        # gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
        dataloader_num_workers=12,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        auto_find_batch_size=True
    )
)


# 学習の実行
model.config.use_cache = False
trainer.train()
model.config.use_cache = True

# モデルの保存
torch.save(model.state_dict(), "clone.model")

# S3にモデルをアップロード
s3 = boto3.client('s3')
bucket_name = "clone-you"
object_name = "clone.model"

print("uploading...")
try:
    s3.upload_file("clone.model", bucket_name, object_name)
except Exception as e:
    print(e)
