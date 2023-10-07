# 基本パラメータ

model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
peft_name = "lorappo-rinna-3.6b"
output_dir = "lorappo-rinna-3.6b-results"

from transformers import AutoTokenizer

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

CUTOFF_LEN = 256  # コンテキスト長

# トークナイズ関数
def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }



# データセットをJSONからロード
import json

with open("formatted_line.json", "r", encoding='utf-8') as f:
    loaded_data = json.load(f)

data_1 = [item for item in loaded_data if not item['input'].startswith('RT')]
data = [item for item in data_1 if not item['input'].startswith('@')]

print("データ数:", len(data))


# プロンプトテンプレートの準備
def generate_prompt(data_point):
    result = f"""### 指示:
{data_point["input"]}

### 回答:
{data_point["completion"]}
"""
    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result


# データセットの準備
VAL_SET_SIZE = 1000

train_dataset = []
val_dataset = []

for i in range(len(data)):
    if i % 5 == 0:
        x = tokenize(generate_prompt(data[i]), tokenizer)
        val_dataset.append(x)
    else:
        x = tokenize(generate_prompt(data[i]), tokenizer)
        train_dataset.append(x)

from transformers import AutoModelForCausalLM

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

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

import transformers
eval_steps = 200
save_steps = 200
logging_steps = 20

# トレーナーの準備
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=transformers.TrainingArguments(
        num_train_epochs=3,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=3,
        push_to_hub=False,
        auto_find_batch_size=True
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 学習の実行
model.config.use_cache = False
trainer.train()
model.config.use_cache = True

# LoRAモデルの保存
trainer.model.save_pretrained(peft_name)