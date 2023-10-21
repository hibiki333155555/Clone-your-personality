import subprocess

base_model = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
lora_weight = "../lorappo-rinna-3.6b"
quantization_type = "int8"
merge_output_dir = "../merged-rinnna-ppo"
output_dir = "../merged-rinnna-ppo-int8" + "_ct2"

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model, use_fast=False
)

peft_model = PeftModel.from_pretrained(
    model,
    lora_weight,
    device="auto",
)

peft_model.merge_and_unload()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

command = f"ct2-transformers-converter --force --model {output_dir} --quantization {quantization_type} --output_dir {output_dir}"
subprocess.run(command, shell=True)