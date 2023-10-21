import json
from transformers import AutoTokenizer
import ctranslate2

class MyModel:
    def __init__(self, model_name, tokenizer_name, quant_model):
        self.generator = ctranslate2.Generator(quant_model, device="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        
    def prompt(instruction):
        p = [
            {"speaker": "ユーザー", "text": instruction},
        ]
        p = [f"{uttr['speaker']}: {uttr['text']}" for uttr in p]
        p = "<NL>".join(p)
        p = p + "<NL>" + "システム: "
        return p
    
    def generate(self, instruction, input=None, maxTokens=256):
        # TokenizationとCTRANSLATE2の入力形式への変換
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(MyModel.prompt(instruction),add_special_tokens=False,))
        results = self.generator.generate_batch(
            [tokens],
            max_length=256,
            sampling_topk=10,
            sampling_temperature=0.9,
            include_prompt_in_result=False,
        )

        text = self.tokenizer.decode(results[0].sequences_ids[0])
        return text

model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
quant_model = "merged-rinnna-ppo-int8_ct2"
tokenizer_name = model_name
my_model = MyModel(model_name, tokenizer_name, quant_model)

print(my_model.generate("今日はいい天気ですね。"))

def handler(event, context):
    body = json.loads(event.get("body", "{}"))
    instruction = body.get("instruction")
    
    if instruction is None:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No instruction was provided'})
        }

    response_text = my_model.generate(instruction)

    return {
        'statusCode': 200,
        'body': json.dumps({'response': response_text})
    }
