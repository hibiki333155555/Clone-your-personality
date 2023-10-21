import json
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import ctranslate2
import boto3
import uuid

class MyModel:
    def __init__(self, model_name, tokenizer_name, quant_model):
        self.generator = ctranslate2.Generator(quant_model, device="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    # データベースに会話を追加
    def insert_conversation(self, speaker, text):
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('llm_conversations')
        
        response = table.put_item(
            Item={
                'conversationID': str(uuid.uuid4()),
                'speaker': speaker,
                'text': text
            }
        )
        return response
    # データベースから過去の会話を取得
    def get_past_conversations(self):
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('llm_conversations')
        
        response = table.scan()
        items = response.get('Items', [])
                
        conversations = [{"speaker": item["speaker"], "text": item["text"]} for item in items]
        
        print(conversations)
        return conversations

    def prompt(self, instruction):
        past_conversations = self.get_past_conversations()
        p = past_conversations + [{"speaker": "ユーザー", "text": instruction}]
        
        p = [f"{uttr['speaker']}: {uttr['text']}" for uttr in p]
        p = "<NL>".join(p)
        p = p + "<NL>" + "システム: "
        return p
    
    def generate(self, instruction, input=None, maxTokens=256):
        # TokenizationとCTRANSLATE2の入力形式への変換
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(self.prompt(instruction),add_special_tokens=False,))
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
quant_model = "../merged-rinnna-ppo-int8_ct2"
tokenizer_name = model_name
my_model = MyModel(model_name, tokenizer_name, quant_model)

print(my_model.generate("なんのチーズがすき？"))

"""
my_model.insert_conversation("ユーザー", "なんのチーズがすき？")
responce_text = my_model.generate("なんのチーズがすき？")
print(responce_text)
my_model.insert_conversation("システム", responce_text)
"""