import json
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class MyModel:
    def __init__(self, model_name, tokenizer_name, peft_name):
        quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).cpu()
        self.model = PeftModel.from_pretrained(
            self.model,
            peft_name,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()

    def generate(self, instruction, input=None, maxTokens=256):
        prompt = f"""### 指示:
    {instruction}

    ### 回答:
    """
        prompt = prompt.replace('\n', '<NL>')
        input_ids = self.tokenizer(prompt,
                                   return_tensors="pt",
                                   truncation=True,
                                   add_special_tokens=False).input_ids
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=maxTokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.75,
            top_k=40,
            no_repeat_ngram_size=2,
        )
        outputs = outputs[0].tolist()
        
        if self.tokenizer.eos_token_id in outputs:
            eos_index = outputs.index(self.tokenizer.eos_token_id)
            decoded = self.tokenizer.decode(outputs[:eos_index])

            sentinel = "### 回答:"
            sentinelLoc = decoded.find(sentinel)
            if sentinelLoc >= 0:
                result = decoded[sentinelLoc + len(sentinel):]
                return result.replace("<NL>", "\n")
            else:
                return 'Warning: Expected prompt template to be emitted.  Ignoring output.'
        else:
            return 'Warning: no <eos> detected ignoring output'

model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
peft_name = "lorappo-rinna-3.6b"
tokenizer_name = model_name
my_model = MyModel(model_name, tokenizer_name, peft_name)

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
