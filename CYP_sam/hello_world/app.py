import json
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
peft_name = "/opt/ml/model"

# モデルの準備
#model = AutoModelForCausalLM.from_pretrained(peft_name, device_map="auto", load_in_8bit=True)

print("model loading")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

"""
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)
"""

model = PeftModel.from_pretrained(
    model,
    peft_name,
    # device_map="auto"
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(peft_name, use_fast=False)

# 評価モード
model.eval()

print("model loaded")

def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    body = json.loads(event.get("body", "{}"))
    instruction = body.get("instruction")
    
    if instruction is None:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No instruction was provided'})
        }

    # 推論
    prompt = f"""### 指示:
    {instruction}

    ### 回答:
    """
    prompt = prompt.replace('\n', '<NL>')
    input_ids = tokenizer(prompt,
                          return_tensors="pt",
                          truncation=True,
                          add_special_tokens=False).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.75,
        top_k=40,
        no_repeat_ngram_size=2,
    )
    outputs = outputs[0].tolist()
    # print(tokenizer.decode(outputs))

    # EOSトークンにヒットしたらデコード完了
    if tokenizer.eos_token_id in outputs:
        eos_index = outputs.index(tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[:eos_index])

        # レスポンス内容のみ抽出
        sentinel = "### 回答:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            result = decoded[sentinelLoc + len(sentinel):]
            return {
                'statusCode': 200,
                'body': json.dumps({'response': result.replace("<NL>", "\n")})
            }
        else:
            return 'Warning: Expected prompt template to be emitted.  Ignoring output.'
    else:
        return 'Warning: no <eos> detected ignoring output'

