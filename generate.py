import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "rinna/japanese-gpt2-small"
peft_name = "lorappo-rinna-3.6b"
output_dir = "lorappo-rinna-3.6b-results"

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)
"""


# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# LoRAモデルの準備
model = PeftModel.from_pretrained(
    model,
    peft_name,
    # device_map="auto"
)

# 評価モード
model.eval()


# プロンプトテンプレートの準備
def generate_prompt(data_point):
    result = f"""### 指示:
    {data_point["instruction"]}

    ### 回答:
    """

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result


# テキスト生成関数の定義
def generate(instruction, input=None, maxTokens=256) -> str:
    # 推論
    prompt = generate_prompt({'instruction': instruction, 'input': input})
    input_ids = tokenizer(prompt,
                          return_tensors="pt",
                          truncation=True,
                          add_special_tokens=False).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=maxTokens,
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
            print(result)
            return result.replace("<NL>", "\n")  # <NL>→改行
        else:
            return 'Warning: Expected prompt template to be emitted.  Ignoring output.'
    else:
        return 'Warning: no <eos> detected ignoring output'


# テキスト生成
print("今日暇？\n{0}".format(generate('今日暇?')))
print("それはあり得ないだろ\n{0}".format(generate('それはあり得ないだろ')))
