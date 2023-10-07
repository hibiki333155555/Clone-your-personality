参考　
- https://qiita.com/tsukemono/items/990490e0090c0607b00c
- https://note.com/npaka/n/nc387b639e50e

1. 既存の日本語llmをダウンロード。　
2. 会話のデータセットを用意
3. LoRAで学習 (low rank adaptation 学習が早くなる。技術詳細は知らん。計算の近似らしい)
4. 重みを保存
5. linebot などで　推論コードを呼び出す。(generate.pyなど)

とりあえず、36億パラメータのrinnaで試す
- https://huggingface.co/rinna/japanese-gpt-neox-3.6b


## 学習用データの整形
とりあえず完了　データ数806  

学習用データフォーマット1 -> ### 指示:<NL>文章<NL><NL>### 入力:<NL>文章<NL><NL>### 回答:<NL>文章<NL>

学習用データフォーマット2 -> ### 指示:<NL>文章<NL><NL>### 回答:<NL>文章<NL>

とりあえずデータ2で。
andy-mori.txtを dictionaryの配列にして generate_promptに渡す。  
メッセージはAさんBさんAさんBさんのように交互ではなくて  
A A A B B A B のように不規則に繰り返される。  
とりあえず　what_did_you_reply_to に文章を保存して学習対象が発言したら(今回はAndyが発言したら)  
replyの対象をinstructionにreplyをoutputにした。  
andyが連続で発言している場合　instruction output　ともに andy　になっている。  
もっといいアイデアを募集  
```python
conversation_list = []

what_did_you_reply_to = ""

with open('./data/andy_mori.txt', 'r', encoding='utf-8') as file:
    
    for line in file:
        parts = line.strip().split(':')
        if len(parts) == 2:
            instruction, output = parts

            if instruction.strip() == 'Andy':
                conversation_dict = {"instruction": what_did_you_reply_to, "output": output.strip()}
                conversation_list.append(conversation_dict)
            
            what_did_you_reply_to = output.strip()

with open('output.json', 'w', encoding='utf-8') as json_file:
    json.dump(conversation_list, json_file, ensure_ascii=False, indent=4)
```

```python
# プロンプトテンプレートの準備
# ↑のoutputが引数
def generate_prompt(data_point):
    result = f"""### 指示:
    {data_point["instruction"]}

    ### 回答:
    {data_point["output"]}"""

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result
```

## 環境構築
requirment

```zsh
pip install -Uqq  git+https://github.com/huggingface/peft.git
pip install -Uqq transformers datasets accelerate 
pip install -i https://test.pypi.org/simple/ bitsandbytes
pip install sentencepiece
```

docker??

仮想環境作成　python3 -m venv name_of_your_env  
仮想環境起動　source name_of_your_env/bin/activate  
仮想環境の無効化　deactivate  
  
jupyter notebookを使いたい場合  
pip install ipykernel  
ipython kernel install --user --name=name_of_your_env  
  
pip install notebook  
jupyter notebook  
  
kernel -> name_of_your_env