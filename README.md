参考　
- https://qiita.com/tsukemono/items/990490e0090c0607b00c
- https://note.com/npaka/n/nc387b639e50e

1. 既存の日本語llmをダウンロード。　
2. 会話のデータセットを用意
3. LoRAで学習 
4. 重みを保存
5. linebot などで　推論コードを呼び出す。(generate.pyなど)

とりあえず、36億パラメータのrinnaで試す
- https://huggingface.co/rinna/japanese-gpt-neox-3.6b


## 学習用データの整形

学習用データフォーマット1 -> ### 指示:<NL>文章<NL><NL>### 入力:<NL>文章<NL><NL>### 回答:<NL>文章<NL>


学習用データフォーマット2 -> ### 指示:<NL>文章<NL><NL>### 回答:<NL>文章<NL>

とりあえずデータ2で。

```python
# プロンプトテンプレートの準備
def generate_prompt(data_point):
    result = f"""### 指示:
    {data_point["instruction"]}

    ### 回答:
    {data_point["output"]}"""

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result
```