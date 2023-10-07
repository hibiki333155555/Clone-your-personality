### 喜びロード
LLMに "あなた" の会話を学習させて　あなたそっくりな "くろーん" を作りましょう。  
あなたの　くろーん　は任意のsnsであなたらしく振る舞い、あなたの時間やエネルギーを節約してくれます。

### 理想とする くろーん
1. あなたらしいこと
2. 会話の相手との関係性を理解していること

### 手順
1. 既存の日本語llmをダウンロード。　
2. 会話のデータセットを用意
3. QLoRAで学習 
4. 重みを保存
5. linebot などで　推論コードを呼び出す。(generate.pyなど)

日本語llmはいろいろあるね。とりあえず、36億パラメータのrinnaで試してみよう。


### 一般的なllmの学習プロセスについて
文章を数値で表現してあげないとコンピュータは理解できないよ。  
文章の数値表現にはいろいろな方法があるよ。  
