参考

- https://qiita.com/tsukemono/items/990490e0090c0607b00c
- https://note.com/npaka/n/nc387b639e50e

1. 既存の日本語 llm をダウンロード。
2. 会話のデータセットを用意
3. LoRA で学習 (low rank adaptation 学習が早くなる。技術詳細は知らん。計算の近似らしい)
4. 重みを保存
5. linebot などで　推論コードを呼び出す。(generate.py など)

とりあえず、36 億パラメータの rinna で試す

- https://huggingface.co/rinna/japanese-gpt-neox-3.6b

## スケジュール

10/7 開始
10/13 train_v2 学習 ok

予定
10/15 bot 化

## 学習用データの整形

andy-mori.txt を dictionary の配列にして generate_prompt に渡す。  
メッセージは A さん B さん A さん B さんのように交互ではなくて  
A A A B B A B のように不規則に繰り返される。  
とりあえず　 what_did_you_reply_to に文章を保存して学習対象が発言したら(今回は Andy が発言したら)  
reply の対象を instruction に reply を output にした。  
andy が連続で発言している場合　 instruction output 　ともに andy 　になっている。  
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

## 環境構築

requirment

```zsh
pip install -Uqq  git+https://github.com/huggingface/peft.git
pip install -Uqq transformers datasets accelerate
pip install -i https://test.pypi.org/simple/ bitsandbytes
pip install sentencepiece
```

docker??

仮想環境作成　 python3 -m venv name_of_your_env  
仮想環境起動　 source name_of_your_env/bin/activate  
仮想環境の無効化　 deactivate

jupyter notebook を使いたい場合  
pip install ipykernel  
ipython kernel install --user --name=name_of_your_env

pip install notebook  
jupyter notebook

kernel -> name_of_your_env

## AWS の環境について

#### やったこと

ローカルで作成したモデルをいれた dokcer 環境を AWS ECR を利用して lambda 関数としてしようできるようにした(lambda のメモリが 3GB しか使えなくて 10GB ほど使う本プロジェクトでは現状動いていない状況)

#### step 1: aws-cli のインストールと docker image の作成

まず aws-cli のインストールから　-> https://docs.aws.amazon.com/ja_jp/cli/latest/userguide/getting-started-install.html こちらからインストールしてください。

次に docker image を作成します。
マシンに docker(docker-compose)が入っていることを確認して(docker --version)、function 配下に requirements.txt を作成して次の単語を入れる。

transformers
torch
peft

次に

dockerfile の FUNCTION_DIR を自分の環境のパスに変更する

そして

sudo docker build --platform linux/amd64 -t makgpt:test1 .
をプロジェクトの root ディレクトリで実行
docker images でビルドした image を確認できる

#### step 2: AWS ECR の紐づけと ECR への docker image の push

ECR の紐づけ -> https://docs.aws.amazon.com/ja_jp/AmazonECR/latest/userguide/getting-started-cli.html　のステップ２を参考に

##### ECR への image の push

docker tag hello-world:latest (aws_account_id).dkr.ecr.(region).amazonaws.com/(repo-name)
でリポジトリに push する image にタグをつけた後に

docker push (aws_account_id).dkr.ecr.(region).amazonaws.com/(repo-name)

で image を push することができる

###### ECR からの image の pull

docker pull (aws_account_id).dkr.ecr.(region).amazonaws.com/(repo-name):(repo-tag)

#### step 3: lambda 関数の作成

aws の lambda 関数のコンソールに行き、
関数の作成 -> コンテナイメージ ->　関数名の入力と ECR イメージの選択 -> 関数の作成

作成した関数のコンソールに飛び、設定からメモリを 10GB,タイムアウトを 10 分に設定してください。
また同じく設定から 関数 URL の作成をしてその関数 URL を使用する

#### step 4: テスト

curl "(関数 url)" -d '{"body":"{\"action\":\"こんにちは\"}"}'
