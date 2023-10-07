import json


# 辞書のリストを格納するための空のリスト
conversation_list = []

what_did_you_reply_to = ""

# ファイルを開いて行ごとに処理
with open('./data/andy_mori.txt', 'r', encoding='utf-8') as file:
    
    for line in file:
        # 行からinstructionとoutputを抽出
        parts = line.strip().split(':')
        if len(parts) == 2:
            instruction, output = parts

            if instruction.strip() == 'Andy':
                conversation_dict = {"instruction": what_did_you_reply_to, "output": output.strip()}
                conversation_list.append(conversation_dict)
            
            what_did_you_reply_to = output.strip()

# 結果をJSON形式で出力
with open('output.json', 'w', encoding='utf-8') as json_file:
    json.dump(conversation_list, json_file, ensure_ascii=False, indent=4)
