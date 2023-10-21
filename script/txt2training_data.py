import json

# データファイルを開く
with open('../data/data.txt', 'r', encoding='utf-8') as file:
    # 会話のリストを保存するための空リストを用意
    conversation_list = []
    # otherのメッセージとandyのメッセージを一時保存するための変数を用意
    other_output = ""
    andy_output = ""
    
    # ファイルの中の各行を一つずつ処理
    for line in file:
        # 行からスピーカーと発言を抽出。":"を基に分割
        parts = line.strip().split(':')
        if len(parts) == 2:
            speaker, output = parts
            speaker = speaker.strip()
            output = output.strip()

            # スピーカーがAndyの場合
            if speaker == 'Andy':
                # もしandy_outputにすでに何か入っていたら、新しい発言を追加
                if andy_output:
                    andy_output += "\n" + output
                    
                # それ以外の場合は新しい発言をandy_outputに代入
                else:
                    andy_output = output
                    
            # スピーカーがAndy以外の場合
            else:
                # もしandyがすでに何か言っていた場合、その会話をリストに追加してリセット
                if andy_output:
                    conversation_list.append({"other": other_output, "andy": andy_output})
                    other_output = output
                    andy_output = ""
                    
                # Andyがまだ発言していない場合、other_outputに発言を追加
                else:
                    # もしother_outputにすでに何か入っていたら、新しい発言を追加
                    if other_output:
                        other_output += "\n" + output
                        
                    # それ以外の場合は新しい発言をother_outputに代入
                    else:
                        other_output = output

    # 最後の部分を処理。もしファイルの最後にotherとandyの両方が発言していた場合、その会話をリストに追加
    if other_output and andy_output:
        conversation_list.append({"other": other_output, "andy": andy_output})

# 結果をJSON形式でファイルに出力
with open('output.json', 'w', encoding='utf-8') as json_file:
    json.dump(conversation_list, json_file, ensure_ascii=False, indent=4)
