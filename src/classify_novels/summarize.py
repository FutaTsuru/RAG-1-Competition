from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from config import setting
from tqdm import tqdm


client = OpenAI()

system_prompt = """私は７つの小説に関する60の質問をもっていますが，どの質問がどの小説に関するものかを把握していません．
                    そこで，llmとrag技術を用いて質問がどの小説に関するものであるかを分類するシステムを構築しています．
                    私の設計では，それぞれの小説の特徴の抽出と要約を作成し，その要約をベクトル化して保存し，それをrag技術で参照して分類を行っていこうとかんがえています．
                    あなたには，要約を作成する手伝いを指定もらいます．要約は設問の分類に使用することを考慮してください．特に以下の点に注意してください
                    ・出力は　特徴：~ 要約:~ という形式で出力してください．
                    ・特徴には質問を分類するのに有用であると考えられる小説の特徴や単語を記してください．
                    ・特徴に含まれるものは例えば登場人物の名前や地名などの固有名詞，小説の中で意味や何らかの役割を持っている物体の名称，小説のジャンルや舞台，時代背景などが考えられます．その他にも重要だと思われる単語は特徴に含めてください．
                    ・文章中からすべての人物名，地名を抽出し．特徴に記載してください．
                    ・人物名でなくても特定の人を指す役職，代名詞等の単語は必ず含めるようにしてください．例えば老婆，少年など．特に文中で繰り返し使われる単語は重要な特徴となることが多いです．
                    ・特定の状況を示す単語や，特徴的な単語，内容を端的に表す単語も特徴に含めてください．
                    ・要約には小説の内容を簡潔にまとめてください．こちらでも固有名詞や人物名，地名などはなるべくそのまま記載してください．
                    ・要約は文章の内容をなるべく損なわないようにしてください，また具体的な出来事やエピソードも漏れなく含めてください．
                    ・要約は4000文字以内でお願いします．ただし，あまりにも短い要約は内容が損なわれることがあるので，なるべく内容を含むようにしてください．
                    ・文章によっては，1,2,3,等と区切られて複数の内容が含まれていることがあります．その場合でも全ての文章について上記のような特徴を抽出することと，内容や具体的なエピソードを損なわれないように要約を作成してください．ただし，内容ごとに特徴や要約をわけないでください． """

next_prompt_part1 = """文章が長いので分割しています．
                    ここまでの内容は次のようなものでした．
                    """
next_prompt_part2 = """
                    次に続きの文章を示しますので，今までの内容と含めて出力を行ってください．
                    
                    """

script_dir = os.path.dirname(os.path.abspath(__file__))

def read_file(file_path):
    """ファイルから文章を読み込む"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_output(output, output_path="output.txt"):
    """出力をファイルに書き込む"""
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(output + "\n")

def gpt4_response(prompt):
    """GPT-4 APIにリクエストを送信し、応答を得る"""
    response = client.chat.completions.create(
        model="gpt-4o",  # モデルを指定
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def process_large_prompt(prompt, max_length=25000):
    """プロンプトが規定の文字数を超える場合、改行で区切って分割"""
    if len(prompt) <= max_length:
        return [prompt]
    
    parts = []
    current_part = ""
    
    for line in prompt.splitlines(True):  # 改行を保持しながら分割
        if len(current_part) + len(line) > max_length:
            parts.append(current_part)
            current_part = ""
        current_part += line
    
    if current_part:
        parts.append(current_part)
    
    return parts

if __name__ == "__main__":
    #for novel in tqdm(setting.novel_lists, desc="Processing summary", unit="file"):
    for novel in ["死生に関するいくつかの断想.txt"]:
        novel_title = novel.replace(".txt", "")
        # ファイルパスを指定
        file_path = os.path.join(script_dir, '..', '..', 'novels', 'works', novel)
        
        # ファイルから文章を読み込み
        prompt = read_file(file_path)
    
        # 入力が25000文字を超える場合に分割
        prompt_parts = process_large_prompt(prompt)
    
        # 各プロンプトをGPT-4に送信して応答を取得
    
        for idx, part in enumerate(prompt_parts):
            if idx > 0:
                part = next_prompt_part1 + output + next_prompt_part2 + part
            output = gpt4_response(part)
            
            # 出力をファイルに保存
        output = f"小説のタイトル：{novel_title}\n\n{output}"
        file_path = os.path.join(script_dir, '..', '..', 'storage', 'summarize', novel)
        write_output(output,file_path)
    
    print("処理が完了しました。結果はsummarizeに保存されました。")
