from openai import OpenAI
import re
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from tqdm import tqdm
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
grand_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
sys.path.append(grand_parent_dir)
from config import setting
import numpy as np

client = OpenAI()

def load_system_prompt(md_file_path):
    """Markdownファイルからシステムプロンプトを読み込む"""
    with open(md_file_path, 'r', encoding='utf-8') as file:
        system_prompt = file.read()
    return system_prompt

def ask_gpt4(question, system_prompt):
    """GPT-4に質問を渡して回答を得る"""
    response = client.chat.completions.create(
        model=setting.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=setting.temperature,
        stop=None,
    )
    responses = response.choices[0].message.content
    
    return responses

def extract_words(text):
    # キーワードの部分を抽出するための正規表現パターン
    keyword_pattern = r"キーワード：(.+)"
    related_word_pattern = r"関連ワード：(.+)"
    important_word_pattern = r"最重要ワード：(.+)"
    answer_example_pattern = r"回答例：(.+)"
    
    # キーワードの部分を取得
    keyword_match = re.search(keyword_pattern, text)
    related_word_match = re.search(related_word_pattern, text)
    important_word_match = re.search(important_word_pattern, text)
    answer_example_match = re.search(answer_example_pattern, text)
    if not(keyword_match) or not(related_word_match) or not(important_word_match) or not(answer_example_match):
        return "","","",""
    keyword = keyword_match.group(1).strip().replace(","," ").replace("　"," ")
    related_word = related_word_match.group(1).strip().replace(","," ").replace("　"," ")
    important_word = important_word_match.group(1).strip().replace(","," ").replace("　"," ")
    answer_example = answer_example_match.group(1).strip().replace(","," ").replace("　"," ")
    return keyword, related_word, important_word, answer_example

def main():
    data = {"index": [], "keyword": [], "related_word": [], "important_word": [], "answer_example": []}
    question_db = pd.read_csv("./question/query.csv")
    system_prompt = load_system_prompt(setting.ANALYZE_SYSTEM_PROMPT_PATH)
    for _, row in tqdm(question_db.iterrows(), total=len(question_db), desc="回答生成"):
        index = row['index']
        query = row['problem']
        keyword, related_word, important_word, answer_example = "","","",""
        s = 0
        while keyword == "" or related_word == "" or important_word == "" or answer_example == "":
            response = ask_gpt4(query, system_prompt)
            keyword, related_word, important_word, answer_example = extract_words(response)
            s += 1
            if s > 1:
                tqdm.write(f"質問{index}の試行回数: {s-1}")
        data["index"].append(index)
        data["keyword"].append(keyword)
        data["related_word"].append(related_word)
        data["important_word"].append(important_word)
        data["answer_example"].append(answer_example)
    keyword_db = pd.DataFrame(data)
    keyword_db.to_csv(setting.KEYWORD_PATH, index=False, header=False)

if __name__ == "__main__":
    main()
