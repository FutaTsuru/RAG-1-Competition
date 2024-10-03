from dotenv import load_dotenv
load_dotenv()
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
grand_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
sys.path.append(grand_parent_dir)
from config import setting
from classify_executor import executor


novel_lists = setting.novel_lists
splited_texts = []

for novel in novel_lists:
    file_path = os.path.join(script_dir, '..', '..', 'storage', 'summarize', novel)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        splited_texts.append(content)

executor = executor(splited_texts)

if __name__ == "__main__":
    executor.run()
