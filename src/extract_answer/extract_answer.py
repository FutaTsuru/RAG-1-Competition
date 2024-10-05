import re
from config import setting

def extract_answer(answer: str) -> str:
    pattern = setting.answer_extracct_pattern
    
    match = re.search(pattern, answer)
    
    if match:
        answer = match.group(1).strip()
        return answer
    else:
        print(answer)
        return "回答が見つかりませんでした。"