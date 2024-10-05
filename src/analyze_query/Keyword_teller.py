import pandas as pd
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from config import setting

class Keyword_teller:
    def __init__(self) -> None:
        self.df = pd.read_csv(setting.KEYWORD_PATH, header=None)

    def connect_query_to_keyword(self, index: int) -> list:
        keyword = list(self.df.iloc[index-1, 1].split(" "))
        return keyword
    
    def connect_query_to_related_word(self, index: int) -> list:
        related_word = (self.df.iloc[index-1, 2].split(" "))
        return related_word
    
    def connect_query_to_important_word(self, index: int) -> str:
        important_word = self.df.iloc[index-1, 3]
        return important_word
    
    def connect_query_to_answer_example(self, index: int) -> str:
        answer_example = self.df.iloc[index-1, 4]
        return answer_example
    

if __name__ == "__main__":
    keyword_teller = Keyword_teller()
    for i in range(1, 61):
        print(f"{i}:{keyword_teller.connect_query_to_keyword(i)},{keyword_teller.connect_query_to_related_word(i)}, {keyword_teller.connect_query_to_important_word(i)}")
