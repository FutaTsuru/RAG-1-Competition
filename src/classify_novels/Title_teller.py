import pandas as pd
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from config import setting

class Title_teller:
    def __init__(self) -> None:
        self.df = pd.read_csv(setting.classify_csv_path)

    def connect_query_to_novel(self, index: int) -> str:
        raw_title = self.df.iloc[index-1][1]
        title = ""
        for ti in setting.novel_lists:
            checker = ti.replace(".txt", "")
            if setting.title_filter_dict[checker] in raw_title:
                title = checker
        if title == "":
            title = "分かりません"
            if "分かりません" not in raw_title:
                print(f"{raw_title}は正しい解釈ができませんでした。")
        return title

if __name__ == "__main__":
    title_teller = Title_teller()
    for i in range(1, 60):
        print(title_teller.connect_query_to_novel(i))