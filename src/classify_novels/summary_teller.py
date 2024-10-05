import pandas as pd
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from config import setting

class Summary_teller:
    def __init__(self) -> None:
        self.summary_csv_path = setting.summary_csv_path

    def connect_title_to_summary(self, title: str) -> str:
        file_name = title + ".txt"
        with open(self.summary_csv_path + file_name, 'r', encoding='utf-8') as file:
            summary = file.read()
        return summary
