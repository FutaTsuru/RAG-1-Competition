from dotenv import load_dotenv 
load_dotenv()  

import sys
import os
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import setting
from make_chunk import document_importer
from make_chunk import small_chunk
from execute_rag.executor import executor

# 1. 知識ベースの準備
novel_lists = setting.novel_lists
documents = document_importer.import_documents(novel_lists)

# テキストを小さなチャンクに分割して保存する関数
# small_chunk.make_and_save_small_chunk(documents, setting.small_chunk_size, setting.small_chunck_overlap, setting.small_chunck_separator)

# 分割したチャンクが格納されているcsvファイルの読み込み
splited_texts_db = pd.read_csv(setting.CHUNK_PATH)

# 5. RAGシステムの使用
executor = executor(splited_texts_db, documents)

if __name__ == "__main__":
    executor.run()