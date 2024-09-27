from dotenv import load_dotenv 
load_dotenv()  

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from config import setting
from make_chunk import document_importer
from make_chunk import small_chunk
from make_chunk import big_chunk
from rag_system import rag_setup
from execute_rag.executor import executor

# 1. 知識ベースの準備
novel_lists = setting.novel_lists
documents = document_importer.import_documents(novel_lists)

texts = []
# テキストを小さなチャンクに分割
texts += small_chunk.make_small_chunk(documents, setting.small_chunk_size, setting.small_chunck_overlap, setting.small_chunck_separator)

# 各小説の全文もベクトル化する。(gpt-4oが処理する最大トークンを超えそうな小説は分割する。)
# texts += big_chunk.make_big_chunck(documents, setting.big_chunk_size, setting.big_chunck_overlap, setting.big_chunck_separator)

# 4. RAGシステムの構築
qa = rag_setup.setup_rag(texts, setting.model, setting.temperature)

# 5. RAGシステムの使用
executor = executor(qa)

if __name__ == "__main__":
    executor.run()