import pandas as pd
from collections import defaultdict
from langchain.text_splitter import CharacterTextSplitter

from config import setting

def make_and_save_small_chunk(documents, chunk_size: int, chunk_overlap: int, separator: str)-> dict:
    """
    テキストを小さなチャンクに分割する関数
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    texts = text_splitter.split_documents(documents)

    title_counter = defaultdict(int)
    splited_texts = []
    title_list = []

    for text in texts:
        title = text.metadata['title']
        title_list.append(title)

        title_counter[title] += 1
        n = title_counter[title]

        splited_text = f"{title}の{n}番目の文章: {text.page_content}"
        splited_texts.append(splited_text)
    
    chunk_dict = {"title": title_list, "chunk": splited_texts}
    chunk_dataset = pd.DataFrame(chunk_dict)
    chunk_dataset.to_csv(setting.CHUNK_PATH, index=False)