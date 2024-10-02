import numpy as np
from typing import List, Tuple
import faiss

import setting

def retrieve_chunks_by_keyword(keyword: str, chunks: List[str], num: int)-> str:
    filtered_chunks = [chunk for chunk in chunks if keyword in chunk]
    response = ""
    for i in range(len(filtered_chunks)):
        response += f"キーワード検索でヒットした{i+1}番目の文章: '{filtered_chunks[i]}'"
    
    return response
    
def get_keyword_counts(keyword: str, text: str)-> str:
    count = 0
    words = text.split()
    for word in words:
        if word == keyword:
            count += 1

    response = f"'{keyword}'の出現回数は{count}回です。"
    return response

def retrieve_similar_chunks(query_embedding : str, index: faiss.IndexFlatIP, chunks: List[str], retrieve_num: int) -> List[Tuple[str, float]]:
    similarities, indices = index.search(query_embedding, retrieve_num)
    
    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        if sim >= setting.similarity_threshold:
            results.append((chunks[idx], sim, idx))

    # 時系列順に並び替える
    results.sort(key=lambda x: x[2])
    filtered_chunks = [chunk for chunk, score, idx in results]
    response = ""
    for i in range(len(filtered_chunks)):
        response += f"類似度検索でヒットした{i+1}番目の文章: '{filtered_chunks[i]}'"
    
    return response