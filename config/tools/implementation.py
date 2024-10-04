import numpy as np
from typing import List, Tuple
import faiss

from config import setting
from src.rag_system import rag_system

def retrieve_chunks_by_keyword(keyword: str, chunks: List[str])-> str:
    filtered_chunks = [chunk for chunk in chunks if keyword in chunk]
    if len(filtered_chunks) > 10:
        print(f"{len(filtered_chunks)}個のチャンクを追加しました。かなり多いです、気をつけてください!")
    response = ""
    for i in range(len(filtered_chunks)):
        response += f"キーワード検索でヒットした{i+1}番目の文章: '{filtered_chunks[i]}'\n"
    
    return response
    
def get_keyword_counts(keyword: str, text: str)-> str:
    count = text.count(keyword)

    response = f"'{keyword}'の出現回数は{count}回です。\n"
    return response

def retrieve_similar_chunks(query : str, index: faiss.IndexFlatIP, chunks: List[str], retrieve_num: int) -> List[Tuple[str, float]]:
    query_embedding = rag_system.get_embeddings([query])
    similarities, indices = index.search(query_embedding, retrieve_num)   
    results = []

    for idx, sim in zip(indices[0], similarities[0]):
        if sim >= setting.similarity_threshold:
            results.append((chunks[idx], sim, idx))

    results.sort(key=lambda x: x[2])

    filtered_chunks = [chunk for chunk, score, idx in results]
    response = ""
    for i in range(len(filtered_chunks)):
        response += f"類似度検索でヒットした{i+1}番目の文章: '{filtered_chunks[i]}'\n"
    
    return response