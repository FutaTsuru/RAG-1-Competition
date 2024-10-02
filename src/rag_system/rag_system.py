import numpy as np
from dotenv import load_dotenv
import json
from typing import List, Tuple
from openai import OpenAI
import faiss

from config import setting
from config.tools import tools_setting
from config.tools import implementation

load_dotenv()
client = OpenAI()

def get_embeddings(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        input=texts,
        model=setting.embedding_model
    )
    embeddings = [data.embedding for data in response.data]
    return np.array(embeddings).astype('float32')

def make_and_save_embeddings(texts: List[str], save_path) -> np.ndarray:
    embeddings_array = get_embeddings(texts)
    np.save(save_path, embeddings_array)


def build_faiss_index(embeddings: np.ndarray, use_cosine: bool = False) -> faiss.IndexFlatIP:
    if use_cosine:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1]) 
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_similar_chunks(query: str, index: faiss.IndexFlatIP, texts: List[str]) -> List[Tuple[str, float]]:
    query_embedding = get_embeddings([query])
    
    similarities, indices = index.search(query_embedding, setting.similarrity_retrieval_num)
    
    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        if sim >= setting.similarity_threshold:
            results.append((texts[idx], sim, idx))

    # 時系列順に並び替える
    results.sort(key=lambda x: x[2])
    return results

def build_prompt(retrieved_chunk: List[str]) -> str:
    with open(setting.SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as file:
            system_prompt = file.read()
    chunk_str = ""
    for i in range(len(retrieved_chunk)):
        chunk_str += f"{i+1}つ目: '{retrieved_chunk[i]}'\n"
    system_prompt = system_prompt.format(retrieved_chunk=chunk_str, function_calling_response="")
    return system_prompt

def generate_answer(system_prompt: str, query: str, chunks: List[str], index: faiss.IndexFlatIP, target_text) -> str:
    function_calling_response = ""
    while True:
        message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        response = client.chat.completions.create(
            model=setting.model,
            messages=message,
            temperature=setting.temperature,
            tools=tools_setting.tools,
            tool_call="auto",
            n=1,
            stop=None,
        )

        if response.choices[0].finish_reason != "tool_calls":
            answer = response.choices[0].message.content
            break
        
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call['function']['arguments'])
        function_name = tool_call['function']['name']
        if function_name == "retrieve_chunks_by_keyword":
            function_calling_response += implementation.retrieve_chunks_by_keyword(arguments["keyword"], chunks)

        elif function_name == "get_keyword_counts":
            function_calling_response += implementation.get_keyword_counts(arguments["keyword"], target_text)
        
        # elif function_name == "retrieve_similar_chunks":
        #     function_calling_response += implementation.retrieve_similar_chunks(query, index, chunks, arguments["retrieval_num"])

        function_call_result_message = {
            "role": "tool",
            "content": json.dumps({
                "arguments": arguments,
                "information": function_calling_response
            }),
            "tool_call_id": response['choices'][0]['message']['tool_calls'][0]['id']
        }

        system_prompt.format(function_calling_response=function_calling_response)

        message.append(response['choices'][0]['message'])
        message.append(function_call_result_message)

    return answer

def run_rag_system(query: str, chunks: List[str], embeddings: np.ndarray, target_text: str):
    index = build_faiss_index(embeddings)

    retrieved_chunks_with_scores = retrieve_similar_chunks(query, index, chunks)

    retrieved_chunks = [chunk for chunk, score, idx in retrieved_chunks_with_scores]

    system_prompt = build_prompt(retrieved_chunks)

    answer = generate_answer(system_prompt, query, chunks, index, target_text)

    return answer, retrieved_chunks
