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
    
    similarities, indices = index.search(query_embedding, setting.retrieval_num)
    
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
    system_prompt = system_prompt.format(retrieved_chunk=chunk_str)
    return system_prompt

def generate_answer(system_prompt: str, query: str) -> str:
    message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
    while True:
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
        argument_name = arguments.keys[0]
        function_name = tool_call['function']['name']
        function = getattr(implementation, function_name, None)

        argument = arguments.get('argument_name')
        chunk = function(argument)

        function_call_result_message = {
            "role": "tool",
            "content": json.dumps({
                f"{argument_name}": argument,
                "information": chunk
            }),
            "tool_call_id": response['choices'][0]['message']['tool_calls'][0]['id']
        }

        message.append(response['choices'][0]['message'])
        message.append(function_call_result_message)

    return answer

def run_rag_system(query: str, texts: List[str], embeddings: np.ndarray):
    index = build_faiss_index(embeddings)

    retrieved_chunks_with_scores = retrieve_similar_chunks(query, index, texts)

    retrieved_chunks = [chunk for chunk, score, idx in retrieved_chunks_with_scores]

    system_prompt = build_prompt(retrieved_chunks)

    answer = generate_answer(system_prompt, query)

    return answer, retrieved_chunks
