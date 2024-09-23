import os
import pandas as pd
import ast
from dotenv import load_dotenv 
load_dotenv()  

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

SYSTEM_PROMPT_PATH = "./config/system_prompt.md"

# 1. 知識ベースの準備
novel_lists = [
    'カインの末裔.txt',
    'サーカスの怪人.txt',
    '芽生.txt',
    '競漕.txt',
    '死生に関するいくつかの断想.txt',
    '小説　不如帰.txt',
    '流行暗殺節.txt'
]

script_dir = os.path.dirname(os.path.abspath(__file__))

documents = []

for novel in novel_lists:
    novel_title = novel.replace(".txt", "")
    file_path = os.path.join(script_dir, '..', 'novels', 'works', novel)
    loader = TextLoader(file_path, encoding='utf-8')
    novel_documents = loader.load()
    novel_documents[0].metadata['title'] = novel_title  # タイトルをメタデータに追加
    documents += novel_documents

# テキストを小さなチャンクに分割
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator="\n")
texts = text_splitter.split_documents(documents)

# 各チャンクの先頭に小説タイトルを付与
for text in texts:
    title = text.metadata['title']
    text.page_content = f"{title}: {text.page_content}"

# 各小説の全文もベクトル化する。(gpt-4oが処理する最大トークンを超えそうな小説は分割する。)
for document in documents:
    max_words = 30000
    title = document.metadata['title']
    novel_length = len(document.page_content)
    if novel_length > max_words:
        novel_half_splitter = CharacterTextSplitter(chunk_size=max_words, chunk_overlap=100, separator="\n")
        half_docs = novel_half_splitter.split_documents([document])
        for half_doc in half_docs:
            half_doc.page_content = f'{title}の文章の大部分: {half_doc.page_content}'
            texts.append(half_doc)
    else:
        document.page_content = f'{title}の全文: {document.page_content}'
        texts.append(document)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as file:
        system_prompt = file.read()

prompt = PromptTemplate(
    template=system_prompt,
    input_variables=["context", "question"],
)

# 4. RAGシステムの構築
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# 5. RAGシステムの使用
def rag_response(query):
    return qa.invoke(query) 

def main():
    question_db = pd.read_csv("./question/query.csv")
    
    # データを格納するためのリストを作成
    data = {"index": [], "answer": [], "reason": []}
    
    for _, row in question_db.iterrows():
        index = row['index']
        problem = row['problem']
        
        # RAGシステムの応答を取得
        response = rag_response(problem)
        reason = response["source_documents"]
        result = response["result"].replace("\n", "")

        # 返答が50字を超える場合、45字以内にする.
        if len(response) > 45:
            response = response[:45]
        
        # データをリストに追加
        data["index"].append(index)
        data["answer"].append(result)
        # data["reason"].append(reason)
        data["reason"].append("なし")
    
    # データをDataFrameに変換
    prediction_db = pd.DataFrame(data)
    
    # CSVに保存
    prediction_db.to_csv("./evaluation/submit/predictions.csv", index=False, header=False)

if __name__ == "__main__":
    main()
