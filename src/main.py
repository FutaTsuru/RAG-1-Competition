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

novel_lists = ['カインの末裔.txt',
               'サーカスの怪人.txt',
               '芽生.txt',
               '競漕.txt',
               '死生に関するいくつかの断想.txt',
               '小説　不如帰.txt',
               '流行暗殺節.txt']

# 1. 知識ベースの準備
script_dir = os.path.dirname(os.path.abspath(__file__))

documents = []

for novel in novel_lists:
    file_path = os.path.join(script_dir, '..', 'novels', 'works', novel)
    loader = TextLoader(file_path, encoding='utf-8')
    documents += loader.load()

# テキストを小さなチャンクに分割
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator="\n")
texts = text_splitter.split_documents(documents)

# 2. ベクトルストアの作成
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. GPT-4モデルの準備
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# 4. RAGシステムの構築
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
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
        problem = problem + ' 50字以内で、回答のみ出力してください。'
        
        # RAGシステムの応答を取得
        response = rag_response(problem)["result"]
        response = response.replace("\n", "")

        # 返答が50字を超える場合、50字以内にする.
        if len(response) > 50:
            response = response[:50]
        
        # データをリストに追加
        data["index"].append(index)
        data["answer"].append(response)
        data["reason"].append("なし")
    
    # データをDataFrameに変換
    prediction_db = pd.DataFrame(data)
    
    # CSVに保存
    prediction_db.to_csv("./evaluation/submit/predictions.csv", index=False, header=False)

if __name__ == "__main__":
    main()
