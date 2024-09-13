import os
from dotenv import load_dotenv 
load_dotenv()  

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. 知識ベースの準備
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', 'novels', 'works', '競漕.txt')

loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# テキストを小さなチャンクに分割
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
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
    chain_type="map_reduce",
    retriever=retriever,
)

# 5. RAGシステムの使用
def rag_response(query):
    return qa.invoke(query)

# 6. ユーザーからの入力を受け取り、AIの応答を表示
def main():
    print("質問を入力してください。終了するには 'exit' と入力してください。")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit', '終了']:
            print("対話を終了します。")
            break
        response = rag_response(query)
        print(f"AI: {response['result']}\n")

if __name__ == "__main__":
    main()
