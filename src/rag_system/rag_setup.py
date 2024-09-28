from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

SYSTEM_PROMPT_PATH = "./config/system_prompt.md"

def setup_rag(texts, model:str,temperature:float):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .75, "k": 10})

    llm = ChatOpenAI(model_name=model, temperature=temperature)

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

    return qa
