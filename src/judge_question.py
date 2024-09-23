from langchain import PromptTemplate
from langchain.chains import RetrievalQA

class JudgeQuestion:
    def __init__(self, llm, retriever, prompt_path):
        self.prompt_template = self.load_system_prompt(prompt_path)
        self.llm = llm
        self.retriever = retriever
    
    def load_system_prompt(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt

    def judge_question(self, query):
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"],
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        return qa.invoke(query)





