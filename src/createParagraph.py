import os
from dotenv import load_dotenv 

from openai import OpenAI
from langchain_core.documents.base import Document

class DevideTextIntoParagraph:
    def __init__(self, documents):
        self.documents = documents

    def devide_text(self):
        load_dotenv()

        texts = []

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        system_prompt = (
            "You are a language model tasked with dividing the given text into paragraphs. "
            "When dividing, pay special attention to the meaning and structure of the content, "
            "ensuring that each paragraph is coherent and logically grouped. "
            "Each paragraph should contain at least 500 tokens, ensuring completeness and clarity."
        )


        for document in self.documents:
            text = document.page_content
            metadata = document.metadata
            title = metadata["title"]
            user_prompt = f"Please divide the following text into meaningful paragraphs:\n\n{text}"

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5 
            )

            divided_text = response.choices[0].message.content

            paragraphs = divided_text.split("\n\n")

            for i, paragraph in enumerate(paragraphs):
                number = i+1
                paragraph = f"{title}の{number}段落目: {paragraph}"
                doc = Document(page_content=paragraph, metadata={"title" : title, "paragraph_number" : number})
                texts.append(doc)

        return texts