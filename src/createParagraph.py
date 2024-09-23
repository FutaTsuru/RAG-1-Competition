import os
from dotenv import load_dotenv 
import pandas as pd
from openai import OpenAI
from langchain_core.documents.base import Document

class DivideTextIntoParagraph:
    def __init__(self, documents):
        self.documents = documents

    def divide_text(self):
        load_dotenv()

        texts = []

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        system_prompt = (
            "You are a language model tasked with dividing the given text into paragraphs. "
            "When dividing, pay special attention to the meaning and structure of the content, "
            "ensuring that each paragraph is coherent and logically grouped. "
            "Each paragraph should contain at least 300 tokens, ensuring completeness and clarity."
        )

        for document in self.documents:
            text = document.page_content
            metadata = document.metadata
            title = metadata["title"]
            user_prompt = f"Please divide the following text into meaningful paragraphs:\n\n{text}"

            response = client.chat.completions.create(
                model="gpt-4o",  # 正しいモデル名に修正
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5 
            )

            divided_text = response.choices[0].message.content

            paragraphs = divided_text.split("\n\n")

            # スライディングウィンドウ方式で段落をペアにする
            for i in range(len(paragraphs) - 1):
                paragraph1 = paragraphs[i].strip()
                paragraph2 = paragraphs[i + 1].strip()

                if paragraph1 and paragraph2:
                    combined_paragraph = f"{title}の{i+1}段落目と{i+2}段落目: {paragraph1}\n\n{paragraph2}"
                    doc = Document(
                        page_content=combined_paragraph,
                        metadata={
                            "title": title,
                            "paragraph_numbers": f"{i+1}-{i+2}"
                        }
                    )
                    texts.append(doc)

        return texts