import os
from langchain_community.document_loaders import TextLoader

def import_documents(novel_lists):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    documents = []

    for novel in novel_lists:
        novel_title = novel.replace(".txt", "")
        file_path = os.path.join(script_dir, '..', '..', 'novels', 'works', novel)
        loader = TextLoader(file_path, encoding='utf-8')
        novel_documents = loader.load()
        novel_documents[0].metadata['title'] = novel_title  # タイトルをメタデータに追加
        documents += novel_documents
    
    return documents