from langchain.text_splitter import CharacterTextSplitter

def make_big_chunck(documents, chunk_size: int, chunk_overlap: int, separator: str):
    """
    各小説の全文もベクトル化する関数。(gpt-4oが処理する最大トークンを超えそうな小説は分割する。)
    """
    texts = []
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
    return texts