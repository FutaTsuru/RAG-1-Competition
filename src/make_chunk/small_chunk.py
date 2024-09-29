from langchain.text_splitter import CharacterTextSplitter

def make_small_chunk(documents, chunk_size: int, chunk_overlap: int, separator: str):
    """
    テキストを小さなチャンクに分割する関数
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    texts = text_splitter.split_documents(documents)

    splited_texts = []

    for text in texts:
        title = text.metadata['title']
        splited_text = f"{title}: {text.page_content}"
        splited_texts.append(splited_text)

    return splited_texts