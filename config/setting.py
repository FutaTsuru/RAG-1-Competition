novel_lists = [
    'カインの末裔.txt',
    'サーカスの怪人.txt',
    '芽生.txt',
    '競漕.txt',
    '死生に関するいくつかの断想.txt',
    '小説　不如帰.txt',
    '流行暗殺節.txt'
]

small_chunk_size = 500
small_chunck_overlap = 100
small_chunck_separator = "\n"

big_chunk_size = 30000
big_chunck_overlap = 4000
big_chunck_separator = "\n"

SYSTEM_PROMPT_PATH = "./config/system_prompt.md"
CHUNK_PATH = "./storage/chunk/chunk.csv"
CHUNK_EMBEDDINGS_PATH = './storage/embeddings/chunk_embeddings.npy'

embedding_model = "text-embedding-ada-002"
retrieval_num = 10
# similarity_threshold = 0.775
similarity_threshold = 0


model = "gpt-4o"
temperature=0.7