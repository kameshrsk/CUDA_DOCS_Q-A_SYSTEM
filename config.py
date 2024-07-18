# config.py

# Milvus Configuration
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "Cuda_docs"
MAX_CHUNK_VARCHAR_LENGTH = 65535

# Data Processing
MAX_CHUNK_LENGTH = 50000
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

# Web Crawling
ALLOWED_DOMAINS = ['docs.nvidia.com']
START_URLS = ['https://docs.nvidia.com/cuda/']
DEPTH_LIMIT = 5
OUTPUT_FILE = 'output/cuda_docs.json'

# Question Answering
QA_MODEL_NAME = "t5-small"
MAX_ANSWER_LENGTH = 250

# Retrieval
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 10
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"