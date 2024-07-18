import torch
import nltk
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import (MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, SENTENCE_TRANSFORMER_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANK, RERANKER_MODEL)

class RetrievalSystem:
    def __init__(self, host=MILVUS_HOST, port=MILVUS_PORT):
        print("Creating Connection to the Milvus DB...")
        connections.connect("default", host=host, port=port)
        self.collection=Collection(COLLECTION_NAME)
        self.collection.load()

        print("Loading Sentence Transformer...")
        self.model=SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

        print("Loading Ranker Models...")
        self.reranker_tokenizer=AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.reranker_model=AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)

    def tokenize(self, text):
        return nltk.word_tokenize(text)

    def retrieval(self, query, top_k=TOP_K_RETRIEVAL):

        question_embeddings=self.model.encode(query, convert_to_tensor=True)
        search_params = {"metric_type": "L2", "params": {"ef": 250}, "offset": 0}

        results=self.collection.search(question_embeddings.cpu().numpy().reshape(1,-1), "embeddings", search_params, top_k,output_fields=["chunk", "url"])

        docs=[hit.entity.get("chunk") for hit in results[0]]
        tokenized_corpus=[self.tokenize(doc) for doc in docs]
        bm25=BM25Okapi(tokenized_corpus)

        vector_score=[hit.score for hit in results[0]]
        bm25_score=bm25.get_scores(self.tokenize(query))

        combined_score=[vs+bs for vs, bs in zip(vector_score, bm25_score)]
        sorted_results=sorted(zip(docs, combined_score), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in sorted_results]

    def rerank(self, query, docs, top_k=TOP_K_RERANK):
        pairs=[[query, doc] for doc in docs]
        inputs=self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)

        with torch.no_grad():
            scores=self.reranker_model(**inputs).logits.squeeze(-1)

        reranked=sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [docs for docs, _ in reranked[:top_k]]

    def retrieve_and_rerank(self, query, top_k=10):
        retrieved_docs=self.retrieval(query)
        reranked_docs=self.rerank(query, retrieved_docs)

        return reranked_docs

if __name__=="__main__":
    retrieval_system=RetrievalSystem()

    query="How does CUDA Manages Memeory?"

    results=retrieval_system.retrieve_and_rerank(query)

    for i, result in enumerate(results):
        print(f"{i+1}: {result}")