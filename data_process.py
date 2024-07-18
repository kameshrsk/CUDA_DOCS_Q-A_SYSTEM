import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import nltk
import numpy as np

from config import (MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, MAX_CHUNK_LENGTH, SENTENCE_TRANSFORMER_MODEL, OUTPUT_FILE, MAX_CHUNK_VARCHAR_LENGTH)

def load_data(path:str):

    with open(path, 'r') as f:
        return json.load(f)
    

def optimal_cluster(embeddings, max_clusters=10):
    if len(embeddings)<=2: return 1

    silhouette_scores=[]

    max_clusters=min(max_clusters, len(embeddings)-1)

    for i in range(2, max_clusters+1):
        kmeans=KMeans(n_clusters=i, random_state=101)
        cluster_labels=kmeans.fit_predict(embeddings)
        avg=silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(avg)

    optimal_clusters=silhouette_scores.index(max(silhouette_scores))+2
    
    return optimal_clusters

def chunk_data(text, model, max_chunk_length=MAX_CHUNK_VARCHAR_LENGTH):

    sentences=nltk.sent_tokenize(text)
    embeddings=model.encode(sentences)

    n_clusters=optimal_cluster(embeddings)

    kmeans=KMeans(n_clusters=n_clusters, random_state=101)

    kmeans.fit(embeddings)

    chunks=[[] for _ in range(n_clusters)]

    for i, label in enumerate(kmeans.labels_):
        chunks[label].append(sentences[i])

    final_chunks=[]

    for chunk in chunks:
        chunk_text=' '.join(chunk)

        while len(chunk_text)>max_chunk_length:
            split_index=chunk_text[:max_chunk_length].rfind(' ')
            if split_index==-1:
                split_index=max_chunk_length
            final_chunks.append(chunk_text[:split_index])
            chunk_text=chunk_text[split_index :]

        final_chunks.append(chunk_text)

    return final_chunks

def create_milvus_collection():

    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    fields=[
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=250),
        FieldSchema(name='chunk', dtype=DataType.VARCHAR, max_length=MAX_CHUNK_VARCHAR_LENGTH)
    ]

    schema=CollectionSchema(fields=fields, description="Cuda Docs Chunks")

    collection=Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params":{"M":16, "efConstruction":500}
    }

    collection.create_index(field_name='embeddings', index_params=index_params)

    return collection

def process_and_store():

    model=SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    data=load_data(OUTPUT_FILE)
    collection=create_milvus_collection()

    for i, item in enumerate(data):
        if i==600: break # breaking after pushing 600 pages for computational purpose
        if item['content']=='': continue

        chunks=chunk_data(item['content'], model)

        for chunk in chunks:
            if len(chunk)>MAX_CHUNK_LENGTH:
                chunk=chunk[:MAX_CHUNK_LENGTH]

            embeddings=model.encode(chunk).tolist()
            collection.insert([
                [embeddings],
                [item['url']],
                [chunk]
            ])
        if i+1%50==0:
            print(f"Chunked and Pushed {i+1} page")

    collection.flush()
    print(f"Inserted {collection.num_entities} entities")

if __name__=="__main__":
    process_and_store()