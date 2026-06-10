import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_COLLECTION, CHROMA_PATH, EMBEDDING_MODEL, N_RESULTS

# Embedding function and ChromaDB client are initialized once at module load.
# sentence-transformers downloads the model on first use — this may take
# 30–60 seconds the very first time. Subsequent runs use a local cache.
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=_ef,
    metadata={"hnsw:space": "cosine"},
)

def get_collection():
    return _collection

def embed_and_store(chunks):
    _collection.add(
        documents = [c["text"] for c in chunks],
        metadatas= [{"prof": c["prof"]} for c in chunks],
        ids=[c["chunk_id"] for c in chunks]
    )
    print(f"Stored {_collection.count()} total chunks in the vector database.")

def retrieve(query, n_results=N_RESULTS):
    if _collection.count == 0:
        return []
    
    results = _collection.query(
        query_texts= query,
        n_results= n_results,
        include= ["documents", "metadatas", "distances"]
    )

    documents = documents["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = [
        {
            "text": doc,
            "prof": meta["prof"],
            "distance": dist
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]

    # Temporary debug print — remove after verifying
    for chunk in chunks:
        print(f"[{chunk['prof']}] (dist: {chunk['distance']:.3f}) {chunk['text'][:80]}...")
    print("\n")
    print("--------")

    return chunks