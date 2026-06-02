import chromadb
from sentence_transformers import SentenceTransformer
client = chromadb.PersistentClient(
    path="data/index/chroma"
)
collection = client.get_collection("arxiv_chunks")
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device="mps"
)

query = "What is ReAct prompting?"
emb = model.encode(query).tolist()
res = collection.query(
    query_embeddings=[emb],
    n_results=5
)
for i, doc in enumerate(res["documents"][0]):
    print("\n" + "="*80)
    print(f"Result {i+1}")
    print(doc[:1000])