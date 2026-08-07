import os
import json
import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

DB_PATH = "./chroma_db"
COLLECTION_NAME = "polygames_qa_support"
JSON_PATH = "polygames_qa.json"

RERANKER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ==========================================
# VECTOR DB
# ==========================================
def load_vectordb(client):
    print("Loading existing ChromaDB database from disk...")
    return client.get_collection(name=COLLECTION_NAME)

def create_vectordb(client):
    print("Initial setup: Processing JSON dataset and generating embeddings...")

    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(
            f"Error: '{JSON_PATH}' file not found! Please place it in the project root directory."
        )

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    for item in qa_dataset:
        # Structure each QA pair as an atomic document entry
        document_text = f"Category: {item['category']}\nQuestion: {item['question']}\nAnswer: {item['answer']}"

        # Embedding using Ollama
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=document_text
        )

        # Store in ChromaDB with metadata for potential filtering
        collection.add(
            ids=[item["id"]],
            embeddings=[response["embedding"]],
            documents=[document_text],
            metadatas=[{
                "category": item["category"],
                "tags": ", ".join(item.get("tags", []))
            }]
        )

    print(f"Successfully saved {len(qa_dataset)} QA entries to ChromaDB ('{DB_PATH}')!")
    return collection

def get_vector_db():
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    existing_collections = [c.name for c in chroma_client.list_collections()]

    if COLLECTION_NAME in existing_collections:
        return load_vectordb(chroma_client)
    return create_vectordb(chroma_client)
# ==========================================
# SEARCH
# ==========================================
def vector_search(query, collection, top_k):
    query_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

def keyword_search(query, bm25, documents, top_k):
    tokenized_query = query.lower().split()
    return bm25.get_top_n(tokenized_query, documents, n=top_k)

def combine_results(v_docs, k_docs):
    return list(dict.fromkeys(v_docs + k_docs))

# ==========================================
# BM25
# ==========================================
def build_bm25_index(collection):
    all_data = collection.get(include=["documents"])
    all_docs = all_data["documents"]

    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, all_docs
# ==========================================
# RERANKER
# ==========================================
def rerank_docs(query, candidate_docs, final_top_k=3):
    pairs = [[query, doc] for doc in candidate_docs]
    scores = RERANKER_MODEL.predict(pairs)
    scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:final_top_k]]

# ==========================================
# RETRIEVAL & GENERATION
# ==========================================
def ask_rag(query, collection,bm25,documents,top_k=3):
    """Retrieves context from ChromaDB and generates an answer using Llama 3.2."""
    print(f"\nSearching context for query: '{query}'")

    v_results = vector_search(query,collection,top_k)
    k_results = keyword_search(query,bm25,documents,top_k)

    cand_docs = combine_results(v_results,k_results)
    final_docs = rerank_docs(query,cand_docs,top_k);

    context = "\n\n---\n\n".join(final_docs)

    system_instruction = (
        "You are an AI support assistant for PolyGames studio.\n"
        "Use the provided Context information below to answer the user's question clearly and politely.\n"
        "If the answer cannot be found in the context, do NOT hallucinate or invent information. "
        'Simply reply: "This information is not available in our support center, please contact live support."'
    )

    user_content = f"""
    Context:
    {context}
    
    Question: 
    {query}"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role":"system","content":system_instruction},{"role": "user", "content": user_content}]
    )

    return response["message"]["content"]


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(JSON_PATH):
        print(f"Error: File '{JSON_PATH}' not found.")
        exit()

    collection = get_vector_db()
    bm25, documents = build_bm25_index(collection)
    
    print("\nPolyGames Player Support Assistant Ready! (Type 'q' or 'exit' to quit)\n")

    while True:
        user_query = input("\nAsk a Question: ")
        if user_query.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break

        if user_query.strip():
            answer = ask_rag(user_query, collection,bm25,documents)
            print("\nResponse:")
            print(answer)
            print("=" * 60)