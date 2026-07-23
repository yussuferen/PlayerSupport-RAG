import os
import json
import chromadb
import ollama

DB_PATH = "./chroma_db"
COLLECTION_NAME = "polygames_qa_support"
JSON_PATH = "polygames_qa.json"

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
# RETRIEVAL & GENERATION
# ==========================================
def ask_rag(query, collection,top_k=3):
    """Retrieves context from ChromaDB and generates an answer using Llama 3.2."""
    print(f"\nSearching context for query: '{query}'")

    # Embed the user query
    query_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )["embedding"]

    # Retrieve top 3 most relevant QA pairs
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    context = "\n\n---\n\n".join(results["documents"][0])

    # 🐛 DEBUG:
    print("\n--- [DEBUG] RETRIEVED CONTEXT FROM CHROMADB ---")
    print(context)
    print("------------------------------------------------\n")

    prompt = f"""You are an AI support assistant for PolyGames studio.
Use the provided Context information below to answer the user's question clearly and politely.
If the answer cannot be found in the context, do NOT hallucinate or invent information. Simply reply: "This information is not available in our support center, please contact live support."

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
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

    print("\nPolyGames Player Support Assistant Ready! (Type 'q' or 'exit' to quit)\n")

    while True:
        user_query = input("\nAsk a Question: ")
        if user_query.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break

        if user_query.strip():
            answer = ask_rag(user_query, collection)
            print("\nResponse:")
            print(answer)
            print("=" * 60)