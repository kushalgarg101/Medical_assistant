from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional

from .embed import load_embed_model
from src.logger.logg import logs

logger = logs("retrieve.log")

def make_client(path: str = r"D:\medicare\src\infrastructure") -> QdrantClient:
    """Initialize and return a local Qdrant client."""
    try:
        client = QdrantClient(path=path)
        logger.info(f"Connected to Qdrant at {path}")
        return client
    except Exception as e:
        logger.error(f"Failed to create Qdrant client: {e}")
        raise


def get_collection(client: QdrantClient, collection_name: str = "Medicare"):
    """Return Qdrant collection metadata if it exists."""
    try:
        return client.get_collection(collection_name)
    except Exception as e:
        logger.warning(f"Collection '{collection_name}' not found or inaccessible: {e}")
        return None

def get_vector_store(
    client: QdrantClient,
    collection_name: str = "Medicare",
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> QdrantVectorStore:
    """Create a QdrantVectorStore for similarity search."""
    if embeddings is None:
        embeddings = load_embed_model()
        logger.info("Loaded embedding model via load_embed_model()")

    try:
        vs = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
        logger.info(f"Vector store ready for collection '{collection_name}'")
        return vs
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise

def retrieve_context(query: str, vector_store: QdrantVectorStore, top_k: int = 5):
    """
    Retrieve top-k most similar documents for a given query,
    returning text, score, and source citation if available.
    """
    try:
        results = vector_store.search(query, search_type="similarity", limit=top_k)
        formatted = []
        for res in results:
            # LangChain Document-like object
            text = getattr(res, "page_content", "") if hasattr(res, "page_content") else str(res)
            meta = getattr(res, "metadata", {}) or {}
            score = getattr(res, "score", None)
            citation = meta.get("source") or meta.get("filename") or meta.get("url") or "unknown"
            formatted.append({"text": text, "score": score, "citation": citation})
        logger.info(f"Retrieved {len(formatted)} results for query: '{query}'")
        return formatted
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []

def test_loop():
    """Simple CLI test loop to interactively check retrieval."""
    client = make_client()
    vs = get_vector_store(client)
    print("Type a query (empty to quit):\n")

    while True:
        query = input("Query> ").strip()
        if not query:
            break
        results = retrieve_context(query, vs)
        if not results:
            print("No results found.")
            continue
        for i, r in enumerate(results, 1):
            snippet = r["text"][:200].replace("\n", " ")
            print(f"\n[{i}] score={r['score']} | source={r['citation']}\n  {snippet}\n")


if __name__ == "__main__":
    pass
    # from langchain_qdrant import QdrantVectorStore
    # from qdrant_client import QdrantClient
    # from langchain_huggingface import HuggingFaceEmbeddings
    # from typing import Optional

    # import sys
    # import os

    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # if project_root not in sys.path:
    #     sys.path.append(project_root)
    
    # from embed import load_embed_model
    # from src.logger.logg import logs

    # logger = logs("retrieve.log")

    # def make_client(path: str = r"D:\medicare\src\infrastructure") -> QdrantClient:
    #     """Initialize and return a local Qdrant client."""
    #     try:
    #         client = QdrantClient(path=path)
    #         logger.info(f"Connected to Qdrant at {path}")
    #         return client
    #     except Exception as e:
    #         logger.error(f"Failed to create Qdrant client: {e}")
    #         raise


    # def get_collection(client: QdrantClient, collection_name: str = "Medicare"):
    #     """Return Qdrant collection metadata if it exists."""
    #     try:
    #         return client.get_collection(collection_name)
    #     except Exception as e:
    #         logger.warning(f"Collection '{collection_name}' not found or inaccessible: {e}")
    #         return None


    # def get_vector_store(
    #     client: QdrantClient,
    #     collection_name: str = "Medicare",
    #     embeddings: Optional[HuggingFaceEmbeddings] = None,
    # ) -> QdrantVectorStore:
    #     """Create a QdrantVectorStore for similarity search."""
    #     if embeddings is None:
    #         embeddings = load_embed_model()
    #         logger.info("Loaded embedding model via load_embed_model()")

    #     try:
    #         vs = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    #         logger.info(f"Vector store ready for collection '{collection_name}'")
    #         return vs
    #     except Exception as e:
    #         logger.error(f"Failed to initialize vector store: {e}")
    #         raise


    # def retrieve_context(query: str, vector_store: QdrantVectorStore, top_k: int = 5):
    #     """
    #     Retrieve top-k most similar documents for a given query,
    #     returning text, score, and source citation if available.
    #     """
    #     try:
    #         results = vector_store.search(query, search_type="similarity", limit=top_k)
    #         formatted = []
    #         for res in results:
    #             # LangChain Document-like object
    #             text = getattr(res, "page_content", "") if hasattr(res, "page_content") else str(res)
    #             meta = getattr(res, "metadata", {}) or {}
    #             score = getattr(res, "score", None)
    #             citation = meta.get("source") or meta.get("filename") or meta.get("url") or "unknown"
    #             formatted.append({"text": text, "score": score, "citation": citation})
    #         logger.info(f"Retrieved {len(formatted)} results for query: '{query}'")
    #         return formatted
    #     except Exception as e:
    #         logger.error(f"Error during retrieval: {e}")
    #         return []


    # def test_loop():
    #     """Simple CLI test loop to interactively check retrieval."""
    #     client = make_client()
    #     vs = get_vector_store(client)
    #     print("Type a query (empty to quit):\n")

    #     while True:
    #         query = input("Query> ").strip()
    #         if not query:
    #             break
    #         results = retrieve_context(query, vs)
    #         if not results:
    #             print("No results found.")
    #             continue
    #         for i, r in enumerate(results, 1):
    #             snippet = r["text"][:200].replace("\n", " ")
    #             print(f"\n[{i}] score={r['score']} | source={r['citation']}\n  {snippet}\n")

    # test_loop()
