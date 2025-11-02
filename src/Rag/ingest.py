from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .embed import load_embed_model

client = QdrantClient(":memory:")

embeddings = load_embed_model()

vector_size = len(embeddings.embed_query("hello joe!"))

if not client.collection_exists("Medicare"):
    client.create_collection(
        collection_name="Medicare",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="Patient Reference Materials",
    embedding=embeddings,
)

if __name__ == '__main__':
    pass
    # from langchain_huggingface import HuggingFaceEmbeddings
    # from qdrant_client.models import Distance, VectorParams
    # from langchain_qdrant import QdrantVectorStore
    # from qdrant_client import QdrantClient
    # from typing import List, Optional, Union
    # from tqdm import tqdm

    # import sys
    # import os

    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # if project_root not in sys.path:
    #     sys.path.append(project_root)

    # from data import pdf_loader
    # from chunk_docs import chunking
    # from src.logger.logg import logs
    # from src.config import settings
    # from embed import load_embed_model

    # logger = logs('ingest.log')

    # # Load and chunk the PDF
    # init_load = pdf_loader(settings.PDF_FILE_PATH)
    # init_chunker = chunking(init_load)
    # logger.info('Number of chunks received to be converted to embeddings: %i', len(init_chunker))
    
    # client = QdrantClient(path=r"D:\medicare\src\infrastructure")

    # embeddings = load_embed_model()

    # vector_size = len(embeddings.embed_query("hello joe!"))  # for obtaining embedding vector size

    # if not client.collection_exists("Medicare"):
    #     client.create_collection(
    #         collection_name="Medicare",
    #         vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    #     )

    # vector_store = QdrantVectorStore(
    #     client=client,
    #     collection_name="Medicare",
    #     embedding=embeddings,
    # )

    # # Progress bar with silent loop (no prints or logs inside to prevent tqdm conflict)
    # with tqdm(total=len(init_chunker), desc="Embedding Chunks", dynamic_ncols=True) as pbar:
    #     for idx, chunk in enumerate(init_chunker):
    #         vector_store.add_documents([chunk])

    #         # Log only every 100th chunk to reduce noise
    #         if (idx + 1) % 100 == 0:
    #             logger.info("Processed %i chunks so far...", idx + 1)

    #         pbar.update(1)

    # # Final log
    # logger.info("Processing complete! Total chunk embeddings processed: %i", len(init_chunker))