from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List

from src.logger.logg import logs
from src.config import settings

logger = logs("embed.log")

def load_embed_model(model_name: str = settings.EMBEDDING_MODEL_NAME):
    """
    Loads an embedding model from langchain hugging face module.
    Args :
        model_name(default = all-MiniLM-L6-v2)
    """

    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    logger.info("Embedding Model successfully loaded: %s", model_name)

    return embedding_model


def embed_chunks(
    chunks: str, model: HuggingFaceEmbeddings = load_embed_model()
) -> List[List[float]]:
    """
    Converts text to vector embeddings using the specified model.
    """
    try:
        if not chunks or not isinstance(chunks, str):
            raise ValueError("Input 'chunks' must be a non-empty string")

        text_embeddings = model.embed_documents([chunks])

        if not text_embeddings or not isinstance(text_embeddings, list):
            raise ValueError("Embedding model returned no embeddings")

        return text_embeddings

    except Exception as e:
        logger.error("Embedding failed: %s", str(e), exc_info=True)
        return []


if __name__ == "__main__":
    pass
    # from langchain_huggingface import HuggingFaceEmbeddings
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

    # logger = logs('embed.log')

    # def load_embed_model(model_name : str = settings.EMBEDDING_MODEL_NAME):
    #     embedding_model = HuggingFaceEmbeddings(model_name = model_name)
    #     logger.info('Embedding Model successfully loaded: %s', model_name)

    #     return embedding_model

    # def embed_chunks(chunks: str, model: HuggingFaceEmbeddings =load_embed_model()) -> List[List[float]]:
    #     """
    #     Converts text to vector embeddings using the specified model.
    #     """
    #     try:
    #         if not chunks or not isinstance(chunks, str):
    #             raise ValueError("Input 'chunks' must be a non-empty string")

    #         text_embeddings = model.embed_documents([chunks])

    #         if not text_embeddings or not isinstance(text_embeddings, list):
    #             raise ValueError("Embedding model returned no embeddings")

    #         return text_embeddings

    #     except Exception as e:
    #         logger.error("Embedding failed: %s", str(e), exc_info=True)
    #         return []

    # # Load and chunk the PDF
    # init_load = pdf_loader(settings.PDF_FILE_PATH)
    # init_chunker = chunking(init_load)
    # logger.info('Number of chunks received to be converted to embeddings: %i', len(init_chunker))

    # # Progress bar with silent loop (no prints or logs inside to prevent tqdm conflict)
    # with tqdm(total=len(init_chunker), desc="Embedding Chunks", dynamic_ncols=True) as pbar:
    #     for idx, chunk in enumerate(init_chunker):
    #         embed_chunks(chunk.page_content)

    #         # Log only every 100th chunk to reduce noise
    #         if (idx + 1) % 100 == 0:
    #             logger.info("Processed %i chunks so far...", idx + 1)

    #         pbar.update(1)

    # # Final log
    # logger.info("Processing complete! Total chunk embeddings processed: %i", len(init_chunker))
