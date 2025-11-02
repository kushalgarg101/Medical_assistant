from typing import List, Iterable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.logger.logg import logs
from text_cleaner import clean_text

logger = logs("chunks.log")

def chunking(
    documents: Iterable[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
    do_clean: bool = True,
) -> List[Document]:
    """
    Partition documents into manageable chunks and assign a deterministic unique id
    to each chunk (stored in chunk.metadata["id"]).
    Steps:
      1. Split documents into chunks.
      2. Clean the chunk texts using clean_text (if do_clean=True).
      3. Assign deterministic chunk ids and return cleaned chunks.
    """
    try:
        documents_list = list(documents)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # 1) Split into raw chunks
        chunks_docs = text_splitter.split_documents(documents_list)
        logger.info("Number of chunks (raw): %d", len(chunks_docs))

        # 2) Clean the chunks (if desired)
        if do_clean:
            cleaned_chunks = clean_text(chunks_docs)
            # sanity check: ensure we still have same number of chunks
            if len(cleaned_chunks) != len(chunks_docs):
                logger.warning(
                    "clean_text changed number of chunks (%d -> %d). "
                    "Proceeding with cleaned list.",
                    len(chunks_docs),
                    len(cleaned_chunks),
                )
            chunks_docs = cleaned_chunks
        else:
            logger.debug("Skipping cleaning of chunks")

        # 3) Assign deterministic ids
        last_page = None
        current_chunk_idx = 0
        for chunk in chunks_docs:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            # Defensive formatting if metadata missing
            source_str = str(source) if source is not None else "unknown_source"
            page_str = str(page) if page is not None else "unknown_page"

            current_page_id = f"{source_str} : {page_str}"
            if current_page_id == last_page:
                current_chunk_idx += 1
            else:
                current_chunk_idx = 0

            chunk_id = f"{current_page_id}:{current_chunk_idx}"
            # Ensure metadata dict exists (Document.metadata should already be a dict)
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["id"] = chunk_id

            last_page = current_page_id

        logger.info("Number of chunks (final): %d", len(chunks_docs))
        return chunks_docs

    except Exception:
        logger.exception("Process of chunking failed")
        raise


if __name__ == "__main__":
    pass
    # from typing import List, Optional, Iterable
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    # from langchain_core.documents import Document

    # import sys
    # import os

    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # if project_root not in sys.path:
    #     sys.path.append(project_root)

    # from data import pdf_loader
    # from text_cleaner import clean_text
    # from src.config import settings
    # from src.logger.logg import logs

    # logger = logs('chunks.log')

    # def chunking(documents: Iterable[Document],
    #             chunk_size: int = 1024,
    #             chunk_overlap: int = 256,
    #             do_clean: bool = True) -> Optional[List[Document]]:
    #     """
    #     Partition documents into manageable chunks and assign a deterministic unique id
    #     to each chunk (stored in chunk.metadata["id"]).
    #     Steps:
    #     1. Split documents into chunks.
    #     2. Clean the chunk texts using clean_text (if do_clean=True).
    #     3. Assign deterministic chunk ids and return cleaned chunks.
    #     """
    #     try:
    #         documents_list = list(documents)

    #         text_splitter = RecursiveCharacterTextSplitter(
    #             chunk_size=chunk_size,
    #             chunk_overlap=chunk_overlap
    #         )

    #         # 1) Split into raw chunks
    #         chunks_docs = text_splitter.split_documents(documents_list)
    #         logger.info("Number of chunks (raw): %d", len(chunks_docs))

    #         # 2) Clean the chunks (if desired)
    #         if do_clean:
    #             cleaned_chunks = clean_text(chunks_docs)
    #             # sanity check: ensure we still have same number of chunks
    #             if len(cleaned_chunks) != len(chunks_docs):
    #                 logger.warning(
    #                     "clean_text changed number of chunks (%d -> %d). "
    #                     "Proceeding with cleaned list.",
    #                     len(chunks_docs), len(cleaned_chunks)
    #                 )
    #             chunks_docs = cleaned_chunks
    #         else:
    #             logger.debug("Skipping cleaning of chunks")

    #         # 3) Assign deterministic ids
    #         last_page = None
    #         current_chunk_idx = 0
    #         for chunk in chunks_docs:
    #             source = chunk.metadata.get("source")
    #             page = chunk.metadata.get("page")
    #             # Defensive formatting if metadata missing
    #             source_str = str(source) if source is not None else "unknown_source"
    #             page_str = str(page) if page is not None else "unknown_page"

    #             current_page_id = f"{source_str} : {page_str}"
    #             if current_page_id == last_page:
    #                 current_chunk_idx += 1
    #             else:
    #                 current_chunk_idx = 0

    #             chunk_id = f"{current_page_id}:{current_chunk_idx}"
    #             # Ensure metadata dict exists
    #             if chunk.metadata is None:
    #                 chunk.metadata = {}
    #             chunk.metadata["id"] = chunk_id

    #             last_page = current_page_id

    #         logger.info("Number of chunks (final): %d", len(chunks_docs))
    #         return chunks_docs

    #     except Exception:
    #         logger.exception("Process of chunking failed")
    #         return None

    # init_load = pdf_loader(settings.PDF_FILE_PATH)
    # ini_chunk = chunking(init_load)
    # if ini_chunk is not None:
    #     print(ini_chunk[123])