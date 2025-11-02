from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document
from typing import List

from src.logger.logg import logs

logger = logs("data.log")


def pdf_loader(file_path: str) -> List[Document]:
    """
    Loads a PDF file and returns its pages.
    """
    try:
        pages = []
        loader = PyPDFLoader(file_path)
        if not loader:
            logger.error("No pages returned by PyPDFLoader for file: %s", file_path)
            raise Exception(f"No content loaded from PDF: {file_path}")
        for page in loader.load():
            pages.append(page)
        logger.info(
            "Successfully loaded PDF: %s -------------\n---------------\n", file_path
        )
        logger.info(
            "Number of pages in the provided PDF: %d -------------\n---------------\n",
            len(pages),
        )
        return pages

    except Exception as e:
        logger.exception(
            "An unexpected error occurred while loading the PDF: %s -------------\n---------------\n",
            e,
        )
        raise


if __name__ == "__main__":
    pass

    # import sys
    # import os

    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # if project_root not in sys.path:
    #     sys.path.append(project_root)

    # from langchain_community.document_loaders.pdf import PyPDFLoader
    # from langchain_core.documents import Document
    # from typing import Optional, List

    # from src.logger.logg import logs
    # from src.config import settings
    # from text_cleaner import clean_text

    # logger = logs('data.log')
    # def pdf_loader(file_path: str) -> Optional[List[Document]]:
    #     """
    #     Loads a PDF file and returns its pages.
    #     """

    #     if not os.path.exists(file_path):
    #         logger.error("File not found: %s", file_path)
    #         raise FileNotFoundError(f"File does not exist: {file_path}")
    #     try:
    #         pages = []
    #         loader = PyPDFLoader(file_path)
    #         if not loader:
    #             logger.error("No pages returned by PyPDFLoader for file: %s", file_path)
    #             raise Exception(f"No content loaded from PDF: {file_path}")
    #         for page in loader.load():
    #             pages.append(page)
    #         logger.info('Successfully loaded PDF: %s -------------\n---------------\n', file_path)
    #         logger.info('Number of pages in the provided PDF: %d -------------\n---------------\n', len(pages))
    #         return pages

    #     except Exception as e:
    #         logger.exception('An unexpected error occurred while loading the PDF: %s -------------\n---------------\n', e)

    #     return None

    # init_loader = pdf_loader(settings.PDF_FILE_PATH)
    # if init_loader is not None:
    #     init_cleaner = clean_text(init_loader)
    #     logger.info('Sample document from provided pdf : %s -------------\n---------------\n', init_cleaner)
    # else:
    #     print('The defined function has returned None.')