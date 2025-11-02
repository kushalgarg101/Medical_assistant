from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=r"D:\medicare\src\.env", extra="ignore", env_file_encoding="utf-8")

    # ----------------------------------- RAG configuration ------------------------------------------------

    # --- Data Configuration ---
    PDF_FILE_PATH: str = r"D:\medicare\Data\comprehensive-clinical-nephrology.pdf"

    # --- Embedding Configuration ---
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # --- GROQ Configuration ---
    GROQ_API_KEY: str 
    GROQ_LLM_MODEL: str = "qwen/qwen3-32b"

    # --- LANGCHAIN & RELATED Configuration ---
    TAVILY_API_KEY: str 

    # --- Comet ML & Opik Configuration ---
    COMET_API_KEY: str = Field("",description="API key for Comet ML and Opik services.")
    COMET_PROJECT: str = Field(
        default="Medic_chatbot",
        description="Project name for Comet ML and Opik tracking.",
    )

settings = Settings()

if __name__ == '__main__':
    pass
    # settings = Settings()
    # print(settings)