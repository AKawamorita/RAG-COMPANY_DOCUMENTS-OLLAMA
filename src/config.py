from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Configurações centrais do projeto RAG."""

    # Banco vetorial: "chroma" ou "redis"
    vector_store_provider: str

    # Provedor do LLM: "ollama" ou "groq"
    llm_provider: str

    # Ollama
    ollama_base_url: str
    llm_model: str
    embedding_model: str

    # Groq
    groq_api_key: str
    groq_model: str

    # Embeddings
    embedding_provider: str
    hf_embedding_model: str

    # ChromaDB
    chroma_collection_name: str
    chroma_persist_dir: Path

    # Redis Vector Search
    redis_url: str
    redis_index_name: str
    redis_key_prefix: str

    # Limite visual usado para classificar a quantidade de chunks indexados.
    vector_store_attention_limit: int

    # Dados e logs
    raw_data_dir: Path
    logs_dir: Path

    # Chunking e retrieval
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    min_relevance_score: float


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def get_settings() -> Settings:
    """Carrega o arquivo .env e devolve as configurações do projeto."""

    load_dotenv(BASE_DIR / ".env")

    return Settings(
        vector_store_provider=os.getenv(
            "VECTOR_STORE_PROVIDER",
            "chroma"
        ).lower(),

        llm_provider=os.getenv("LLM_PROVIDER", "ollama").lower(),

        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_model=os.getenv("OLLAMA_LLM_MODEL", "gemma4:e4b"),
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),

        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),

        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama").lower(),
        hf_embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        ),

        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            "company_documents_rag"
        ),
        chroma_persist_dir=BASE_DIR / os.getenv(
            "CHROMA_PERSIST_DIR",
            "data/chroma"
        ),

        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        redis_index_name=os.getenv(
            "REDIS_INDEX_NAME",
            "company_documents_rag"
        ),
        redis_key_prefix=os.getenv(
            "REDIS_KEY_PREFIX",
            "company_documents_rag"
        ),
        vector_store_attention_limit=_get_int(
            "VECTOR_STORE_ATTENTION_LIMIT",
            100,
        ),

        raw_data_dir=BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw"),
        logs_dir=BASE_DIR / os.getenv("LOGS_DIR", "logs"),

        chunk_size=_get_int("CHUNK_SIZE", 900),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
        retrieval_k=_get_int("RETRIEVAL_K", 4),
        min_relevance_score=_get_float("MIN_RELEVANCE_SCORE", 0.35),
    )
