"""
embeddings_factory.py

Cria o modelo de embeddings usado pelo ChromaDB.

Permite usar:
- OllamaEmbeddings
- HuggingFaceEmbeddings

Se você não quer depender do Ollama, use:
EMBEDDING_PROVIDER=huggingface
"""

from src.config import get_settings


def get_embeddings():
    """Retorna o modelo de embeddings configurado no .env."""

    settings = get_settings()

    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model
        )

    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"EMBEDDING_PROVIDER inválido: {settings.embedding_provider}. "
        "Use 'huggingface' ou 'ollama'."
    )