"""
llm_factory.py

Cria o LLM usado pelo projeto.

Permite alternar entre:
- Ollama local - O Ollama apresentou uma demora na respostas muito alta Talez relacionado ao LLM usado (Gemma4)
- Groq API << Utilizado
"""

import os

from src.config import get_settings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

def get_llm():
    """Retorna o modelo de linguagem configurado no .env."""

    settings = get_settings()

    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY não foi configurada no arquivo .env."
            )

        os.environ["GROQ_API_KEY"] = settings.groq_api_key

        return ChatGroq(
            model=settings.groq_model,
            temperature=0,
            max_retries=2,
        )

    if settings.llm_provider == "ollama":
        
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    raise ValueError(
        f"LLM_PROVIDER inválido: {settings.llm_provider}. Use 'groq' ou 'ollama'."
    )