"""API HTTP local para o projeto RAG.

Execute diretamente com:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src.monitor import summarize_logs
from src.rag_chain import CorporateRAG
from src.retriever import get_vector_store_status


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Project RAG API",
    description="API local para consultas ao LangGraph com Redis Vector Search.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Contrato de entrada da consulta RAG."""

    question: str = Field(min_length=3, max_length=2000)
    doc_type: str | None = None
    k: int = Field(default=4, ge=1, le=10)


@lru_cache(maxsize=1)
def get_rag() -> CorporateRAG:
    """Reutiliza uma única instância do RAG entre as requisições."""
    return CorporateRAG()


def serialize_document(document: Any) -> dict[str, Any]:
    """Converte o documento recuperado para um objeto JSON."""
    if isinstance(document, dict):
        return document

    return {
        "score": getattr(document, "score", 0),
        "metadata": getattr(document, "metadata", {}),
        "content": getattr(document, "content", ""),
    }


def serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normaliza apenas os objetos que não são serializáveis diretamente."""
    response = dict(result)
    response["retrieved_docs"] = [
        serialize_document(document)
        for document in result.get("retrieved_docs", [])
    ]
    return jsonable_encoder(response)


@app.get("/health")
def health() -> dict[str, str]:
    """Confirma que o processo HTTP esta ativo."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Confirma que o RAG pode ser inicializado."""
    try:
        get_rag()
        return {"status": "ready", "rag": "initialized"}
    except Exception as exc:
        logger.exception("Falha ao inicializar o RAG")
        raise HTTPException(
            status_code=503,
            detail="RAG is unavailable",
        ) from exc


@app.get("/api/v1/monitor/summary")
def monitor_summary() -> dict[str, Any]:
    """Devolve as métricas produzidas pelo monitor atual do projeto."""
    try:
        return jsonable_encoder(summarize_logs())
    except Exception as exc:
        logger.exception("Falha ao consultar as metricas")
        raise HTTPException(
            status_code=500,
            detail="Unable to load monitoring summary",
        ) from exc


@app.get("/api/v1/vector-store/status")
def vector_store_status() -> dict[str, Any]:
    """Informa o backend ativo e a quantidade real de chunks indexados."""
    return jsonable_encoder(get_vector_store_status())


@app.post("/api/v1/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Executa o fluxo completo do RAG e devolve resposta, metricas e fontes."""
    try:
        result = get_rag().ask(
            request.question,
            doc_type=request.doc_type,
            k=request.k,
        )
        return serialize_result(result)
    except Exception as exc:
        logger.exception("Falha ao executar a consulta RAG")
        raise HTTPException(
            status_code=500,
            detail="RAG query failed",
        ) from exc
