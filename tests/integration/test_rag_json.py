"""Testes de integração do retorno do RAG.

Estes testes exercitam o fluxo real definido em ``src.rag_chain.CorporateRAG``:
LangGraph, Redis Retriever e o provedor de LLM configurado no ambiente.

O método ``CorporateRAG.ask`` retorna um ``dict``, mas o campo
``retrieved_docs`` contém objetos ``RetrievedDocument``. A função
``_to_json_response`` converte esses objetos para uma representação JSON-safe
antes das validações.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.rag_chain import CorporateRAG


QUERY = "Informe sobre Invoice Order ID: 10512"


def _document_to_source(document: Any) -> dict[str, Any]:
    """Converte um RetrievedDocument em um dicionário serializável."""

    metadata = dict(getattr(document, "metadata", {}) or {})
    score = getattr(document, "score", metadata.get("score", 0.0))
    source_label = getattr(document, "source_label", None)

    source = {
        **metadata,
        "score": float(score),
    }

    if source_label:
        source["source_label"] = str(source_label)

    return source


def _to_json_response(result: dict[str, Any]) -> dict[str, Any]:
    """Seleciona o contrato público do RAG e o transforma em JSON-safe."""

    return {
        "answer": str(result.get("answer", "")),
        "fallback": bool(result.get("fallback", False)),
        "avg_similarity": float(result.get("avg_similarity", 0.0)),
        "latency_ms": float(result.get("latency_ms", 0.0)),
        "warning": str(result.get("warning", "")),
        "security_blocked": bool(result.get("security_blocked", False)),
        "sources": [
            _document_to_source(document)
            for document in result.get("retrieved_docs", [])
        ],
    }


@pytest.fixture(scope="module")
def order_response() -> dict[str, Any]:
    """Executa o RAG uma única vez para todos os testes deste módulo."""

    rag = CorporateRAG()
    result = rag.ask(QUERY)
    response = _to_json_response(result)

    # Exibido somente quando o Pytest é executado com a opção -s.
    print(json.dumps(response, indent=2, ensure_ascii=False))

    return response


def test_response_is_valid_json(order_response: dict[str, Any]) -> None:
    """Garante que o contrato completo pode ser serializado como JSON."""

    serialized = json.dumps(order_response, ensure_ascii=False)
    deserialized = json.loads(serialized)

    assert isinstance(deserialized, dict)
    assert deserialized == order_response


def test_response_contract(order_response: dict[str, Any]) -> None:
    """Valida os campos públicos e seus tipos."""

    expected_fields = {
        "answer",
        "fallback",
        "avg_similarity",
        "latency_ms",
        "warning",
        "security_blocked",
        "sources",
    }

    assert expected_fields == set(order_response)
    assert isinstance(order_response["answer"], str)
    assert isinstance(order_response["fallback"], bool)
    assert isinstance(order_response["avg_similarity"], float)
    assert isinstance(order_response["latency_ms"], float)
    assert isinstance(order_response["sources"], list)


def test_order_10386_has_supported_answer(
    order_response: dict[str, Any],
) -> None:
    """A consulta conhecida deve produzir resposta, evidência e não usar fallback."""

    assert order_response["answer"].strip()
    assert order_response["fallback"] is False
    assert order_response["security_blocked"] is False
    assert order_response["latency_ms"] > 0
    assert 0.0 <= order_response["avg_similarity"] <= 1.0
    assert order_response["sources"]


def test_order_10386_uses_exact_match(
    order_response: dict[str, Any],
) -> None:
    """Valida a recuperação exata observada para o order_id 10512."""

    exact_sources = [
        source
        for source in order_response["sources"]
        if source.get("retrieval_type") == "exact_match"
        and str(source.get("matched_identifier")) == "10512"
    ]

    assert exact_sources, (
        "Nenhuma fonte exact_match foi encontrada para o identificador 10512."
    )

    assert any(
        str(source.get("doc_type", "")).casefold() == "purchase order"
        for source in exact_sources
    )
    assert any(
        source.get("source") == "company-document-text.csv"
        or source.get("source_file") == "company-document-text.csv"
        for source in exact_sources
    )

    for source in exact_sources:
        assert source.get("chunk_id") is not None
        assert source.get("redis_id")

