"""Monitoramento simples do RAG.

O objetivo aqui não é substituir ferramentas profissionais de observabilidade,
mas demonstrar pensamento de MLOps/RAGOps em um projeto de portfólio.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.config import get_settings


LOG_FILE_NAME = "rag_queries.jsonl"


def estimate_tokens(text: str) -> int:
    """Estimativa simples de tokens.

    Para portfólio, uma aproximação comum é considerar cerca de 4 caracteres por
    token em textos ocidentais. Não substitui contadores específicos de modelo.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def get_log_path() -> Path:
    """Retorna o caminho do arquivo de logs."""
    settings = get_settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings.logs_dir / LOG_FILE_NAME


def log_query(event: dict[str, Any]) -> None:
    """Registra uma consulta em formato JSONL."""
    log_path = get_log_path()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_logs(log_path: Path | None = None) -> list[dict[str, Any]]:
    """Lê os logs gravados."""
    log_path = log_path or get_log_path()
    if not log_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize_logs(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Gera métricas agregadas dos logs."""
    records = records if records is not None else read_logs()
    total = len(records)
    if total == 0:
        return {
            "total_consultas": 0,
            "fallback_rate": 0.0,
            "latencia_media_ms": 0.0,
            "similaridade_media": 0.0,
            "fontes_media": 0.0,
            "tokens_estimados_media": 0.0,
            "distribuicao_doc_type": {},
        }

    fallback_count = sum(1 for item in records if item.get("fallback") is True)
    latencies = [float(item.get("latency_ms", 0)) for item in records]
    scores = [float(item.get("avg_similarity", 0)) for item in records]
    source_counts = [int(item.get("source_count", 0)) for item in records]
    token_counts = [int(item.get("estimated_tokens", 0)) for item in records]

    doc_types: list[str] = []
    for item in records:
        doc_types.extend(item.get("retrieved_doc_types", []) or [])

    return {
        "total_consultas": total,
        "fallback_rate": round(fallback_count / total, 4),
        "latencia_media_ms": round(mean(latencies), 2),
        "similaridade_media": round(mean(scores), 4),
        "fontes_media": round(mean(source_counts), 2),
        "tokens_estimados_media": round(mean(token_counts), 2),
        "distribuicao_doc_type": dict(Counter(doc_types)),
    }


def main() -> None:
    """Exibe resumo de monitoramento via linha de comando."""
    parser = argparse.ArgumentParser(description="Resumo dos logs RAG.")
    parser.parse_args()

    summary = summarize_logs()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
