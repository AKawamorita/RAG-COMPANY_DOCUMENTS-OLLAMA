"""Avaliação básica do RAG.

Este script usa perguntas de controle para medir:

- se o tipo de documento esperado apareceu entre os documentos recuperados;
- precision@k aproximado por tipo de documento;
- se a resposta contém termos esperados;
- se o sistema entrou em fallback.

Exemplo:
    python -m src.evaluator
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import BASE_DIR, get_settings
from src.rag_chain import CorporateRAG


DEFAULT_QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.csv"
DEFAULT_OUTPUT_FILE = BASE_DIR / "data" / "processed" / "evaluation_results.csv"


def _split_expected_terms(value: Any) -> list[str]:
    """Converte termos esperados separados por pipe em lista."""
    if pd.isna(value):
        return []
    return [term.strip().lower() for term in str(value).split("|") if term.strip()]


def contains_expected_terms(answer: str, expected_terms: list[str]) -> bool:
    """Verifica se todos os termos esperados aparecem na resposta."""
    answer_lower = answer.lower()
    return all(term in answer_lower for term in expected_terms)


def evaluate_case(rag: CorporateRAG, row: pd.Series, k: int) -> dict[str, Any]:
    """Avalia uma pergunta de controle."""
    question = str(row["question"])
    expected_doc_type = str(row.get("expected_doc_type", "")).strip()
    expected_terms = _split_expected_terms(row.get("expected_terms"))

    result = rag.ask(question, doc_type=None, k=k)
    docs = result.get("retrieved_docs", [])

    retrieved_types = [doc.metadata.get("doc_type", "unknown") for doc in docs]
    relevant_count = sum(1 for doc_type in retrieved_types if doc_type == expected_doc_type)
    precision_at_k = relevant_count / max(1, len(docs))
    hit_at_k = relevant_count > 0

    answer = result.get("answer", "")
    terms_ok = contains_expected_terms(answer, expected_terms) if expected_terms else None

    return {
        "question": question,
        "expected_doc_type": expected_doc_type,
        "retrieved_doc_types": "|".join(retrieved_types),
        "precision_at_k_doc_type": round(precision_at_k, 4),
        "hit_at_k_doc_type": hit_at_k,
        "expected_terms": "|".join(expected_terms),
        "answer_contains_expected_terms": terms_ok,
        "avg_similarity": round(float(result.get("avg_similarity", 0.0)), 4),
        "fallback": result.get("fallback", False),
        "latency_ms": result.get("latency_ms", 0.0),
        "answer": answer,
    }


def run_evaluation(questions_file: Path, output_file: Path, k: int) -> pd.DataFrame:
    """Executa avaliação para todas as perguntas."""
    df_questions = pd.read_csv(questions_file)
    rag = CorporateRAG()

    results = [evaluate_case(rag, row, k=k) for _, row in df_questions.iterrows()]
    df_results = pd.DataFrame(results)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False, encoding="utf-8")
    return df_results


def main() -> None:
    """Executa avaliação via linha de comando."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Avaliação básica do RAG.")
    parser.add_argument("--questions-file", type=str, default=str(DEFAULT_QUESTIONS_FILE))
    parser.add_argument("--output-file", type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--k", type=int, default=settings.retrieval_k)
    args = parser.parse_args()

    df_results = run_evaluation(
        questions_file=Path(args.questions_file),
        output_file=Path(args.output_file),
        k=args.k,
    )

    print("=== Avaliação concluída ===")
    print(df_results[["question", "precision_at_k_doc_type", "hit_at_k_doc_type", "fallback"]])
    print(f"\nArquivo salvo em: {args.output_file}")


if __name__ == "__main__":
    main()
