"""Ingestão de documentos corporativos no Redis Vector Search.

O script aceita dois formatos principais:

1. CSV com texto já extraído dos documentos.
2. PDF, usando PyPDFLoader.

Exemplo de uso:
    python -m src.ingest --reset

O pipeline executa:
    leitura dos arquivos -> limpeza simples -> criação de chunks -> embeddings -> Redis.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from redis import Redis
from redis.exceptions import ResponseError

from src.config import get_settings
from src.embeddings_factory import get_embeddings


# Nomes comuns encontrados em datasets de OCR ou documentos corporativos.
TEXT_COLUMN_CANDIDATES = [
    "text",
    "extracted_text",
    "ocr_text",
    "content",
    "document_text",
    "texto",
    "raw_text",
]

DOC_TYPE_COLUMN_CANDIDATES = [
    "label",
    "category",
    "doc_type",
    "document_type",
    "type",
    "class",
    "tipo",
]

SOURCE_COLUMN_CANDIDATES = [
    "filename",
    "file_name",
    "document",
    "source",
    "image",
    "pdf",
    "arquivo",
]


def get_redis_setting(settings: Any, name: str, default: str | int) -> str | int:
    """Obtém uma configuração do ambiente, do Settings ou usa o padrão informado."""
    env_name = name.upper()
    env_value = os.getenv(env_name)
    if env_value is not None:
        return int(env_value) if isinstance(default, int) else env_value

    return getattr(settings, name, default)


def redact_redis_url(redis_url: str) -> str:
    """Oculta a senha da URL antes de exibi-la no console."""
    return re.sub(r"(://[^:/@\s]+:)[^@]+@", r"\1***@", redis_url)


def clean_text(text: str) -> str:
    """Remove ruídos básicos do texto antes de gerar chunks."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_existing_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    """Localiza a primeira coluna existente em uma lista de candidatos."""
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def infer_doc_type_from_path(path: Path) -> str:
    """Tenta inferir o tipo de documento pelo nome da pasta ou do arquivo."""
    known_types = ["invoice", "purchase_order", "shipping_order", "inventory_report"]
    path_text = str(path).lower()
    for doc_type in known_types:
        if doc_type in path_text:
            return doc_type
    return path.parent.name if path.parent.name not in {"raw", "data"} else "unknown"


def load_csv_documents(csv_path: Path) -> list[Document]:
    """Carrega documentos de um CSV.

    O carregador tenta identificar automaticamente a coluna de texto e algumas
    colunas de metadados, como tipo do documento e nome do arquivo.
    """
    df = pd.read_csv(csv_path)

    text_col = first_existing_column(df.columns, TEXT_COLUMN_CANDIDATES)
    if not text_col:
        raise ValueError(
            f"Não encontrei coluna de texto em {csv_path.name}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    doc_type_col = first_existing_column(df.columns, DOC_TYPE_COLUMN_CANDIDATES)
    source_col = first_existing_column(df.columns, SOURCE_COLUMN_CANDIDATES)

    documents: list[Document] = []

    for idx, row in df.iterrows():
        text = clean_text(str(row.get(text_col, "")))
        if len(text) < 20:
            continue

        doc_type = str(row.get(doc_type_col, "unknown")) if doc_type_col else "unknown"
        source_name = str(row.get(source_col, csv_path.name)) if source_col else csv_path.name

        metadata = {
            "source": source_name,
            "source_file": csv_path.name,
            "row_id": int(idx),
            "doc_type": doc_type,
            "word_count": len(text.split()),
        }

        # Preserva alguns metadados adicionais úteis, sem exagerar no volume.
        for col in df.columns:
            if col == text_col:
                continue
            value = row.get(col)
            if pd.isna(value):
                continue
            value_str = str(value)
            if len(value_str) <= 120:
                metadata[col] = value_str

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    """Carrega um PDF e inclui metadados básicos."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    for page in pages:
        page.page_content = clean_text(page.page_content)
        page.metadata.update(
            {
                "source": pdf_path.name,
                "source_file": pdf_path.name,
                "doc_type": infer_doc_type_from_path(pdf_path),
                "word_count": len(page.page_content.split()),
            }
        )

    return [page for page in pages if len(page.page_content) >= 20]


def load_documents(data_dir: Path) -> list[Document]:
    """Carrega todos os CSVs e PDFs encontrados na pasta de dados."""
    documents: list[Document] = []

    csv_files = sorted(data_dir.rglob("*.csv"))
    pdf_files = sorted(data_dir.rglob("*.pdf"))

    for csv_file in csv_files:
        try:
            docs = load_csv_documents(csv_file)
            documents.extend(docs)
            print(f"[OK] CSV carregado: {csv_file} | documentos: {len(docs)}")
        except Exception as exc:  # noqa: BLE001 - log simples para protótipo
            print(f"[ERRO] Falha ao carregar CSV {csv_file}: {exc}")

    for pdf_file in pdf_files:
        try:
            docs = load_pdf_documents(pdf_file)
            documents.extend(docs)
            print(f"[OK] PDF carregado: {pdf_file} | páginas: {len(docs)}")
        except Exception as exc:  # noqa: BLE001 - log simples para protótipo
            print(f"[ERRO] Falha ao carregar PDF {pdf_file}: {exc}")

    return documents


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Divide documentos em chunks para melhorar a recuperação semântica."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )
    chunks = splitter.split_documents(documents)

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_chars"] = len(chunk.page_content)

    return chunks


def build_document_ids(chunks: list[Document]) -> list[str]:
    """Gera IDs determinísticos para evitar duplicatas em novas ingestões."""
    ids: list[str] = []

    for chunk in chunks:
        identity = "|".join(
            [
                str(chunk.metadata.get("source_file", "")),
                str(chunk.metadata.get("source", "")),
                str(chunk.metadata.get("row_id", "")),
                str(chunk.metadata.get("page", "")),
                str(chunk.metadata.get("chunk_id", "")),
                chunk.page_content,
            ]
        )
        ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())

    return ids


def drop_redis_index(redis_url: str, index_name: str) -> bool:
    """Remove somente o índice do RAG e os documentos associados a ele."""
    client = Redis.from_url(redis_url, decode_responses=True)

    try:
        client.ping()
        client.execute_command("FT.DROPINDEX", index_name, "DD")
        return True
    except ResponseError as exc:
        error_message = str(exc).lower()
        if "unknown index name" in error_message or "no such index" in error_message:
            return False
        raise
    finally:
        client.close()


def build_redis(chunks: list[Document], reset: bool = False) -> RedisVectorStore:
    """Cria ou atualiza o índice vetorial no Redis."""
    settings = get_settings()

    redis_url = str(
        get_redis_setting(settings, "redis_url", "redis://localhost:6379/0")
    )
    index_name = str(
        get_redis_setting(settings, "redis_index_name", "rag_documents")
    )
    key_prefix = str(
        get_redis_setting(settings, "redis_key_prefix", "rag:document")
    )
    batch_size = int(get_redis_setting(settings, "redis_batch_size", 100))

    if batch_size <= 0:
        raise ValueError("REDIS_BATCH_SIZE deve ser maior que zero.")

    if reset:
        removed = drop_redis_index(redis_url, index_name)
        status = "removido" if removed else "ainda não existia"
        print(f"[OK] Índice Redis '{index_name}' {status}.")

    embeddings = get_embeddings()

    config = RedisConfig(
        index_name=index_name,
        redis_url=redis_url,
        key_prefix=key_prefix,
        distance_metric="COSINE",
        metadata_schema=[
            {"name": "source", "type": "tag"},
            {"name": "source_file", "type": "tag"},
            {"name": "doc_type", "type": "tag"},
            {"name": "row_id", "type": "numeric"},
            {"name": "page", "type": "numeric"},
            {"name": "chunk_id", "type": "numeric"},
            {"name": "word_count", "type": "numeric"},
        ],
    )
    vector_store = RedisVectorStore(embeddings=embeddings, config=config)

    document_ids = build_document_ids(chunks)
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        vector_store.add_documents(
            documents=chunks[start:end],
            ids=document_ids[start:end],
        )
        print(f"[OK] Chunks enviados ao Redis: {end}/{len(chunks)}")

    return vector_store


def main() -> None:
    """Executa o pipeline de ingestão via linha de comando."""
    parser = argparse.ArgumentParser(
        description="Ingestão de documentos no Redis Vector Search."
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Pasta com CSVs/PDFs.")
    parser.add_argument("--reset", action="store_true", help="Recria a base vetorial do zero.")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path(args.data_dir) if args.data_dir else settings.raw_data_dir

    print("=== Ingestão RAG ===")
    print(f"Pasta de dados: {data_dir}")
    redis_url = str(
        get_redis_setting(settings, "redis_url", "redis://localhost:6379/0")
    )
    print(
        "Redis: "
        f"{redact_redis_url(redis_url)}"
    )
    print(
        "Índice Redis: "
        f"{get_redis_setting(settings, 'redis_index_name', 'rag_documents')}"
    )
    print(f"Embedding model: {settings.embedding_model}")

    documents = load_documents(data_dir)
    if not documents:
        raise RuntimeError(
            "Nenhum documento válido encontrado. Coloque CSVs/PDFs em data/raw/ "
            "ou confira os nomes das colunas de texto."
        )

    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    build_redis(chunks, reset=args.reset)

    print("=== Concluído ===")
    print(f"Documentos carregados: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)}")
    print(
        "Índice Redis: "
        f"{get_redis_setting(settings, 'redis_index_name', 'rag_documents')}"
    )


if __name__ == "__main__":
    main()
