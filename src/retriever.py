"""
retriever.py

Camada de recuperação de documentos corporativos.

Este módulo isola o acesso ao banco vetorial. O backend pode ser ChromaDB ou
Redis Vector Search, selecionado por VECTOR_STORE_PROVIDER no arquivo .env.

Estratégia usada:
1. Busca exata por identificadores, como invoice id, order id, purchase order.
2. Busca vetorial por similaridade semântica.
3. Junção dos resultados, removendo duplicidades.
4. Reranking simples, priorizando matches exatos.

Isso melhora muito consultas administrativas com números e códigos, porque
busca vetorial pura nem sempre encontra bem identificadores numéricos.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.config import get_settings
from src.embeddings_factory import get_embeddings


@dataclass
class RetrievedDocument:
    """Representa um documento recuperado com conteúdo, score e metadados."""

    content: str
    metadata: dict[str, Any]
    score: float

    @property
    def source_label(self) -> str:
        """Gera um rótulo amigável da fonte para exibição no prompt e na resposta."""

        source = (
            self.metadata.get("source")
            or self.metadata.get("source_file")
            or "fonte_desconhecida"
        )

        doc_type = self.metadata.get("doc_type", "unknown")
        row_id = self.metadata.get("row_id")
        chunk_id = self.metadata.get("chunk_id")
        retrieval_type = self.metadata.get("retrieval_type")

        parts = [
            f"fonte={source}",
            f"tipo={doc_type}",
        ]

        if row_id is not None:
            parts.append(f"linha={row_id}")

        if chunk_id is not None:
            parts.append(f"chunk={chunk_id}")

        if retrieval_type:
            parts.append(f"busca={retrieval_type}")

        return " | ".join(parts)


def _classify_record_count(record_count: int, attention_limit: int) -> dict[str, str]:
    """Classifica visualmente a quantidade de chunks do banco vetorial."""

    if record_count == 0:
        return {
            "status": "empty",
            "status_label": "Aviso: base vetorial vazia",
            "status_icon": "🔴",
        }

    if record_count <= attention_limit:
        return {
            "status": "attention",
            "status_label": "Atenção: poucos chunks indexados",
            "status_icon": "🟡",
        }

    return {
        "status": "ok",
        "status_label": "OK",
        "status_icon": "🟢",
    }


def get_vector_store_status() -> dict[str, Any]:
    """Obtém provider, disponibilidade e quantidade real de chunks indexados.

    A consulta é feita diretamente no índice Redis ou na collection Chroma,
    sem carregar embeddings nem inicializar o fluxo completo do RAG.
    """

    settings = get_settings()
    provider = settings.vector_store_provider
    provider_labels = {
        "redis": "Redis",
        "chroma": "ChromaDB",
    }
    provider_icons = {
        "redis": "⚡",
        "chroma": "🟣",
    }

    base_status: dict[str, Any] = {
        "backend": provider,
        "backend_label": provider_labels.get(provider, provider.title()),
        "backend_icon": provider_icons.get(provider, "🗄️"),
        "record_type": "chunks",
        "attention_limit": settings.vector_store_attention_limit,
    }

    try:
        if provider == "redis":
            from redis import Redis

            client = Redis.from_url(settings.redis_url)
            try:
                client.ping()
                index_info = client.ft(settings.redis_index_name).info()
                raw_count = index_info.get(
                    "num_docs",
                    index_info.get(b"num_docs", 0),
                )
                if isinstance(raw_count, bytes):
                    raw_count = raw_count.decode("utf-8")
                record_count = int(float(raw_count))
            finally:
                client.close()

            store_name = settings.redis_index_name

        elif provider == "chroma":
            import chromadb

            client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir)
            )
            collection = client.get_collection(
                name=settings.chroma_collection_name
            )
            record_count = int(collection.count())
            store_name = settings.chroma_collection_name

        else:
            raise ValueError(
                "VECTOR_STORE_PROVIDER inválido: "
                f"{provider}. Use 'chroma' ou 'redis'."
            )

        return {
            **base_status,
            "available": True,
            "store_name": store_name,
            "record_count": record_count,
            **_classify_record_count(
                record_count,
                settings.vector_store_attention_limit,
            ),
        }

    except Exception as exc:
        return {
            **base_status,
            "available": False,
            "store_name": (
                settings.redis_index_name
                if provider == "redis"
                else settings.chroma_collection_name
            ),
            "record_count": None,
            "status": "unavailable",
            "status_label": "Banco vetorial indisponível",
            "status_icon": "⚪",
            "error": f"{type(exc).__name__}: {exc}",
        }


def extract_identifiers(text: str) -> list[str]:
    """
    Extrai possíveis identificadores da pergunta.

    Exemplos capturados:
    - 10707
    - INV-1001
    - PO-2024-001
    - ORDER-10707
    - INVOICE 10707

    Essa função ajuda porque embeddings nem sempre lidam bem com números,
    códigos e IDs específicos.
    """

    if not text:
        return []

    text_upper = text.upper()

    patterns = [
        # Números com 4 ou mais dígitos.
        r"\b\d{4,}\b",

        # Códigos como INV-1001, PO-2024, ORDER_10707.
        r"\b[A-Z]{2,15}[-_ ]?\d{2,}\b",

        # Casos como invoice order id 10707, order id 10707 etc.
        r"\b(?:INVOICE|ORDER|ID|PO|PURCHASE|NUMBER|NUMERO|NÚMERO)\s+[-_ ]?\d{2,}\b",
    ]

    identifiers: list[str] = []

    for pattern in patterns:
        identifiers.extend(re.findall(pattern, text_upper))

    cleaned: list[str] = []

    for identifier in identifiers:
        identifier = identifier.strip()

        # Quando vier "ORDER ID 10707", também guardamos só "10707".
        numbers = re.findall(r"\d{4,}", identifier)
        cleaned.append(identifier)

        for number in numbers:
            cleaned.append(number)

    # Remove duplicados preservando ordem.
    return list(dict.fromkeys(cleaned))


class CorporateRetriever:
    """
    Retriever para documentos corporativos.

    Ele usa ChromaDB ou Redis como banco vetorial e combina:
    - busca exata por identificadores;
    - busca semântica por embeddings.
    """

    def __init__(self) -> None:
        """Inicializa configurações, embeddings e o banco vetorial selecionado."""

        self.settings = get_settings()
        self.provider = self.settings.vector_store_provider
        self.vector_engine = ""

        embeddings = get_embeddings()

        if self.provider == "chroma":
            from langchain_chroma import Chroma
            self.vector_engine = self.provider

            self.vector_store = Chroma(
                collection_name=self.settings.chroma_collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.settings.chroma_persist_dir),
            )
            return

        if self.provider == "redis":
            from langchain_redis import RedisVectorStore
            self.vector_engine = self.provider

            self.vector_store = RedisVectorStore.from_existing_index(
                index_name=self.settings.redis_index_name,
                embedding=embeddings,
                redis_url=self.settings.redis_url,
            )
            return

        raise ValueError(
            "VECTOR_STORE_PROVIDER inválido: "
            f"{self.provider}. Use 'chroma' ou 'redis'."
        )

    def _redis_has_indexed_field(self, field_name: str) -> bool:
        """Informa se um campo faz parte do índice Redis atual."""

        if self.provider != "redis":
            return False

        try:
            return any(
                field.name == field_name
                for field in self.vector_store.index.schema.fields.values()
            )
        except (AttributeError, TypeError):
            return False

    def _exact_chroma_search(
        self,
        identifiers: list[str],
        limit: int,
        doc_type: Optional[str],
    ) -> list[RetrievedDocument]:
        """Executa a busca literal usando a API nativa do ChromaDB."""

        exact_results: list[RetrievedDocument] = []

        for identifier in identifiers:
            search_variants = list(
                dict.fromkeys(
                    [identifier, identifier.upper(), identifier.lower()]
                )
            )

            for search_value in search_variants:
                result = self.vector_store._collection.get(
                    where_document={"$contains": search_value},
                    limit=limit,
                    include=["documents", "metadatas"],
                )

                documents = result.get("documents", [])
                metadatas = result.get("metadatas", [])
                ids = result.get("ids", [])

                for idx, document in enumerate(documents):
                    metadata = metadatas[idx] if idx < len(metadatas) else {}
                    chroma_id = ids[idx] if idx < len(ids) else None

                    if doc_type and metadata.get("doc_type") != doc_type:
                        continue

                    exact_results.append(
                        RetrievedDocument(
                            content=document,
                            metadata={
                                **dict(metadata),
                                "retrieval_type": "exact_match",
                                "matched_identifier": identifier,
                                "chroma_id": chroma_id,
                            },
                            score=1.0,
                        )
                    )

        return exact_results

    def _exact_redis_search(
        self,
        identifiers: list[str],
        limit: int,
        doc_type: Optional[str],
    ) -> list[RetrievedDocument]:
        """Executa busca textual exata no índice do Redis Vector Search."""

        from redisvl.query import FilterQuery
        from redisvl.query.filter import Tag, Text

        content_field = self.vector_store.config.content_field
        indexed_fields = {
            field.name
            for field in self.vector_store.index.schema.fields.values()
        }
        return_fields = [content_field]

        if "_metadata_json" in indexed_fields:
            return_fields.append("_metadata_json")

        return_fields.extend(
            field_name
            for field_name in indexed_fields
            if field_name
            not in {
                content_field,
                self.vector_store.config.embedding_field,
                "_index_name",
                "_metadata_json",
            }
        )

        exact_results: list[RetrievedDocument] = []

        for identifier in identifiers:
            filter_expression = Text(content_field) == identifier

            if doc_type and self._redis_has_indexed_field("doc_type"):
                filter_expression = (
                    filter_expression & (Tag("doc_type") == doc_type)
                )

            query = FilterQuery(
                filter_expression=filter_expression,
                return_fields=return_fields,
                num_results=limit,
            )

            results = self.vector_store.index.query(query)

            for result in results:
                metadata: dict[str, Any] = {}
                metadata_json = result.get("_metadata_json")

                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                if not metadata:
                    metadata = {
                        key: value
                        for key, value in result.items()
                        if key
                        not in {
                            "id",
                            content_field,
                            self.vector_store.config.embedding_field,
                            "_index_name",
                            "_metadata_json",
                        }
                        and not key.startswith("_")
                    }

                if doc_type and metadata.get("doc_type") != doc_type:
                    continue

                exact_results.append(
                    RetrievedDocument(
                        content=str(result.get(content_field, "")),
                        metadata={
                            **metadata,
                            "retrieval_type": "exact_match",
                            "matched_identifier": identifier,
                            "redis_id": result.get("id"),
                        },
                        score=1.0,
                    )
                )

        return exact_results

    def _exact_document_search(
        self,
        query: str,
        limit: int = 5,
        doc_type: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """
        Faz busca exata no conteúdo armazenado no backend selecionado.

        Essa busca complementa a busca vetorial, principalmente para:
        - invoice id;
        - order id;
        - purchase order;
        - códigos de documento;
        - números de pedido;
        - identificadores administrativos.

        Exemplo:
            query = "me informe sobre invoice order id 10707"

        Nesse caso, a função procura literalmente por "10707" nos documentos.
        """

        identifiers = extract_identifiers(query)

        if not identifiers:
            return []

        try:
            if self.provider == "chroma":
                return self._exact_chroma_search(
                    identifiers=identifiers,
                    limit=limit,
                    doc_type=doc_type,
                )

            return self._exact_redis_search(
                identifiers=identifiers,
                limit=limit,
                doc_type=doc_type,
            )
        except Exception as exc:
            print(f"Aviso: falha na busca exata ({self.provider}): {exc}")
            return []

    def _vector_filter(self, doc_type: Optional[str]) -> Any:
        """Converte o filtro de tipo para a sintaxe de cada backend."""

        if not doc_type:
            return None

        if self.provider == "chroma":
            return {"doc_type": doc_type}

        if self._redis_has_indexed_field("doc_type"):
            from redisvl.query.filter import Tag

            return Tag("doc_type") == doc_type

        return None

    def _vector_search(
        self,
        query: str,
        k: int,
        doc_type: Optional[str],
    ) -> list[RetrievedDocument]:
        """Executa a busca semântica e normaliza scores entre os backends."""

        backend_filter = self._vector_filter(doc_type)
        search_k = k

        # Se um índice Redis antigo não tiver doc_type indexado, buscamos mais
        # candidatos e aplicamos o filtro em Python.
        if self.provider == "redis" and doc_type and backend_filter is None:
            search_k = max(k * 5, 20)

        if self.provider == "chroma":
            docs_and_scores = (
                self.vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=search_k,
                    filter=backend_filter,
                )
            )
        else:
            docs_and_distances = self.vector_store.similarity_search_with_score(
                query=query,
                k=search_k,
                filter=backend_filter,
            )
            docs_and_scores = [
                (doc, max(-1.0, min(1.0, 1.0 - float(distance))))
                for doc, distance in docs_and_distances
            ]

        vector_docs: list[RetrievedDocument] = []

        for doc, score in docs_and_scores:
            if doc_type and doc.metadata.get("doc_type") != doc_type:
                continue

            vector_docs.append(
                RetrievedDocument(
                    content=doc.page_content,
                    metadata={
                        **dict(doc.metadata),
                        "retrieval_type": "vector_similarity",
                        "vector_engine": self.vector_engine,
                    },
                    score=float(score),
                )
            )

        return vector_docs[:k]

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        doc_type: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """
        Recupera documentos usando busca híbrida simples.

        Estratégia:
        1. Tenta encontrar identificadores exatos no backend selecionado.
        2. Executa busca vetorial por similaridade.
        3. Junta os resultados.
        4. Remove duplicidades.
        5. Ordena priorizando maior score.

        O parâmetro doc_type permite filtrar por tipo de documento, por exemplo:
        - invoice
        - purchase_order
        - shipping_order
        - inventory_report
        """

        k = k or self.settings.retrieval_k
        exact_docs = self._exact_document_search(
            query=query,
            limit=k,
            doc_type=doc_type,
        )
        vector_docs = self._vector_search(
            query=query,
            k=k,
            doc_type=doc_type,
        )

        merged: list[RetrievedDocument] = []
        seen_keys: set[str] = set()

        for doc in exact_docs + vector_docs:
            key = (
                f"{doc.metadata.get('source')}|"
                f"{doc.metadata.get('source_file')}|"
                f"{doc.metadata.get('row_id')}|"
                f"{doc.metadata.get('chunk_id')}|"
                f"{doc.content[:120]}"
            )

            if key not in seen_keys:
                merged.append(doc)
                seen_keys.add(key)

        merged.sort(key=lambda item: item.score, reverse=True)

        return merged[:k]


def format_context(
    docs: list[RetrievedDocument],
    max_chars_per_doc: int = 1800,
) -> str:
    """
    Formata os documentos recuperados para entrar no prompt do LLM.

    O objetivo é deixar claro para o modelo:
    - qual documento foi recuperado;
    - qual fonte foi usada;
    - qual foi o score;
    - qual foi o conteúdo disponível.
    """

    blocks: list[str] = []

    for idx, doc in enumerate(docs, start=1):
        content = doc.content[:max_chars_per_doc]

        blocks.append(
            f"[DOCUMENTO {idx}]\n"
            f"Fonte: {doc.source_label}\n"
            f"Score de similaridade: {doc.score:.3f}\n"
            f"Conteúdo:\n{content}"
        )

    return "\n\n".join(blocks)


def average_score(docs: list[RetrievedDocument]) -> float:
    """Calcula o score médio dos documentos recuperados."""

    if not docs:
        return 0.0

    return sum(doc.score for doc in docs) / len(docs)
