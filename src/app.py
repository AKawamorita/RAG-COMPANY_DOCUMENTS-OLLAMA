"""Interface Streamlit para consulta RAG.

Execução local:
    streamlit run src/app.py

No Docker Compose, configure:
    RAG_API_URL=http://api:8000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.monitor import summarize_logs
from src.rag_chain import CorporateRAG
from src.retriever import get_vector_store_status


st.set_page_config(page_title="Project RAG", page_icon="📄", layout="wide")

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT_SECONDS = int(os.getenv("RAG_API_TIMEOUT_SECONDS", "120"))


@st.cache_resource
def load_rag() -> CorporateRAG:
    """Carrega o RAG uma única vez para o modo de execução direta."""
    return CorporateRAG()


def call_api(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = API_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Executa uma chamada para a FastAPI local e devolve o JSON."""
    body = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        url=f"{API_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API retornou HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Não foi possível acessar a API em {API_URL}: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("A API retornou uma resposta que não é JSON válido") from exc


def load_monitor_summary(use_api: bool) -> dict[str, Any]:
    """Obtém as métricas pelo endpoint ou diretamente do monitor local."""
    if use_api:
        return call_api("GET", "/api/v1/monitor/summary", timeout=10)
    return summarize_logs()


def load_vector_store_status(use_api: bool) -> dict[str, Any]:
    """Obtém o status do banco vetorial pela API ou pelo código local."""
    if use_api:
        return call_api("GET", "/api/v1/vector-store/status", timeout=10)
    return get_vector_store_status()


def render_vector_store_status(status: dict[str, Any]) -> None:
    """Exibe provider, contagem real e classificação visual da base vetorial."""
    backend_icon = status.get("backend_icon", "🗄️")
    backend_label = status.get("backend_label", "Desconhecido")
    record_count = status.get("record_count")
    count_label = (
        f"{int(record_count):,}".replace(",", ".")
        if record_count is not None
        else "—"
    )

    status_icon = status.get("status_icon", "⚪")
    status_label = status.get("status_label", "Status desconhecido")

    st.markdown(
        f"{backend_icon} **Banco vetorial:** {backend_label}  \n"
        f"📦 **Chunks indexados:** {count_label}  \n"
        f"{status_icon} **Status:** {status_label}"
    )

    if status.get("store_name"):
        st.caption(f"Índice/collection: {status['store_name']}")

    if status.get("error"):
        with st.expander("Detalhes da conexão"):
            st.code(status["error"])


def render_sources(result: dict[str, Any]) -> None:
    """Renderiza documentos recebidos tanto da API quanto do modo direto."""
    documents = result.get("retrieved_docs", [])

    if not documents:
        st.info("Nenhuma fonte foi retornada.")
        return

    for idx, doc in enumerate(documents, start=1):
        if isinstance(doc, dict):
            score = float(doc.get("score", 0) or 0)
            metadata = doc.get("metadata", {})
            content = str(doc.get("content", ""))
        else:
            score = float(getattr(doc, "score", 0) or 0)
            metadata = getattr(doc, "metadata", {})
            content = str(getattr(doc, "content", ""))

        st.markdown(f"**Fonte {idx}** — score `{score:.3f}`")
        st.json(metadata)
        st.write(content[:1000])


def main() -> None:
    """Renderiza a aplicação."""
    st.title("📄 Projeto RAG — Documentos Corporativos")
    st.caption(
        "LangChain + LangGraph + Redis Vector Search + ((Ollama/Gemma) / Groq)"
    )

    with st.sidebar:
        st.header("Configurações")
        execution_mode = st.radio(
            "Modo de execução",
            options=["API local", "Direto no Streamlit"],
            index=0,
            help=(
                "API local usa a FastAPI do Docker. O modo direto preserva "
                "o comportamento anterior para desenvolvimento."
            ),
        )
        use_api = execution_mode == "API local"

        if use_api:
            st.caption(f"API: {API_URL}")
            if st.button("Verificar API"):
                try:
                    health = call_api("GET", "/health", timeout=5)
                    ready = call_api("GET", "/ready", timeout=30)
                    st.success("API disponível")
                    st.json({"health": health, "ready": ready})
                except RuntimeError as exc:
                    st.error(str(exc))

        try:
            vector_status = load_vector_store_status(use_api)
        except Exception as exc:
            vector_status = {
                "backend_label": "Desconhecido",
                "record_count": None,
                "status": "unavailable",
                "status_icon": "⚪",
                "status_label": "Não foi possível consultar o banco vetorial",
                "error": str(exc),
            }

        render_vector_store_status(vector_status)

        doc_type = st.selectbox(
            "Filtrar por tipo de documento",
            options=[
                "",
                "invoice",
                "purchase_order",
                "shipping_order",
                "inventory_report",
                "unknown",
            ],
            index=0,
        )
        k = st.slider(
            "Quantidade de documentos recuperados",
            min_value=1,
            max_value=10,
            value=4,
        )

        st.divider()
        st.subheader("Monitoramento")

        if st.button("Atualizar métricas"):
            try:
                st.session_state["monitor_summary"] = load_monitor_summary(use_api)
                st.session_state.pop("monitor_error", None)
            except RuntimeError as exc:
                st.session_state["monitor_error"] = str(exc)

        if "monitor_summary" not in st.session_state and not use_api:
            st.session_state["monitor_summary"] = summarize_logs()

        if st.session_state.get("monitor_error"):
            st.warning(st.session_state["monitor_error"])

        summary = st.session_state.get("monitor_summary")
        if summary is not None:
            st.json(summary)
        elif use_api:
            st.caption("Clique em Atualizar métricas para consultar a API.")

    question = st.text_input(
        "Digite sua pergunta",
        placeholder=(
            "Ex.: Faça um resumo de todas as invoices que têm como produto "
            "Guarana Fantastica"
        ),
    )

    if st.button("Consultar", type="primary") and question:
        with st.spinner("Consultando documentos..."):
            try:
                if use_api:
                    result = call_api(
                        "POST",
                        "/api/v1/query",
                        payload={
                            "question": question,
                            "doc_type": doc_type or None,
                            "k": k,
                        },
                    )
                else:
                    rag = load_rag()
                    result = rag.ask(question, doc_type=doc_type or None, k=k)
            except Exception as exc:
                st.error(f"Falha ao executar a consulta: {exc}")
                return

        st.subheader("Resposta")
        st.write(result.get("answer", ""))

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Similaridade média",
            f"{float(result.get('avg_similarity', 0) or 0):.3f}",
        )
        col2.metric(
            "Latência",
            f"{float(result.get('latency_ms', 0) or 0):.0f} ms",
        )
        col3.metric("Fallback", str(result.get("fallback", False)))

        with st.expander("Fontes recuperadas"):
            render_sources(result)


if __name__ == "__main__":
    main()
