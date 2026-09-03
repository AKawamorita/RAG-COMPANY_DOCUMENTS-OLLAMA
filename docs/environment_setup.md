# Instalação do ambiente virtual

## 1. Criar ambiente virtual

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Atualizar pip

```bash
python -m pip install --upgrade pip
```

## 3. Instalar dependências

```bash
pip install -r requirements.txt
```
Obs. O arquivo foi testado e está OK. Caso precise existe um arquivo .zip este arquivo foi gerado pelo pip freeze.

## 4. Configurar variáveis de ambiente

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

No Windows PowerShell:

```powershell
copy .env.example .env
```

Edite o `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=[sua_chave_groq_aqui]
GROQ_MODEL=llama-3.1-8b-instant

EMBEDDING_PROVIDER=huggingface
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHROMA_COLLECTION_NAME=company_documents_rag
CHROMA_PERSIST_DIR=data/chroma
RAW_DATA_DIR=data/raw
LOGS_DIR=logs

CHUNK_SIZE=900
CHUNK_OVERLAP=120
RETRIEVAL_K=8
MIN_RELEVANCE_SCORE=0.25
```

O valor de `[sua_chave_groq_aqui]` deve ser substituida pela API-Key gerada no Groq. https://console.groq.com/home

```bash
ollama list
```

## 7. Rodar interface

```bash
streamlit run src/app.py
```

## 8. Rodar documentação

> A documentação também está disponivel em meu portifólio na seção Portfólio Técnico: https://akawamorita.github.io/ 

```bash
mkdocs serve
```

## 9. Incluindo arquivos de documentos

> Link do DataSet: https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset 

