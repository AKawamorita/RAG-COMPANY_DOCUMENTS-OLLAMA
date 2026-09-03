# Monitoramento

## Objetivo

O monitoramento tem como objetivo demonstrar práticas básicas de RAGOps/MLOps em um projeto de portfólio.

## Logs registrados

Cada consulta é registrada em `logs/rag_queries.jsonl` com campos como:

- data/hora UTC;
- pergunta original;
- pergunta sanitizada;
- filtro de tipo de documento;
- similaridade média;
- quantidade de fontes recuperadas;
- tipos de documentos recuperados;
- indicação de fallback;
- latência;
- tokens estimados;
- alerta de sanitização.

## Métricas agregadas

O script `src/monitor.py` calcula:

- total de consultas;
- fallback rate;
- latência média;
- similaridade média;
- média de fontes recuperadas;
- média de tokens estimados;
- distribuição dos tipos de documento recuperados.

## Como executar

```bash
python -m src.monitor
```

## Por que isso é importante?

Em aplicações de RAG, não basta apenas gerar respostas. É necessário acompanhar:

- quando o sistema não encontra evidência;
- quando a similaridade está baixa;
- se a latência está aceitável;
- se algumas categorias são mais difíceis de recuperar;
- se há perguntas fora do escopo;
- se o sistema está respondendo sem fonte.

## Evoluções possíveis

- armazenar logs em banco relacional;
- criar dashboard com Streamlit;
- adicionar métricas por usuário ou por documento;
- monitorar drift de perguntas;
- integrar LangSmith ou OpenTelemetry;
- criar alertas para queda de similaridade média.
