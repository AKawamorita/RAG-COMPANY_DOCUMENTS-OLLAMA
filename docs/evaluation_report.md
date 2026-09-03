# Relatório de Avaliação

## Objetivo da avaliação

Avaliar se o RAG consegue recuperar documentos relevantes e gerar respostas coerentes com base nas evidências encontradas.

## Métricas implementadas

### Precision@k por tipo de documento

Mede a proporção de documentos recuperados cujo tipo corresponde ao tipo esperado da pergunta.

Exemplo:

```text
Pergunta: Qual é o fornecedor da invoice INV-1001?
Tipo esperado: invoice
k: 4
Documentos recuperados: invoice, invoice, purchase_order, shipping_order
Precision@4 = 2 / 4 = 0,50
```

### Hit@k por tipo de documento

Indica se pelo menos um documento do tipo esperado apareceu entre os `k` documentos recuperados.

### Termos esperados na resposta

Verifica se a resposta contém termos importantes esperados, como número do documento, fornecedor ou valor.

### Fallback rate

Mede a quantidade de perguntas que receberam resposta de fallback:

```text
Não encontrado no documento com evidência suficiente.
```

## Como executar

```bash
python -m src.evaluator
```

O arquivo de saída será salvo em:

```text
data/processed/evaluation_results.csv
```

## Como evoluir a avaliação

Sugestões:

1. criar um conjunto maior de perguntas de controle;
2. adicionar ground truth com documento esperado;
3. medir recall@k por documento específico;
4. usar avaliação LLM-as-a-judge com critérios controlados;
5. comparar RAG contra LLM puro;
6. registrar erros por categoria de documento.

## Exemplo de interpretação para portfólio

> A avaliação inicial demonstrou que o sistema consegue recuperar documentos relevantes por tipo e responder perguntas simples com rastreabilidade. Perguntas sem evidência suficiente são direcionadas para fallback, reduzindo risco de alucinação.
