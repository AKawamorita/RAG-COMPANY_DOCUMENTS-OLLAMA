# Otimização de Uso de Tokens em RAG

Esta seção documenta algumas estratégias consideradas para tornar o uso de tokens mais eficiente em uma arquitetura RAG. O objetivo não é implementar todas essas técnicas neste momento, mas demonstrar consciência técnica sobre custo, desempenho e qualidade do contexto enviado ao modelo de linguagem.

## 1. Prompt Caching

O **Prompt Caching** pode ser utilizado quando parte do prompt enviado ao modelo é repetitiva, como instruções fixas do sistema, regras de segurança, formato de resposta e contexto técnico comum.

Para favorecer essa técnica, a estrutura do prompt deve manter as partes mais estáticas no início, como:

- instruções do sistema;
- regras de geração de SQL;
- restrições de segurança;
- contexto técnico reaproveitável.

A pergunta do usuário deve ficar ao final do prompt, pois tende a variar a cada requisição.

Essa organização facilita o reaproveitamento de tokens cacheados pelo provedor de IA, reduzindo custo e latência quando suportado pela API utilizada.

## 2. Reranking

O **Reranking** pode ser usado após a busca inicial no banco vetorial. Em vez de enviar muitos chunks diretamente para o LLM, o sistema primeiro recupera uma quantidade maior de candidatos e depois reordena esses trechos por relevância.

Exemplo conceitual:

- recuperar inicialmente 15 ou 25 chunks candidatos;
- aplicar um modelo de reranking;
- enviar ao LLM apenas os 3 ou 5 trechos mais relevantes.

Essa abordagem reduz o volume de tokens enviados ao modelo principal e aumenta a chance de que o contexto usado na resposta seja realmente útil.

## 3. Semantic Caching

O **Semantic Caching** pode ser aplicado no nível da aplicação para evitar chamadas repetidas ao LLM quando perguntas semelhantes já foram respondidas anteriormente.

Diferente de um cache tradicional, que depende de textos idênticos, o cache semântico compara a similaridade entre perguntas.

Exemplo conceitual:

- uma pergunta do usuário é convertida em embedding;
- o sistema verifica se já existe uma pergunta semanticamente parecida no cache;
- se a similaridade for alta, a resposta anterior pode ser reutilizada;
- se não houver correspondência suficiente, o fluxo RAG normal é executado.

Essa estratégia pode ser implementada futuramente com soluções como Redis, ChromaDB ou bibliotecas específicas de cache semântico.

## 4. Hierarchical Chunking

O **Hierarchical Chunking** é uma estratégia de organização dos documentos em diferentes níveis de granularidade.

Em vez de quebrar os documentos apenas em pedaços fixos, o conteúdo pode ser organizado em níveis como:

- documento;
- seção;
- tabela;
- campo;
- regra de negócio;
- exemplo de consulta SQL.

Essa estrutura ajuda o RAG a recuperar primeiro o contexto mais específico e, quando necessário, complementar com uma visão mais ampla.

No caso de documentação SQL, essa abordagem é útil para separar:

- descrição da tabela;
- lista de colunas;
- relacionamentos;
- regras de negócio;
- exemplos de consultas.

Com isso, o sistema evita enviar páginas inteiras de documentação ao LLM e passa a enviar apenas os trechos mais relevantes para a pergunta.

## 5. Estratégia Recomendada para o Projeto

Para este projeto, a otimização de tokens pode ser considerada da seguinte forma:

1. Manter prompts estruturados, com instruções fixas no início e pergunta do usuário no final, favorecendo Prompt Caching.
2. Usar Reranking para reduzir a quantidade de chunks enviados ao LLM.
3. Considerar Semantic Caching caso o sistema passe a receber perguntas repetitivas ou muito semelhantes.
4. Organizar a documentação em Markdown com Hierarchical Chunking, separando tabelas, campos, regras e exemplos.
5. Priorizar poucos chunks de alta relevância em vez de muitos chunks genéricos.

