# Module 6-7: Retrieval-Augmented Generation

## Why RAG
Language models do not know the Sprints curriculum. Retrieval-augmented generation loads curriculum documents, splits them into chunks, embeds each chunk and stores the vectors in a vector database. At question time the system retrieves the most similar chunks and gives them to the model as context.

## Ingestion pipeline
Step one is loading documents from the curriculum folder. Step two is chunking. The Sprints standard is 500 tokens per chunk with a 50 token overlap, so that sentences cut at a boundary still appear whole in the next chunk. Step three is embedding with an embedding model. Step four is storing vectors and metadata in ChromaDB with a persistent directory.

## Retrieval
Use similarity search with a small k, typically 3 to 4. Store the source file name as metadata so answers can cite the module. Retrieval must complete within 1000 milliseconds so the orchestrator does not time out. Cache the vector store between requests and avoid rebuilding it on every query.

## Agentic RAG and self-reflection
Plain RAG retrieves once. Agentic RAG adds a reflection step: after retrieval the agent judges whether the context is sufficient to answer. If not, it reformulates the query, for example by expanding abbreviations or naming the module, and searches again. Limit the number of reformulations to avoid loops, normally two.

## Evaluating retrieval
Measure latency in milliseconds and check that the expected source document appears in the top results for a set of test questions.
