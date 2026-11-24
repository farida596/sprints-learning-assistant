# Sprints Learning Assistant

An AI-powered learning and career assistant for Sprints learners. The application brings curriculum Q&A, learner progress lookup, Python assignment assessment, and career-placement guidance into one conversational interface.

## Features

- **Curriculum Q&A:** retrieves relevant passages from the included learning materials and uses them to ground responses.
- **Learner progress:** looks up anonymized sample learner records stored in SQLite.
- **Python assessment:** runs submitted Python code in a restricted subprocess and returns execution-grounded feedback.
- **Career guidance:** checks readiness information and preferred employers against the sample placement policy.
- **Human approval:** pauses recommendations involving tier-one employers until a career expert approves or rejects them.
- **Conversation memory:** keeps conversation state isolated by learner thread while the process is running.
- **Offline fallback:** uses deterministic routing, lexical retrieval, and rule-based assessment when no OpenAI API key is configured.

## Architecture

The LangGraph orchestrator routes each message to one of two workers:

- **Advisor:** curriculum retrieval with self-reflection, plus read-only learner database tools.
- **Assessor:** sandboxed code execution, structured grading, and an evaluator-review-refinement loop.

The curriculum retriever uses ChromaDB when an OpenAI key and the relevant packages are available. It falls back to a local BM25 lexical index if vector retrieval is unavailable or exceeds its time budget.

## Requirements

- Python 3.10 or newer
- An OpenAI API key for model-powered responses (optional; offline fallbacks are available)

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/farida596/sprints-learning-assistant.git
   cd sprints-learning-assistant
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   macOS / Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create your local environment file:

   ```bash
   ```powershell
   # Windows PowerShell
   Copy-Item .env.example .env
   ```

   ```bash
   # macOS / Linux
   cp .env.example .env
   ```

   Add your own API key to `.env`. Never commit this file.

5. Start a chat using a sample learner ID:

   ```bash
   python -m src.main --chat SPR-1013
   ```

   Type `exit` to end the session.

The first run creates the local SQLite database from the anonymized CSV file. To rebuild it manually:

```bash
python -m src.main --rebuild-db --chat SPR-1013
```

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The graph integration tests require LangGraph. They are skipped automatically if it is not installed.

## Repository structure

```text
.
├── data/
│   ├── curriculum_docs/       # Included course material
│   ├── sprints_learner_db.csv # Anonymized sample learner records
│   └── README.md
├── src/
│   ├── agents/
│   │   ├── advisor_agent.py
│   │   ├── assessor_agent.py
│   │   └── orchestrator.py
│   ├── tools/
│   │   ├── code_repl.py
│   │   ├── db_queries.py
│   │   └── rag_retriever.py
│   ├── config.py
│   ├── logic.py
│   ├── main.py
│   └── state.py
├── tests/
├── .env.example
├── .gitignore
└── requirements.txt
```

## Data and security notes

- The included learner names are anonymized sample labels. Do not replace them with real learner data unless you have authorization.
- The SQLite database and ChromaDB index are generated locally and are not committed.
- Keep API keys in your local `.env` file. Do not commit secrets.
- The Python execution tool applies restrictions and resource limits, but running arbitrary user code always requires careful security review before production deployment.
- The in-memory LangGraph checkpointer is intended for this sample application; production deployments should use persistent, access-controlled storage.

---

Built as part of the Sprints learning experience.
