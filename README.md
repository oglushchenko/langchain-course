# langchain-course

This project can run with either OpenAI or a local Ollama model.

## Use local `gemma3:270M-F16` (Docker)

1. Start your local model server in Docker so it is reachable at `http://localhost:12434`.
2. Create/update your `.env` file:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma3:latest
OLLAMA_BASE_URL=http://localhost:12434
```

3. Install dependencies and run:

```powershell
uv sync
uv run python main.py
```

## Use OpenAI (default)

If `LLM_PROVIDER` is not set to `ollama`, the app uses OpenAI.

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5
```
