# Deploying Cortexa Chunk Intelligence Demo

GitHub username: `Sidpav`
Suggested repository name: `cortexa-chunk-intelligence-demo`

## 1. Create the GitHub repo

Create a new GitHub repository under:

```text
https://github.com/Sidpav/cortexa-chunk-intelligence-demo
```

Keep the repo public or private based on your company preference.

## 2. Push this folder to GitHub

Open terminal inside this folder and run:

```bash
git init
git add .
git commit -m "Initial Cortexa chunk intelligence demo"
git branch -M main
git remote add origin https://github.com/Sidpav/cortexa-chunk-intelligence-demo.git
git push -u origin main
```

If `remote origin already exists`, run:

```bash
git remote set-url origin https://github.com/Sidpav/cortexa-chunk-intelligence-demo.git
git push -u origin main
```

## 3. Deploy on Streamlit Cloud

Go to Streamlit Community Cloud and create a new app.

Use:

```text
Repository: Sidpav/cortexa-chunk-intelligence-demo
Branch: main
Main file path: app.py
```

## 4. Start with mock mode

In Streamlit Cloud app settings, open Secrets and add:

```toml
LLM_BACKEND = "mock"
```

Reboot the app. This confirms the online UI, upload, chunking, retrieval, inconsistency, and novelty panels work.

## 5. Use a real online model

For OpenRouter:

```toml
LLM_BACKEND = "openrouter"
OPENROUTER_API_KEY = "your_openrouter_api_key"
OPENROUTER_MODEL = "qwen/qwen-2.5-7b-instruct"
```

For Groq:

```toml
LLM_BACKEND = "groq"
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

Do not use Ollama for Streamlit Cloud deployment. Ollama is local-only unless hosted separately.
