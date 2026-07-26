
# Medical Research Assistant (RAG)

A retrieval-augmented chatbot that answers questions from medical research papers you upload — it only responds based on what's actually in the PDFs, not from whatever the underlying model already "knows." If the answer isn't in the documents, it says so instead of making something up.

Live demo: https://medicalresearchassistant-2jamtgims7kgasryzsy9cz.streamlit.app/

## Why I built this

I wanted a way to quickly query a stack of research papers without manually skimming through all of them, while also being confident the answers weren't hallucinated. So instead of just hitting an LLM directly, this pulls relevant chunks from the actual uploaded documents first, then forces the model to answer strictly from that context and cite where it got the info.

## What it does

- Upload one or more PDFs from the browser (no need to drop files into a folder manually)
- Text gets split into chunks and embedded, then stored in a Chroma vector DB
- Ask a question, it retrieves the most relevant + diverse chunks (using MMR so you don't get 5 near-duplicate passages)
- The LLM answers using only that retrieved context, with sources (filename + page number) listed at the end
- Chat history sticks around for the session, with a button to clear it

## Stack

- Streamlit for the UI
- LangChain for the orchestration/pipeline
- Mistral (`mistral-small-2506`) as the LLM
- `all-MiniLM-L6-v2` for embeddings
- Chroma as the vector store
- `PyPDFLoader` for reading PDFs

## Running it yourself

```bash
git clone https://github.com/Junaeid-Ali/Medical_Research_Assistant.git
cd Medical_Research_Assistant

python -m venv .venv
.venv\Scripts\activate      # or source .venv/bin/activate on mac/linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own keys:

```
MISTRAL_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

Then:

```bash
streamlit run app.py
```

## Deploying

Deployed on Streamlit Community Cloud — point it at `app.py`, add the same env vars under Settings → Secrets instead of a `.env` file.

Heads up: Streamlit Cloud's storage isn't persistent, so the vector DB gets rebuilt whenever the app restarts. If you upload PDFs, process them, then the app sleeps/restarts, you'll need to re-upload. Wasn't worth setting up external storage for a project this size, but if you're forking this for something bigger, swapping Chroma for a hosted vector DB (Pinecone, Qdrant Cloud, etc.) would fix that.

Also — if you ever hit an `AttributeError` around `PyExtensionType` on deploy, that's a known `pyarrow` 21+ breaking change with the `datasets` library (pulled in transitively through `sentence-transformers`). Already pinned `pyarrow<21` in requirements.txt to avoid it, but flagging it here in case it resurfaces after a dependency update.

## Disclaimer

This is a research/learning tool, not a medical device. It doesn't give medical advice or clinical recommendations — don't use it for anything beyond exploring what's in your own documents.
