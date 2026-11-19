# Local AI Assistant V2 (Windows + Ollama)

Features: chat history, rename/delete chats, PDF/DOCX/TXT/MD/CSV text extraction, simple keyword document retrieval, manually saved long-term notes, image questions through a separate Ollama vision model, optional DuckDuckGo web search, CPU/RAM/NVIDIA GPU monitor, approximate tokens/second, and LAN access through Streamlit.

## Recommended setup for GTX 1650 4GB + 12GB RAM

- Keep `qwen2.5:3b` as the reliable starting text model.
- Optional text upgrade to test: `qwen3:4b` (about 2.5 GB quantized download in Ollama library). It may be slower or partly CPU-offloaded on a 4GB GPU.
- Vision model: `moondream` (about 1.7 GB). It is designed as a small edge vision model, but can still be slow on this laptop.
- Do not load/run both models simultaneously; Ollama may keep models resident. If memory is tight, close other GPU-heavy apps and set `OLLAMA_KEEP_ALIVE=0` in the Ollama environment or unload models using `ollama stop MODEL`.

## Install

1. Install Ollama for Windows and ensure it is running.
2. Open CMD in this folder.
3. Run:

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. In a separate CMD window, pull models:

   ```cmd
   ollama pull qwen2.5:3b
   ollama pull moondream
   ```

   Optional text model comparison:

   ```cmd
   ollama pull qwen3:4b
   ```

5. Start the app by double-clicking `run_app.bat`, or run:

   ```cmd
   .venv\Scripts\activate
   python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```

6. On the laptop open `http://localhost:8501`. On a phone connected to the same Wi-Fi/hotspot, use `http://LAPTOP_IPV4:8501`.

## Set a password before LAN use

In CMD, before starting Streamlit:

```cmd
set LOCAL_AI_PASSWORD=Choose-A-Long-Private-Password
run_app.bat
```

This password is checked by the Streamlit app. Do not expose port 8501 or Ollama port 11434 to the public internet. Do not set up router port forwarding. Only use LAN access on a network you trust.

## Notes / limitations

- Document retrieval is lightweight keyword matching, not a full vector database/RAG pipeline. Scanned PDFs need OCR and are not read by this version unless they contain a text layer.
- Web search is optional and requires internet. Search queries go to DuckDuckGo via `ddgs`.
- Saved memory is manually entered by you; the app does not silently learn personal details.
- Images are sent to the selected local Ollama vision model; they are not stored in chat history by this app.
- The GPU monitor uses `nvidia-smi`; if unavailable, CPU/RAM stats still work.
- This is a local personal project, not a hardened multi-user production service. Keep Windows firewall limited to trusted private networks.
