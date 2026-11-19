import os, io, re, json, time, base64, hashlib, sqlite3, subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
import psutil

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "local_ai_v2.db"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "qwen2.5:3b")
DEFAULT_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "moondream")

st.set_page_config(page_title="Local AI Assistant", page_icon="🤖", layout="wide")

# ---------- Database ----------
@contextmanager
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    with db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL, created TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, sha TEXT UNIQUE NOT NULL, content TEXT NOT NULL, created TEXT NOT NULL)")


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_chat(title="New chat"):
    with db() as con:
        cur = con.execute("INSERT INTO chats(title,created,updated) VALUES(?,?,?)", (title, now(), now()))
        return cur.lastrowid


def get_chats():
    with db() as con:
        return con.execute("SELECT * FROM chats ORDER BY updated DESC").fetchall()


def get_messages(chat_id):
    with db() as con:
        return con.execute("SELECT role,content FROM messages WHERE chat_id=? ORDER BY id", (chat_id,)).fetchall()


def save_message(chat_id, role, content):
    with db() as con:
        con.execute("INSERT INTO messages(chat_id,role,content,created) VALUES(?,?,?,?)", (chat_id, role, content, now()))
        con.execute("UPDATE chats SET updated=? WHERE id=?", (now(), chat_id))


def rename_chat(chat_id, title):
    with db() as con:
        con.execute("UPDATE chats SET title=?,updated=? WHERE id=?", (title[:80], now(), chat_id))


def delete_chat(chat_id):
    with db() as con:
        con.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        con.execute("DELETE FROM chats WHERE id=?", (chat_id,))


def save_memory(content):
    content = content.strip()
    if content:
        with db() as con:
            con.execute("INSERT INTO memories(content,created) VALUES(?,?)", (content, now()))


def get_memories():
    with db() as con:
        return con.execute("SELECT * FROM memories ORDER BY id DESC LIMIT 30").fetchall()


def delete_memory(memory_id):
    with db() as con:
        con.execute("DELETE FROM memories WHERE id=?", (memory_id,))


def save_document(name, content, sha):
    with db() as con:
        con.execute("INSERT OR IGNORE INTO documents(name,sha,content,created) VALUES(?,?,?,?)", (name, sha, content, now()))


def get_documents():
    with db() as con:
        return con.execute("SELECT id,name,sha,created FROM documents ORDER BY id DESC").fetchall()


def delete_document(doc_id):
    with db() as con:
        con.execute("DELETE FROM documents WHERE id=?", (doc_id,))


def retrieve_document_chunks(query, limit=4):
    terms = [t for t in re.findall(r"[a-zA-Z0-9_]{3,}", query.lower()) if t not in {"the","and","for","with","that","this","from","what","when","where","which","your","you","are"}]
    if not terms:
        return []
    with db() as con:
        docs = con.execute("SELECT name,content FROM documents").fetchall()
    hits = []
    for doc in docs:
        chunks = [c.strip() for c in re.split(r"\n\s*\n", doc["content"]) if c.strip()]
        if not chunks:
            chunks = [doc["content"]]
        for chunk in chunks:
            low = chunk.lower()
            score = sum(low.count(term) for term in terms)
            if score:
                hits.append((score, doc["name"], chunk[:2500]))
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[:limit]


init_db()

# ---------- Helpers ----------
def ollama_get(path, timeout=5):
    return requests.get(f"{OLLAMA_URL}{path}", timeout=timeout)


def installed_models():
    try:
        data = ollama_get("/api/tags", timeout=4).json()
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def extract_text(upload):
    name = upload.name.lower()
    raw = upload.getvalue()
    if name.endswith(".txt") or name.endswith(".md") or name.endswith(".csv"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(raw))
        return "\n\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Supported files: PDF, DOCX, TXT, MD and CSV")


def web_search(query, max_results=5):
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        return list(results)
    except Exception as e:
        return [{"title": "Web search unavailable", "body": f"Install/update ddgs or check internet. Details: {e}", "href": ""}]


def hardware_stats():
    stats = {"CPU": psutil.cpu_percent(interval=0.25), "RAM used": psutil.virtual_memory().percent, "RAM total GB": round(psutil.virtual_memory().total / (1024**3), 1)}
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
        p = [x.strip() for x in out.split(",")]
        if len(p) >= 5:
            stats.update({"GPU": p[0], "GPU use %": p[1], "VRAM used MB": p[2], "VRAM total MB": p[3], "GPU temp C": p[4]})
    except Exception:
        stats["GPU"] = "nvidia-smi unavailable / driver not detected"
    return stats


def stream_ollama(model, messages, temperature=0.7):
    payload = {"model": model, "messages": messages, "stream": True,
               "options": {"temperature": temperature, "num_ctx": 4096}}
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(10, 600))
    response.raise_for_status()
    full = ""
    final_data = {}
    for line in response.iter_lines():
        if not line:
            continue
        item = json.loads(line.decode("utf-8"))
        piece = item.get("message", {}).get("content", "")
        if piece:
            full += piece
            yield piece, None
        if item.get("done"):
            final_data = item
    yield "", final_data


def make_system_prompt(use_docs, query):
    parts = ["You are a helpful private local AI assistant. Be accurate, explain uncertainty, and use clear beginner-friendly language."]
    memories = get_memories()
    if memories:
        parts.append("User-approved saved notes (use only when relevant):\n" + "\n".join("- " + m["content"] for m in memories[:15]))
    if use_docs:
        chunks = retrieve_document_chunks(query)
        if chunks:
            ctx = "\n\n".join(f"[Document: {name}]\n{chunk}" for _, name, chunk in chunks)
            parts.append("Relevant excerpts from uploaded files. Base file-specific answers on these excerpts and say if the answer is not present:\n" + ctx)
    return "\n\n".join(parts)


# ---------- Optional password gate ----------
password = os.getenv("LOCAL_AI_PASSWORD", "")
if password:
    if not st.session_state.get("authenticated", False):
        st.title("🔐 Local AI Assistant")
        entered = st.text_input("Enter your local app password", type="password")
        if st.button("Unlock", type="primary"):
            if entered == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()
else:
    st.sidebar.warning("LAN access has no password. Set LOCAL_AI_PASSWORD before using this on a shared network.")

# ---------- Session / sidebar ----------
if "active_chat" not in st.session_state:
    chats = get_chats()
    st.session_state.active_chat = chats[0]["id"] if chats else create_chat()

st.sidebar.title("🤖 Local AI")
if st.sidebar.button("＋ New chat", use_container_width=True):
    st.session_state.active_chat = create_chat()
    st.rerun()

chats = get_chats()
if chats:
    labels = {f"{c['title']}  ·  #{c['id']}": c["id"] for c in chats}
    current_label = next((k for k,v in labels.items() if v == st.session_state.active_chat), list(labels)[0])
    selected = st.sidebar.selectbox("Chat history", list(labels.keys()), index=list(labels).index(current_label))
    if labels[selected] != st.session_state.active_chat:
        st.session_state.active_chat = labels[selected]
        st.rerun()

active = st.session_state.active_chat
current_row = next((c for c in get_chats() if c["id"] == active), None)
with st.sidebar.expander("Rename / delete chat"):
    new_title = st.text_input("Chat title", value=current_row["title"] if current_row else "New chat", key=f"rename_{active}")
    if st.button("Save title"):
        rename_chat(active, new_title.strip() or "New chat")
        st.rerun()
    if st.button("Delete this chat", type="secondary"):
        delete_chat(active)
        st.session_state.active_chat = create_chat()
        st.rerun()

models = installed_models()
st.sidebar.subheader("Model settings")
text_models = models or [DEFAULT_TEXT_MODEL]
text_default_index = text_models.index(DEFAULT_TEXT_MODEL) if DEFAULT_TEXT_MODEL in text_models else 0
text_model = st.sidebar.selectbox("Text model", text_models, index=text_default_index)
vision_candidates = [m for m in models if any(x in m.lower() for x in ["moondream", "llava", "gemma3", "qwen3.5"])]
if DEFAULT_VISION_MODEL not in vision_candidates:
    vision_candidates.insert(0, DEFAULT_VISION_MODEL)
vision_model = st.sidebar.selectbox("Vision model", list(dict.fromkeys(vision_candidates)))
use_docs = st.sidebar.checkbox("Use uploaded document context", value=True)
use_web = st.sidebar.checkbox("Search the web for this question", value=False)
temperature = st.sidebar.slider("Creativity", 0.0, 1.2, 0.7, 0.1)

st.sidebar.divider()
st.sidebar.subheader("📄 Documents")
files = st.sidebar.file_uploader("Upload PDF / DOCX / TXT / MD / CSV", type=["pdf","docx","txt","md","csv"], accept_multiple_files=True)
if files:
    for f in files:
        raw = f.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        try:
            content = extract_text(f)
            if content.strip():
                save_document(f.name, content, sha)
                st.sidebar.success(f"Added: {f.name}")
            else:
                st.sidebar.warning(f"No extractable text: {f.name}")
        except Exception as e:
            st.sidebar.error(f"{f.name}: {e}")

docs = get_documents()
if docs:
    with st.sidebar.expander(f"Saved documents ({len(docs)})"):
        for d in docs:
            c1, c2 = st.columns([4,1])
            c1.caption(d["name"])
            if c2.button("✕", key=f"doc_{d['id']}"):
                delete_document(d["id"])
                st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🧠 Long-term memory")
mem_text = st.sidebar.text_area("Save a note for future chats", placeholder="Example: I prefer explanations with simple C++ examples.", height=90)
if st.sidebar.button("Save memory"):
    save_memory(mem_text)
    st.sidebar.success("Memory saved")
    st.rerun()
memories = get_memories()
with st.sidebar.expander(f"Saved notes ({len(memories)})"):
    for m in memories:
        st.caption(m["content"][:180])
        if st.button("Delete note", key=f"mem_{m['id']}"):
            delete_memory(m["id"])
            st.rerun()

# ---------- Main UI ----------
st.title("🤖 Local AI Assistant")
st.caption("Runs on your laptop with Ollama. Chat history, local documents, saved notes, image questions, optional web search and hardware monitoring.")

with st.expander("📊 Hardware monitor", expanded=False):
    stats = hardware_stats()
    cols = st.columns(4)
    for i, (k,v) in enumerate(stats.items()):
        cols[i % 4].metric(k, str(v))
    if st.button("Refresh hardware stats"):
        st.rerun()

if not models:
    st.warning("Cannot find Ollama models. Make sure Ollama is running and at least one model is installed. Try: ollama list")

# Render conversation
for msg in get_messages(active):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

image_file = st.file_uploader("Optional image for vision questions", type=["png","jpg","jpeg","webp"], key=f"image_{active}")
prompt = st.chat_input("Message your local AI...")
if prompt:
    save_message(active, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
        if image_file:
            st.image(image_file, width=320)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            # Optional web results are fetched and passed as context, not treated as trusted instructions.
            web_context = ""
            if use_web:
                results = web_search(prompt)
                web_context = "\n\nWeb search results (verify claims; sources may be incomplete):\n" + "\n".join(
                    f"- {r.get('title','')} | {r.get('body','')} | {r.get('href','')}" for r in results)
            history = [{"role": r["role"], "content": r["content"]} for r in get_messages(active)][-12:]
            system = make_system_prompt(use_docs, prompt) + web_context
            messages = [{"role":"system", "content":system}] + history
            if image_file:
                img_b64 = base64.b64encode(image_file.getvalue()).decode("ascii")
                messages[-1] = {"role":"user", "content":prompt, "images":[img_b64]}
                chosen_model = vision_model
            else:
                chosen_model = text_model
            answer = ""
            metrics = {}
            for piece, final_data in stream_ollama(chosen_model, messages, temperature):
                if piece:
                    answer += piece
                    placeholder.markdown(answer + "▌")
                if final_data is not None:
                    metrics = final_data
            placeholder.markdown(answer)
            save_message(active, "assistant", answer)
            eval_count = metrics.get("eval_count")
            eval_duration = metrics.get("eval_duration")
            if eval_count and eval_duration:
                tok_s = eval_count / (eval_duration / 1e9)
                st.caption(f"Generated {eval_count} tokens · approximately {tok_s:.1f} tokens/sec · model: {chosen_model}")
            else:
                st.caption(f"Model: {chosen_model}")
        except requests.exceptions.ConnectionError:
            placeholder.error("Cannot connect to Ollama. Open Ollama, then test `ollama list` in CMD.")
        except Exception as e:
            placeholder.error(f"Error: {e}")

st.divider()
st.caption("Privacy: chats, saved notes and extracted document text are stored in local_ai_v2.db beside this app. Web search sends your search query to the search provider. Do not upload sensitive files unless you are comfortable storing their extracted text locally.")
