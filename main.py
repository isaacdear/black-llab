import os
import re
import sys
import shutil
from dotenv import load_dotenv

# THIS IS THE MISSING MAGIC BUTTON:
load_dotenv()

import json
import time
import asyncio
import hashlib
import base64
import secrets
import subprocess
from collections import deque
from io import BytesIO
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple

import httpx
import uvicorn
import psutil
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from PIL import Image

# -------------------------
# Optional dependencies
# -------------------------
try:
    from pix2text import Pix2Text
    _P2T_AVAILABLE = True
except Exception:
    Pix2Text = None
    _P2T_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    _DDG_AVAILABLE = True
except Exception:
    DDGS = None
    _DDG_AVAILABLE = False

try:
    import trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    _TRAFILATURA_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    print(">> [BOOTSTRAP] Booting Cross-Encoder (Sniper Retrieval)...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    _CROSS_ENCODER_AVAILABLE = True
except Exception as e:
    print(f">> [WARNING] Cross-Encoder not available: {e}")
    cross_encoder = None
    _CROSS_ENCODER_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

import networkx as nx
from networkx.readwrite import json_graph

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(CURRENT_DIR, "obsidian_graph.json")

try:
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
        obsidian_graph = json_graph.node_link_graph(graph_data)
        print(f">> [BOOTSTRAP] Knowledge Graph loaded with {obsidian_graph.number_of_nodes()} nodes.")
except Exception:
    obsidian_graph = nx.Graph()
    print(">> [BOOTSTRAP] Initialized fresh Knowledge Graph.")

def save_graph():
    try:
        data = json_graph.node_link_data(obsidian_graph)
        with open(GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f">> [GRAPH WARN] Failed to save graph: {e}")

import chromadb
import uuid
import PyPDF2
from io import BytesIO

# NEW VISION IMPORTS
from pdf2image import convert_from_bytes
import numpy as np

_CHROMA_AVAILABLE = True

# Spin up the local vector database
chroma_client = chromadb.PersistentClient(path="./nexus_cache")
doc_collection = chroma_client.get_or_create_collection(name="rag_documents")

# Boot up the heavy Vision model
if _P2T_AVAILABLE:
    print(">> [BOOTSTRAP] Booting Pix2Text Vision Engine (CPU Stability Mode)...")
    # Explicitly disabling CoreML to prevent the 'Error in building plan' crash on Mac
    p2t = Pix2Text.from_config(device='cpu')
else:
    p2t = None

# ============================================================
# Local-only safety defaults & Configuration
# ============================================================

HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "8000"))
NEXUS_TOKEN = os.getenv("NEXUS_TOKEN", "__NEXUS_TOKEN__")

MAX_REQUEST_BYTES = int(os.getenv("NEXUS_MAX_REQUEST_BYTES", str(25 * 1024 * 1024)))
MAX_TEXT_CHARS = int(os.getenv("NEXUS_MAX_TEXT_CHARS", "128000"))
MAX_IMAGE_B64_CHARS = int(os.getenv("NEXUS_MAX_IMAGE_B64_CHARS", str(3 * 1024 * 1024)))
MAX_IMAGES_PER_MESSAGE = 10 # <--- Increased to 10

CHAT_RPM = int(os.getenv("NEXUS_CHAT_RPM", "20"))
BUDGET_RPM = int(os.getenv("NEXUS_BUDGET_RPM", "60"))

SESSION_IDLE_TTL_SECONDS = int(os.getenv("NEXUS_SESSION_TTL", str(365 * 24 * 60 * 60))) # Keeps history for 1 year
MAX_SESSION_MESSAGES = int(os.getenv("NEXUS_MAX_SESSION_MESSAGES", "36"))
SUMMARIZE_TRIGGER_MESSAGES = int(os.getenv("NEXUS_SUMMARY_TRIGGER", "14"))
KEEP_LAST_AFTER_SUMMARY = int(os.getenv("NEXUS_KEEP_LAST", "8"))

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/api")
KEEPALIVE_SECONDS = {
    "router": int(os.getenv("NEXUS_KEEPALIVE_ROUTER", "3600")),
    "daily": int(os.getenv("NEXUS_KEEPALIVE_DAILY", "3600")),
    "giant": int(os.getenv("NEXUS_KEEPALIVE_GIANT", "600")),
}
GIANT_IDLE_TTL_SECONDS = int(os.getenv("NEXUS_GIANT_IDLE_TTL", "600"))


MODELS = {
    # The Router is strictly for background complexity grading
    "router": {"name": os.getenv("ROUTER_MODEL", "ministral-3:3b"), "type": "local", "display": "Router (Ministral)"},
    
    "giant":  {"name": os.getenv("GIANT_MODEL", "qwen3.5:9b"), "type": "local", "display": "Giant (Qwen 9B)"},
    
    "openclaw": {"name": "openclaw", "type": "agent", "display": "OpenClaw Agent"},
    # Cloud Fail-safes
    "opus":   {"name": os.getenv("OPUS_MODEL", "claude-opus-4-6"), "type": "anthropic", "display": "Claude Opus 4.6"},
    "glm":    {"name": os.getenv("GLM_MODEL", "glm-5"), "type": "openai_compat", "display": "GLM"},
    "search": {"name": os.getenv("SONAR_MODEL", "sonar-pro"), "type": "openai_compat", "display": "Perplexity Sonar"},
    "xiaomi": {
        "name": "mimo-v2-flash", 
        "type": "openai_compat", 
        "display": "Xiaomi MiMo Flash"
    },
}

ANTHROPIC_MESSAGES_URL = os.getenv("ANTHROPIC_MESSAGES_URL", "https://api.anthropic.com/v1/messages")
PERPLEXITY_CHAT_URL = os.getenv("PERPLEXITY_CHAT_URL", "https://api.perplexity.ai/chat/completions")
ZHIPU_CHAT_URL = os.getenv("ZHIPU_CHAT_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
XIAOMI_CHAT_URL = os.getenv("XIAOMI_CHAT_URL", "https://api.xiaomimimo.com/v1/chat/completions")

API_KEYS = {
    "opus": os.getenv("ANTHROPIC_API_KEY", ""),
    "glm": os.getenv("ZHIPU_API_KEY", ""),
    "search": os.getenv("PERPLEXITY_API_KEY", ""),
    "xiaomi": os.getenv("XIAOMI_API_KEY", ""),
}

API_URLS = {
    "opus": ANTHROPIC_MESSAGES_URL,
    "glm": ZHIPU_CHAT_URL,
    "search": PERPLEXITY_CHAT_URL,
    "xiaomi": XIAOMI_CHAT_URL,
}

BUDGET_FILE = os.getenv("BUDGET_FILE", "budget.json")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_FILE = os.path.join(BASE_DIR, os.getenv("NEXUS_SESSIONS_FILE", "sessions.json"))
PRICING = {"opus": 0.02, "glm": 0.002, "xiaomi": 0.0002, "search": 0.005}

NEXUS_CACHE = os.getenv("NEXUS_CACHE", "1") != "0"
CHROMA_PATH = os.getenv("CHROMA_PATH", "./nexus_cache")
cache_collection = None
memory_collection = None 

if _CHROMA_AVAILABLE:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    if NEXUS_CACHE:
        cache_collection = chroma_client.get_or_create_collection(name="prompt_cache")
    memory_collection = chroma_client.get_or_create_collection(name="obsidian_memory")

CACHE_TTL_DAYS = int(os.getenv("NEXUS_CACHE_TTL_DAYS", "14"))
CACHE_TTL_SECONDS = CACHE_TTL_DAYS * 24 * 60 * 60
CACHE_CLEANUP_INTERVAL_SECONDS = int(os.getenv("NEXUS_CACHE_CLEANUP_INTERVAL", "3600"))

local_compute_lock = asyncio.Semaphore(1)
session_io_lock = asyncio.Lock()
budget_io_lock = asyncio.Lock()


def now_ts() -> float:
    return time.time()

async def scrape_url_async(url: str) -> str:
    """Silently fetches a URL and rips the FULL article text out, ignoring ads/navbars."""
    if not _TRAFILATURA_AVAILABLE: return ""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            res = await client.get(url)
            if res.status_code == 200:
                content = await run_in_thread(
                    lambda: trafilatura.extract(
                        res.text, include_links=False, include_images=False, favor_precision=True
                    )
                )
                if content:
                    return minify_text(content)
    except Exception as e:
        print(f">> [Trafilatura] Failed to scrape {url}: {type(e).__name__}")
    return ""

async def scrape_js_url_async(url: str) -> str:
    """Boots an invisible Chromium browser to render JS-heavy sites, then rips the HTML."""
    if not _PLAYWRIGHT_AVAILABLE or not _TRAFILATURA_AVAILABLE: return ""
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox', 
                    '--disable-setuid-sandbox', 
                    '--disable-dev-shm-usage',
                    '--disable-gpu'
                ]
            )
            
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            async def route_intercept(route):
                try:
                    if route.request.resource_type in ["image", "media", "font", "stylesheet"]:
                        await route.abort()
                    else:
                        await route.continue_()
                except Exception:
                    pass
                    
            await page.route("**/*", route_intercept)
            
            # Wait until the network is mostly idle (meaning React/Vue has finished rendering the DOM)
            await page.goto(url, wait_until="networkidle", timeout=15000)
            
            # Extract the raw, fully-rendered HTML
            raw_html = await page.content()
            await browser.close()
            
            # Pass the rendered HTML back to Trafilatura to strip the navbars and format to Markdown
            content = await run_in_thread(
                lambda: trafilatura.extract(
                    raw_html, include_links=False, include_images=False, favor_precision=True
                )
            )
            
            if content:
                return minify_text(content)
                
    except Exception as e:
        print(f">> [Playwright] Failed to render {url}: {type(e).__name__}")
        
    return ""

async def smart_scrape_async(url: str) -> str:
    """Tries the fast HTTP scrape first. If it hits a JS-wall, deploys the headless browser."""

    content = await scrape_url_async(url)
    
    if _PLAYWRIGHT_AVAILABLE and (not content or len(content) < 500):
        print(f">> [Smart Scraper] JS-Wall detected on {url}. Engaging Headless Chromium...")
        js_content = await scrape_js_url_async(url)
        if js_content:
            return js_content
            
    return content


def safe_json_loads(s: str, default: Any) -> Any:
    try:
        s = re.sub(r'```json\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'```', '', s).strip()
        return json.loads(s)
    except Exception:
        return default

def sse_pack(obj: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

def sse_done() -> bytes:
    return b"data: [DONE]\n\n"

def strip_data_url_prefix(b64_or_dataurl: str) -> str:
    if "," in b64_or_dataurl and "base64" in b64_or_dataurl[:64]:
        return b64_or_dataurl.split(",", 1)[1]
    return b64_or_dataurl

def compress_image_base64(base64_str: str, max_size: int = 1024) -> str:
    try:
        base64_str = strip_data_url_prefix(base64_str)
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return base64_str

async def run_in_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

def _require_token(request: Request) -> None:
    token = request.headers.get("X-Nexus-Token", "")
    if not token or token != request.app.state.nexus_token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:] == request.app.state.nexus_token:
            return
        raise HTTPException(status_code=401, detail="Unauthorized")

def _enforce_payload_limits(request: Request) -> None:
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="Payload too large")
        except ValueError:
            pass

def _validate_message_shape(msg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(msg, dict):
        raise HTTPException(status_code=400, detail="Invalid message object")
    role = msg.get("role")
    content = msg.get("content", "")
    images = msg.get("images")
    if role not in ("user", "assistant", "system"):
        raise HTTPException(status_code=400, detail="Invalid role")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="Invalid content type")
    if len(content) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"Message too long (>{MAX_TEXT_CHARS} chars)")
    if images is not None:
        if not isinstance(images, list):
            raise HTTPException(status_code=400, detail="Invalid images format")
        if len(images) > MAX_IMAGES_PER_MESSAGE:
            raise HTTPException(status_code=400, detail=f"Too many images (max {MAX_IMAGES_PER_MESSAGE})")
        for im in images:
            if not isinstance(im, str):
                raise HTTPException(status_code=400, detail="Invalid image encoding")
            if len(im) > MAX_IMAGE_B64_CHARS:
                raise HTTPException(status_code=400, detail="Image too large (base64)")
    out = {"role": role, "content": content}
    if images:
        out["images"] = images
    return out

def minify_text(text: str) -> str:
    """Aggressively removes token-wasting junk from raw text rips."""
    # 1. Collapse 3+ newlines into just 2 (standard paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 2. Collapse multiple spaces/tabs into a single space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # 3. Remove excessive dot leaders (e.g., in Table of Contents)
    text = re.sub(r'\.{4,}', '...', text)
    return text.strip()

def semantic_chunking(text: str, max_chunk_len: int = 1500, overlap_len: int = 300) -> List[str]:
    """Upgraded chunker with overlapping windows and safe table-row splitting."""
    raw_paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = ""
    
    for p in raw_paragraphs:
        p = p.strip()
        if not p: continue
        
        # 1. If it fits comfortably, add it to the current chunk
        if len(current_chunk) + len(p) < max_chunk_len:
            current_chunk += "\n\n" + p if current_chunk else p
            
        else:
            # Chunk is full. Save it.
            if current_chunk: 
                chunks.append(current_chunk)
            
            # 2. Handle giant blocks (like massive Markdown tables) gracefully
            if len(p) > max_chunk_len:
                # Split by single newlines (table rows) so we don't sever Markdown syntax mid-cell!
                lines = p.split('\n')
                current_chunk = ""
                for line in lines:
                    if len(current_chunk) + len(line) < max_chunk_len:
                        current_chunk += "\n" + line if current_chunk else line
                    else:
                        if current_chunk: chunks.append(current_chunk)
                        current_chunk = line # Start new chunk with the overflow row
            else:
                # 3. SLIDING WINDOW OVERLAP: Carry over the end of the last chunk
                overlap_text = current_chunk[-overlap_len:] if current_chunk else ""
                
                # Snap the overlap to the nearest whole word so we don't cut words in half
                if " " in overlap_text:
                    overlap_text = overlap_text[overlap_text.find(" ")+1:] 
                    
                current_chunk = overlap_text + "\n\n" + p
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

async def background_raptor_summaries(app: FastAPI, chunks: List[str], filename: str):
    """Takes semantic chunks, asks Qwen to summarize them, AND builds the Knowledge Graph."""
    if not doc_collection or not chunks: return
    
    print(f">> [RAPTOR & GRAPH] Initiating background processing for {filename}...")
    
    block_size = 2
    blocks = ["\n\n".join(chunks[i:i + block_size]) for i in range(0, len(chunks), block_size)]
    
    # Setup for Xiaomi Graph Extraction
    api_url = API_URLS.get("xiaomi")
    api_key = API_KEYS.get("xiaomi")
    
    for i, block_text in enumerate(blocks):
        
        prompt = (
            f"You are a data extraction AI. Read the following document excerpt and write a "
            f"concise, 3-4 sentence summary of the core facts. Ignore filler.\n\n"
            f"EXCERPT:\n{block_text}"
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = await app.state.ollama.post("/generate", json={
                    "model": MODELS["giant"]["name"], 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {"temperature": 0.1}
                }, timeout=600.0)
                
                summary_text = res.json().get("response", "").strip()
                
                if summary_text:
                    final_summary = f"[RAPTOR SUMMARY OF {filename} Part {i+1}]: {summary_text}"
                    
                    cid = f"raptor_{filename}_{uuid.uuid4().hex[:8]}"
                    meta = {"source": filename, "type": "raptor_summary", "block": i}
                    
                    await run_in_thread(lambda: doc_collection.add(
                        documents=[final_summary], metadatas=[meta], ids=[cid]
                    ))
                    
                break 
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f">> [RAPTOR WARN] Ollama busy/timeout on block {i}. Retrying ({attempt + 1}/{max_retries}) in 5s...")
                    await asyncio.sleep(5)
                else:
                    print(f">> [RAPTOR ERROR] Failed to summarize block {i} of {filename}: {repr(e)}")

        # ==========================================
        # TASK 2: GRAPH RAG EXTRACTION (Xiaomi MiMo)
        # ==========================================
        if api_key:
            graph_prompt = f"""Extract 3 to 5 core relationship triples from the text below.
Focus on people, organizations, concepts, and how they interact.
You MUST output ONLY a valid JSON array of objects in this exact format:
[ {{"source": "Entity 1", "target": "Entity 2", "relation": "description of relationship"}} ]

TEXT:
{block_text}"""
            
            try:
                async with httpx.AsyncClient() as client:
                    res = await client.post(
                        api_url, 
                        headers={"Authorization": f"Bearer {api_key}"}, 
                        json={
                            "model": MODELS["xiaomi"]["name"], 
                            "messages": [{"role": "user", "content": graph_prompt}], 
                            "temperature": 0.1
                        },
                        timeout=30.0
                    )
                    
                    # Safely parse the JSON response
                    triples = safe_json_loads(res.json()["choices"][0]["message"]["content"], [])
                    
                    if isinstance(triples, list):
                        for triple in triples:
                            src = str(triple.get("source", "")).strip().upper()
                            tgt = str(triple.get("target", "")).strip().upper()
                            rel = str(triple.get("relation", "")).strip()
                            
                            if src and tgt and rel:
                                obsidian_graph.add_edge(src, tgt, relation=rel, source_file=filename)
                        
                        # Save the updated graph to disk
                        save_graph()
            except Exception as e:
                print(f">> [GRAPH EXTRACT ERROR] Block {i}: {e}")
            
    print(f">> [RAPTOR & GRAPH] Completed hierarchical tree and graph mapping for {filename}.")
            
# ============================================================
# Core Logic (Budget, Limits, Sys Monitor)
# ============================================================

async def get_budget(app: FastAPI) -> Dict[str, Any]:
    async with budget_io_lock:
        if not os.path.exists(BUDGET_FILE):
            data = {"daily_limit": 2.00, "spent": 0.00, "date": time.strftime("%Y-%m-%d")}
            with open(BUDGET_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
            app.state.budget = data
            return data
            
        with open(BUDGET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if data.get("date") != time.strftime("%Y-%m-%d"):
            data["spent"] = 0.00
            data["date"] = time.strftime("%Y-%m-%d")
            with open(BUDGET_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
                
        app.state.budget = data
        return data

async def update_spend(app: FastAPI, model_key: str, input_text: str, output_text: str) -> None:
    if model_key not in PRICING:
        return
    estimated_tokens = (len(input_text) + len(output_text)) / 4
    cost = (estimated_tokens / 1000) * PRICING[model_key]
    
    async with budget_io_lock:
        budget = app.state.budget
        budget["spent"] += cost
        with open(BUDGET_FILE, "w", encoding="utf-8") as f:
            json.dump(budget, f)

def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"

def _rate_allow(app: FastAPI, bucket: str, ip: str, rpm: int) -> bool:
    now = now_ts()
    window = 60.0
    key = f"{bucket}:{ip}"
    if key not in app.state.rate:
        app.state.rate[key] = deque()
    dq = app.state.rate[key]
    cutoff = now - window
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= rpm:
        return False
    dq.append(now)
    return True

async def system_stats_loop(app: FastAPI) -> None:
    while True:
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_gb = mem.available / (1024 ** 3)
            ram_percent = mem.percent 
            app.state.sys = {"cpu": float(cpu), "ram_gb": float(ram_gb), "ram_percent": float(ram_percent), "ts": now_ts()}
        except Exception:
            app.state.sys = {"cpu": 0.0, "ram_gb": 999.0, "ram_percent": 0.0, "ts": now_ts()}
        await asyncio.sleep(2.0)

def is_mac_busy(app: FastAPI) -> bool:
    sys_ = app.state.sys
    cpu = float(sys_.get("cpu", 0.0))
    ram_gb = float(sys_.get("ram_gb", 999.0))
    return cpu > 45.0 or ram_gb < 10.0

# ============================================================
# Session Persistence & Summarization
# ============================================================

async def load_sessions(app: FastAPI) -> None:
    async with session_io_lock:
        if os.path.exists(SESSIONS_FILE):
            try:
                with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                    app.state.sessions = json.load(f)
            except Exception as e:
               print(f">> [WARNING] Could not read sessions file. Starting fresh. Error: {e}")
               app.state.sessions = {}
        else:
            app.state.sessions = {}

async def save_sessions(app: FastAPI) -> None:
    async with session_io_lock:
        try:
            with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(app.state.sessions, f)
        except Exception as e:
            print(f"\n>> ⚠️ [CRITICAL ERROR] Failed to save chat history to disk: {str(e)}\n")

def get_or_create_session(app: FastAPI, session_id: str) -> Dict[str, Any]:
    sess = app.state.sessions.get(session_id)
    if sess is None:
        sess = {"messages": [], "summary": "", "updated": now_ts()}
        app.state.sessions[session_id] = sess
    return sess

async def summarize_incremental(app: FastAPI, existing_summary: str, messages_to_absorb: List[Dict[str, Any]]) -> str:
    absorb_text = "\n".join([f"{m.get('role','')}: {m.get('content','')}" for m in messages_to_absorb if m.get("content")])
    prompt = f"You maintain a rolling, concise conversation summary. Update the summary using NEW MESSAGES.\nKeep key facts, decisions, constraints, and any code snippets. Be compact.\nEXISTING SUMMARY:\n{existing_summary}\nNEW MESSAGES:\n{absorb_text}\nReturn UPDATED SUMMARY ONLY."
    try:
        res = await app.state.ollama.post("/generate", json={"model": MODELS["router"]["name"], "prompt": prompt, "stream": False})
        return (res.json().get("response") or "").strip()
    except Exception:
        return (existing_summary + "\n" + absorb_text)[:8000].strip()

async def session_append_and_maybe_summarize(app: FastAPI, session_id: str, msg: Dict[str, Any]) -> Dict[str, Any]:
    sess = get_or_create_session(app, session_id)
    sess["messages"].append(msg)
    sess["updated"] = now_ts()

    if len(sess["messages"]) > MAX_SESSION_MESSAGES:
        sess["messages"] = sess["messages"][-MAX_SESSION_MESSAGES:]

    if len(sess["messages"]) > SUMMARIZE_TRIGGER_MESSAGES:
        absorb = sess["messages"][:-KEEP_LAST_AFTER_SUMMARY]
        keep = sess["messages"][-KEEP_LAST_AFTER_SUMMARY:]
        sess["summary"] = await summarize_incremental(app, sess.get("summary", ""), absorb)
        sess["messages"] = keep

    await save_sessions(app)
    return sess

def build_messages_for_model(sess_summary: str, sess_messages: List[Dict[str, Any]], eval_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    
    system_instruction = (
        "You are Black LLAB, a highly efficient AI assistant. "
        "CRITICAL RULE: DO NOT repeat or echo the user's instructions back to them. "
        "DO NOT use preambles like 'Here is the email you requested'. "
        "Start your response DIRECTLY with the requested answer, code, or data."
    )

    dynamic_injection = ""
    complexity = eval_data.get("complexity", 30)
    
    if eval_data.get("is_code"):
        dynamic_injection = "\n[SYSTEM OVERRIDE: You are an elite software architect. Output production-ready, modular, and heavily optimized code.]"
    elif complexity > 80:
        dynamic_injection = (
            "\n[SYSTEM OVERRIDE: This is a highly complex analytical task. You MUST use strict Chain-of-Thought reasoning. "
            "Before providing your final answer, you MUST create a <think> tag. Inside it, you MUST extract and quote the EXACT VERBATIM sentences or table rows from the document that contain the data requested. "
            "If the document explicitly lists things, quote the list exactly as written before paraphrasing. "
            "Do not blend names or external concepts. Only output your final, polished response AFTER closing the </think> tag.]"
        )
    elif complexity < 30:
        dynamic_injection = "\n[SYSTEM OVERRIDE: Respond as briefly and conversationally as possible.]"

    system_instruction += dynamic_injection
    
    if sess_summary:
        full_system = f"{system_instruction}\n\nPrevious context summary:\n{sess_summary}"
        out.append({"role": "system", "content": full_system})
    else:
        out.append({"role": "system", "content": system_instruction})
        
    out.extend(sess_messages)
    return out

async def sessions_cleanup_loop(app: FastAPI) -> None:
    while True:
        try:
            now = now_ts()
            dead = [sid for sid, sess in app.state.sessions.items() if now - float(sess.get("updated", now)) > SESSION_IDLE_TTL_SECONDS]
            if dead:
                async with session_io_lock:
                     for sid in dead:
                        app.state.sessions.pop(sid, None)
                     with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
                        json.dump(app.state.sessions, f)
        except Exception:
            pass
        await asyncio.sleep(60.0)


async def cache_store(prompt_probe_text: str, answer: str) -> None:
    if cache_collection is None: return
    def _upsert():
        cid = hashlib.sha256(prompt_probe_text.encode("utf-8")).hexdigest()
        meta = {"answer": answer, "ts": now_ts()}
        if hasattr(cache_collection, "upsert"):
            cache_collection.upsert(documents=[prompt_probe_text], metadatas=[meta], ids=[cid])
        else:
            try: cache_collection.delete(ids=[cid])
            except Exception: pass
            cache_collection.add(documents=[prompt_probe_text], metadatas=[meta], ids=[cid])
    try: await run_in_thread(_upsert)
    except Exception: pass

async def cache_lookup(prompt_probe_text: str) -> Optional[str]:
    if cache_collection is None: return None
    cutoff = now_ts() - CACHE_TTL_SECONDS
    def _q(): return cache_collection.query(query_texts=[prompt_probe_text], n_results=1)
    try:
        results = await run_in_thread(_q)
        if results.get("distances") and results["distances"][0] and results["distances"][0][0] < 0.2:
            meta = (results.get("metadatas") or [[{}]])[0][0] or {}
            ts = float(meta.get("ts", 0) or 0)
            if ts and ts < cutoff:
                try:
                    cid = (results.get("ids") or [[""]])[0][0]
                    if cid: await run_in_thread(lambda: cache_collection.delete(ids=[cid]))
                except Exception: pass
                return None
            return meta.get("answer")
    except Exception: return None
    return None

async def cache_cleanup_loop(app: FastAPI) -> None:
    if cache_collection is None: return
    while True:
        try:
            cutoff = now_ts() - CACHE_TTL_SECONDS
            def _cleanup():
                expired, offset, limit = [], 0, 500
                while True:
                    batch = cache_collection.get(include=["metadatas", "ids"], limit=limit, offset=offset)
                    ids, metas = batch.get("ids") or [], batch.get("metadatas") or []
                    if not ids: break
                    for cid, meta in zip(ids, metas):
                        try: ts = float((meta or {}).get("ts", 0) or 0)
                        except Exception: ts = 0
                        if ts and ts < cutoff: expired.append(cid)
                    offset += len(ids)
                    if len(ids) < limit: break
                if expired: cache_collection.delete(ids=expired)
            await run_in_thread(_cleanup)
        except Exception: pass
        await asyncio.sleep(float(CACHE_CLEANUP_INTERVAL_SECONDS))

async def touch_ollama_model(app: FastAPI, model_key: str) -> None:
    if MODELS.get(model_key, {}).get("type") != "local": return
    keep_alive = KEEPALIVE_SECONDS.get(model_key, 600)
    try:
        await app.state.ollama.post("/generate", json={"model": MODELS[model_key]["name"], "prompt": "ping", "keep_alive": keep_alive, "stream": False})
        app.state.model_last_used[model_key] = now_ts()
    except Exception: pass

async def reformulate_search_query(app: FastAPI, session_id: str, raw_query: str) -> str:
    """Uses the fast Xiaomi router model to rewrite conversational queries into standalone search terms."""
    
    session_data = app.state.sessions.get(session_id, {})
    
    if isinstance(session_data, dict):
        history = session_data.get("messages", [])
    else:
        history = session_data
        
    history_list = list(history)
    
    if not history_list:
        return raw_query
        
    context_msgs = history_list[-4:]
    conversation_text = ""
    for msg in context_msgs:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        role = "User" if (isinstance(msg, dict) and msg.get("role") == "user") else "AI"
        if content:
            conversation_text += f"{role}: {content}\n"

    prompt = f"""You are an advanced search query reformulator.
Look at the conversation history and rewrite the User's Latest Prompt into a standalone web search query.
- Replace ALL pronouns (it, that, they, he) with the specific names of the entities being discussed.
- If the user asks to compare two things, explicitly name BOTH things in your rewritten query.
- DO NOT answer the prompt. Output ONLY the standalone search string.
- If the prompt is just a greeting or does not require a search (e.g., "hi", "thanks", "ok"), output EXACTLY: NO_SEARCH

Conversation History:
{conversation_text}

User's Latest Prompt: {raw_query}

Standalone Search Query:"""

    try:
        url = API_URLS.get("router", API_URLS.get("chat", "http://127.0.0.1:11434/v1"))
        key = API_KEYS.get("router", API_KEYS.get("chat", ""))
        model_name = MODELS.get("router", MODELS.get("chat", {})).get("name", "xiaomi")
        
        endpoint = url if url.endswith("chat/completions") else f"{url.rstrip('/')}/chat/completions"

        async with httpx.AsyncClient() as client:
            res = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}"} if key else {},
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0, 
                    "max_tokens": 50
                },
                timeout=5.0 
            )
            
            if res.status_code == 200:
                rewritten = res.json()["choices"][0]["message"]["content"].strip()
                rewritten = rewritten.strip('"').strip("'")
                
                if rewritten == "NO_SEARCH":
                    return raw_query
                    
                if rewritten.lower() != raw_query.lower():
                    print(f">> [REFORMULATOR] Memory Injected: '{raw_query}' -> '{rewritten}'")
                    return rewritten
                    
    except Exception as e:
        print(f">> [REFORMULATOR] Silent Failure: {e}. Falling back to raw query.")
        
    return raw_query

async def evaluate_prompt_with_router(app: FastAPI, user_text: str) -> Dict[str, Any]:
    """Uses the 4B model to grade the prompt, physically blocking its ability to 'think'."""
    prompt = f"""<system_instructions>
You are an expert AI traffic router. Your job is to score user prompts based on complexity so they can be routed to the correct model size.
You MUST output ONLY valid JSON. No explanations, no markdown formatting outside of the JSON block.
CRITICAL RULE: DO NOT use <think> tags. DO NOT output any reasoning or internal monologue. Start your response IMMEDIATELY with the opening JSON brace {{.
</system_instructions>

<complexity_scale>
0-15: Casual greetings, simple facts, one-sentence questions. (e.g., "Hello", "What is the capital of France?")
16-35: Basic tasks, professional emails, summarizing short text, simple lists. (e.g., "Write an email to my boss about a delay.")
36-60: Standard coding, logic puzzles, data formatting, multi-step instructions. (e.g., "Write a python script to scrape a website.")
61-85: Complex debugging, advanced system design, deep analytical reasoning. (e.g., "Why is my React useEffect causing an infinite loop here?")
86-100: Frontier AI research, highly abstract math, massive context synthesis.
</complexity_scale>

<examples>
User: "Write a professional email explaining why a database migration is delayed."
JSON: {{"complexity": 25, "is_code": false, "is_vision": false, "importance": 50, "needs_search": false}}

User: "Fix the race condition in this multithreaded Golang server."
JSON: {{"complexity": 75, "is_code": true, "is_vision": false, "importance": 80, "needs_search": false}}

User: "What is the weather in Tokyo right now?"
JSON: {{"complexity": 10, "is_code": false, "is_vision": false, "importance": 20, "needs_search": true}}

User: "Do some research on Gartner to help me prepare for an interview."
JSON: {{"complexity": 45, "is_code": false, "is_vision": false, "importance": 60, "needs_search": true}}
</examples>

<user_message>
{user_text}
</user_message>

RETURN JSON:"""
    
    default_eval = {
        "complexity": 30, 
        "is_code": False, 
        "is_vision": False, 
        "importance": 50,
        "needs_search": False
    }

    dynamic_timeout = min(300.0, max(30.0, 30.0 + (len(user_text) / 1000.0)))
    
    try:
        res = await app.state.ollama.post("/generate", json={
            "model": MODELS["router"]["name"],
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": KEEPALIVE_SECONDS.get("router", 3600),
            "options": {
                "temperature": 0.0,       
                "num_predict": 150        
            }
        }, timeout=dynamic_timeout)
        
        result_text = res.json().get("response", "")
        parsed = safe_json_loads(result_text, default_eval)
        
        parsed["complexity"] = int(parsed.get("complexity", 30))
        parsed["importance"] = int(parsed.get("importance", 50))
        parsed["needs_search"] = bool(parsed.get("needs_search", False))
        
        return parsed
        
    except Exception as e:
        print(f">> [ROUTER FAILURE] Error: {type(e).__name__} - {str(e)}. Falling back to default.")
        return default_eval

async def execute_search(app: FastAPI, query: str, engine: str, max_results: int = 15) -> str:
    """Dispatches search to local SearxNG container, with failover to Perplexity Sonar."""
    
    if engine in ["duckduckgo", "ddg"]:
        try:
            import urllib.parse
            
            async def _do_search():
                url = f"http://127.0.0.1:8080/search?q={urllib.parse.quote(query)}&format=json&language=en"
                
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json"
                }
                
                async with httpx.AsyncClient() as client:
                    res = await client.get(url, headers=headers, timeout=15.0)
                    if res.status_code == 200:
                        data = res.json()
                        return data.get("results", [])[:max_results]
                    else:
                        print(f">> [SearxNG] API Error {res.status_code}: {res.text}")
                return []
            
            results = await _do_search()
            
            if results: 
                print(f">> [SearxNG] Fetched {len(results)} aggregated live links. Initiating Deep Extraction...")
                
                async def process_result(r):
                    title = r.get('title', 'Unknown Title')
                    snippet = r.get('content', '') 
                    url = r.get('url', '')
                    
                    if url and _TRAFILATURA_AVAILABLE:
                        scraped_text = await smart_scrape_async(url)
                        if scraped_text:
                            return f"### SOURCE: {title}\nURL: {url}\n{scraped_text}"
                    return f"### SOURCE: {title}\nURL: {url}\n{snippet}"

                extracted_articles = await asyncio.gather(*(process_result(r) for r in results))
                combined_web_text = "\n\n---\n\n".join(extracted_articles)
                
                if len(combined_web_text) > 8000:
                    print(f">> [WEB RAG] Massive payload detected. Engaging In-Memory Sniper...")
                    web_chunks = semantic_chunking(combined_web_text, max_chunk_len=1200)
                    
                    if _CROSS_ENCODER_AVAILABLE and cross_encoder is not None:
                        pairs = [[query, chunk] for chunk in web_chunks]
                        scores = await run_in_thread(lambda: cross_encoder.predict(pairs)) 
                        scored_chunks = list(zip(scores, web_chunks))
                        scored_chunks.sort(key=lambda x: x[0], reverse=True)
                        best_chunks = [chunk for score, chunk in scored_chunks[:6] if score > 0.0]
                        if not best_chunks: best_chunks = [chunk for score, chunk in scored_chunks[:3]]
                        combined_web_text = "\n\n...\n\n".join(best_chunks)
                        
                return combined_web_text
                
            print(">> [SearxNG] Returned empty results. Failing over to Sonar...")
            engine = "sonar"
            
        except Exception as e: 
            print(f">> [SearxNG] Fatal Error: {type(e).__name__} - {str(e)}. Failing over to Sonar...")
            engine = "sonar"
            
    # --- 2. Perplexity Sonar (High Complexity / Premium / Failover) ---
    if engine == "sonar":
        if not API_KEYS.get("search"):
            return "SYSTEM ERROR: Local SearxNG failed, and Perplexity Sonar is not configured to catch the failover."
            
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    API_URLS["search"],
                    headers={"Authorization": f"Bearer {API_KEYS['search']}"},
                    json={
                        "model": MODELS["search"]["name"], 
                        "messages": [{"role": "user", "content": f"Search the web and summarize the latest data regarding: {query}"}]
                    },
                    timeout=30.0
                )
                
                if res.status_code == 200:
                    data = res.json()
                    print(f">> [SONAR PRO] Successfully synthesized live web data.")
                    return data["choices"][0]["message"]["content"]
                else:
                    print(f">> [SONAR PRO] API Rejected Request: {res.status_code} - {res.text}")
                    return f"SYSTEM ERROR: Sonar API failed with status {res.status_code}."
                    
        except Exception as e:
            print(f">> [SONAR PRO] Network Error: {str(e)}")
            return "SYSTEM ERROR: The backup search engine (Sonar) also failed to connect."
            
    return "SYSTEM ERROR: Search yielded no results across all engines."

def select_model_routing(app: FastAPI, eval_data: Dict[str, Any], is_agent: bool) -> str:
    """Applies dynamic routing logic. Router is strictly for background evaluation only."""
    if eval_data.get("forced_model"):
        return eval_data["forced_model"]
        
    comp = eval_data.get("complexity", 30)
    
    # Check RAM/CPU usage from our background loop
    sys_stats = getattr(app.state, "sys", {})
    ram_percent = float(sys_stats.get("ram_percent", 0.0))
    cpu_percent = float(sys_stats.get("cpu", 0.0))
    
    # --- MEMORY BAND GUARD ---
    if ram_percent > 85.0 or cpu_percent > 90.0:
        print(f">> SYSTEM UNDER LOAD (RAM: {ram_percent}%). Punting task to Xiaomi.")
        return "xiaomi"

    # Route OpenClaw Agent Commands
    if is_agent:
        if comp > 10: return "xiaomi"
        else: return "giant" # Qwen 9B handles all local agent logic                 
    
    # Route Standard Chat Commands
    else:
        if comp <= 80: return "xiaomi"
        elif comp <= 95: return "glm"
        else: return "opus"

# ============================================================
# LLM Execution Adapters (Ollama, OpenAI-Compat, Anthropic)
# ============================================================

async def unload_ollama_model(app: FastAPI, model_key: str) -> None:
    """Forces Ollama to drop a model from RAM immediately."""
    if model_key not in MODELS or MODELS[model_key]["type"] != "local": return
    try:
        await app.state.ollama.post("/generate", json={"model": MODELS[model_key]["name"], "prompt": "", "keep_alive": 0})
    except Exception: pass

async def stream_ollama(app: FastAPI, model_key: str, messages: List[Dict[str, Any]]) -> AsyncGenerator[bytes, None]:
    model_name = MODELS[model_key]["name"]
    keep_alive_val = KEEPALIVE_SECONDS.get(model_key, -1)
    
    payload = {
        "model": model_name, 
        "messages": messages,
        "keep_alive": keep_alive_val
    }
    
    try:
        async with app.state.ollama.stream("POST", "/chat", json=payload, timeout=120.0) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield sse_pack({"choices": [{"delta": {"content": content}}]})
                        
                        if data.get("done"):
                            total_tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
                            eval_dur = data.get("eval_duration", 0) / 1e9
                            yield sse_pack({
                                "nexus_stats": {
                                    "tokens": total_tokens,
                                    "time": eval_dur
                                }
                            })
                    except Exception: pass
    except httpx.ConnectError:
        yield sse_pack({"choices": [{"delta": {"content": "\n\n> ⚠️ **Local Engine Offline:** Could not connect to Ollama. Make sure the Ollama app is running!\n"}}]})
    except Exception as e:
        yield sse_pack({"choices": [{"delta": {"content": f"\n\n> ⚠️ **Local Execution Error:** {str(e)}\n"}}]})
        
    yield sse_done()

async def stream_openai_compat(app: FastAPI, model_key: str, messages: List[Dict[str, Any]]) -> AsyncGenerator[bytes, None]:
    api_url = API_URLS.get(model_key)
    api_key = API_KEYS.get(model_key)
    model_name = MODELS[model_key]["name"]
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_name, "messages": messages, "stream": True}
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", api_url, headers=headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    yield sse_pack({"choices": [{"delta": {"content": f"\n\n> ⚠️ **Cloud API Rejected Request ({response.status_code}):** {err_bytes.decode('utf-8', errors='replace')}\n"}}]})
                    return
                
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                yield sse_pack({"choices": [{"delta": {"content": delta}}]})
                        except Exception: 
                            pass
    except httpx.ConnectError:
        # Gracefully handle dead URLs or offline internet
        yield sse_pack({"choices": [{"delta": {"content": "\n\n> ⚠️ **Cloud Connection Error:** Could not reach the API server. Check your internet or URL.\n"}}]})
    except Exception as e:
        # Gracefully handle timeouts or other unexpected crashes
        yield sse_pack({"choices": [{"delta": {"content": f"\n\n> ⚠️ **API Error:** {str(e)}\n"}}]})
        
    yield sse_done()

async def stream_anthropic(app: FastAPI, messages: List[Dict[str, Any]]) -> AsyncGenerator[bytes, None]:
    api_url = API_URLS.get("opus")
    api_key = API_KEYS.get("opus")
    model_name = MODELS["opus"]["name"]
    
    # Adapt OpenAI message format to Anthropic
    system_prompt = ""
    anthropic_msgs = []
    
    for m in messages:
        if m["role"] == "system": 
            system_prompt += m["content"] + "\n"
        else: 
            content = m["content"]
            
            # If this message contains our massive XML context block, we split it!
            if "<context>" in content and "</context>" in content:
                parts = content.split("</context>")
                heavy_context_part = parts[0] + "</context>"
                light_query_part = parts[1]
                
                # We format the content as an array and freeze the heavy part in RAM
                anthropic_msgs.append({
                    "role": m["role"], 
                    "content": [
                        {
                            "type": "text",
                            "text": heavy_context_part,
                            "cache_control": {"type": "ephemeral"} 
                        },
                        {
                            "type": "text",
                            "text": light_query_part
                        }
                    ]
                })
            else:
                anthropic_msgs.append({"role": m["role"], "content": content})
                
    # We leave the system cache tag as well. Anthropic allows up to 4 cache breakpoints per request.
    system_payload = [
        {
            "type": "text",
            "text": system_prompt.strip(),
            "cache_control": {"type": "ephemeral"}
        }
    ] if system_prompt else []
        
    headers = {
        "x-api-key": api_key, 
        "anthropic-version": "2023-06-01", 
        "Content-Type": "application/json",
        "anthropic-beta": "prompt-caching-2024-07-31"
    }
    
    payload = {
        "model": model_name, 
        "messages": anthropic_msgs, 
        "system": system_payload, 
        "stream": True, 
        "max_tokens": 4096
    }
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", api_url, headers=headers, json=payload, timeout=60.0) as response:
                
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    yield sse_pack({"choices": [{"delta": {"content": f"\n\n> ⚠️ **Anthropic API Rejected Request ({response.status_code}):** {err_bytes.decode('utf-8', errors='replace')}\n"}}]})
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {}).get("text", "")
                                if delta: yield sse_pack({"choices": [{"delta": {"content": delta}}]})
                        except Exception: pass
    except httpx.ConnectError:
        yield sse_pack({"choices": [{"delta": {"content": "\n\n> ⚠️ **Cloud Connection Error:** Could not reach the Anthropic API. Check your internet connection.\n"}}]})
    except Exception as e:
        yield sse_pack({"choices": [{"delta": {"content": f"\n\n> ⚠️ **Anthropic API Error:** {str(e)}\n"}}]})
        
    yield sse_done()

async def stream_system2_xiaomi(app: FastAPI, messages: List[Dict[str, Any]], user_query: str) -> AsyncGenerator[bytes, None]:
    """Frontier-level Best-of-N Test-Time Compute Pipeline strictly using Xiaomi."""
    
    api_url = API_URLS.get("xiaomi")
    api_key = API_KEYS.get("xiaomi")
    model_name = MODELS["xiaomi"]["name"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 1. Open the UI's reasoning dropdown
    yield sse_pack({"choices": [{"delta": {"content": "<think>\n🚀 **System 2 Engaged:** Spawning 3 parallel cognitive threads...\n"}}]})

    # 2. PARALLEL GENERATION (Best-of-N)
    personas = [
        "Focus on absolute logical efficiency, directness, and brevity.",
        "Focus on edge cases, security vulnerabilities, and robust error handling.",
        "Play devil's advocate: find the hidden flaws in standard approaches and solve them."
    ]

    async def generate_candidate(persona: str, idx: int) -> str:
        cand_msgs = messages.copy()
        cand_msgs.append({"role": "system", "content": f"SYSTEM OVERRIDE: {persona}"})
        payload = {"model": model_name, "messages": cand_msgs, "temperature": 0.7}
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(api_url, headers=headers, json=payload, timeout=45.0)
                return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Thread {idx} Failed: {e}"

    # Fire all 3 Xiaomi calls at the exact same time
    candidates = await asyncio.gather(*(generate_candidate(p, i) for i, p in enumerate(personas)))
    
    yield sse_pack({"choices": [{"delta": {"content": "✅ Threads complete. Booting Reward Model to critique logic & hallucinations...\n"}}]})

    # 3. LOCAL REWARD MODELING (Scoring)
    scoring_prompt = f"""You are an elite AI Reward Model. Evaluate these 3 candidate responses to the user's query.
USER QUERY: {user_query}

[CANDIDATE 1 - Efficiency]: {candidates[0]}
[CANDIDATE 2 - Security]: {candidates[1]}
[CANDIDATE 3 - Devil's Advocate]: {candidates[2]}

Critique each candidate based on logic, factual accuracy, and alignment with constraints. Identify the BEST candidate.
You MUST output ONLY valid JSON in this exact format:
{{"best_index": 0, "critique": "short explanation of why this won"}}
"""
    payload_rm = {
        "model": model_name,
        "messages": [{"role": "user", "content": scoring_prompt}],
        "temperature": 0.0 # Zero temp for strict deterministic scoring
    }

    best_idx = 0
    critique = "Default selection."
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(api_url, headers=headers, json=payload_rm, timeout=30.0)
            rm_text = res.json()["choices"][0]["message"]["content"]
            
            # Use your existing safe JSON parser
            rm_data = safe_json_loads(rm_text, {})
            best_idx = int(rm_data.get("best_index", 0))
            critique = rm_data.get("critique", "Evaluation complete.")
    except Exception as e:
        yield sse_pack({"choices": [{"delta": {"content": f"⚠️ RM Error: {e}. Defaulting to Thread 1.\n"}}]})

    # Fallback bounds check
    if not (0 <= best_idx <= 2): best_idx = 0
    best_candidate = candidates[best_idx]

    yield sse_pack({"choices": [{"delta": {"content": f"🏆 **RM Decision (Thread {best_idx + 1} Wins):** {critique}\nSynthesizing final master response...\n</think>\n\n"}}]})

    # 4. FINAL SYNTHESIS STREAM
    synth_prompt = f"""You are the final synthesis node.
USER QUERY: {user_query}
OPTIMAL REASONING PATH (Thread {best_idx + 1}): {best_candidate}
REWARD MODEL CRITIQUE: {critique}

Refine, polish, and output the absolute best final response using the Optimal Reasoning Path. 
Do not apologize, do not use <think> tags, and do not reference the threads. Just deliver the final answer directly."""

    synth_msgs = [{"role": "system", "content": "You are Black LLAB's final synthesis engine."}, {"role": "user", "content": synth_prompt}]
    payload_synth = {"model": model_name, "messages": synth_msgs, "stream": True, "temperature": 0.2}

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", api_url, headers=headers, json=payload_synth, timeout=60.0) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                yield sse_pack({"choices": [{"delta": {"content": delta}}]})
                        except Exception: pass
    except Exception as e:
        yield sse_pack({"choices": [{"delta": {"content": f"\n\n> ⚠️ **Synthesis Error:** {str(e)}\n"}}]})
        
    yield sse_done()

async def stream_real_openclaw(app: FastAPI, prompt: str, search_successful: bool = False, model_key: str = "giant", complexity: int = 100) -> AsyncGenerator[bytes, None]:
    """Bridges Nexus to the isolated Docker OpenClaw sandbox and streams its terminal."""
    
    clean_prompt = re.sub(r'^openclaw:?\s*', '', prompt, flags=re.IGNORECASE).strip()
    safe_prompt = clean_prompt.replace("'", "'\\''")

    xiaomi_key = API_KEYS.get("xiaomi", "")
    opus_key = API_KEYS.get("opus", "")
    glm_key = API_KEYS.get("glm", "")

    # Spoof the API provider block dynamically based on user dropdown selection!
    if model_key == "xiaomi" and xiaomi_key:
        provider_config = f"c.models.providers.xiaomi={{api:'openai-completions', baseUrl:'https://api.xiaomimimo.com/v1', apiKey:'{xiaomi_key}', models:[{{id:'mimo-v2-flash', name:'mimo-v2-flash'}}]}}; c.agents.defaults.model.primary='xiaomi/mimo-v2-flash';"
    elif model_key == "opus" and opus_key:
        provider_config = f"c.models.providers.anthropic={{api:'anthropic-messages', baseUrl:'https://api.anthropic.com/v1', apiKey:'{opus_key}', models:[{{id:'claude-opus-4-6', name:'claude-opus-4-6'}}]}}; c.agents.defaults.model.primary='anthropic/claude-opus-4-6';"
    elif model_key == "glm" and glm_key:
        provider_config = f"c.models.providers.glm={{api:'openai-completions', baseUrl:'https://open.bigmodel.cn/api/paas/v4', apiKey:'{glm_key}', models:[{{id:'glm-5', name:'glm-5'}}]}}; c.agents.defaults.model.primary='glm/glm-5';"
    else:
        provider_config = "c.models.providers.openai={api:'openai-responses', baseUrl:'http://host.docker.internal:11434/v1', apiKey:'ollama', models:[{id:'qwen3.5:9b', name:'qwen3.5:9b'}]}; c.agents.defaults.model.primary='openai/qwen3.5:9b';"

    print(f">> [OPENCLAW] model_key={model_key}")

    minified_node = (
        "const fs=require('fs'); const p='/home/node/.openclaw/openclaw.json'; let c={}; "
        "try{c=JSON.parse(fs.readFileSync(p))}catch(e){} "
        "c.models=c.models||{}; c.models.providers=c.models.providers||{}; "
        "c.agents=c.agents||{}; c.agents.defaults=c.agents.defaults||{}; c.agents.defaults.model=c.agents.defaults.model||{}; "
        f"{provider_config} "
        "c.agents.defaults.timeoutSeconds=600; "
        "fs.writeFileSync(p, JSON.stringify(c));"
    )

    setup_cmd = (
        "mkdir -p /home/node/.openclaw && "
        f"node -e \"{minified_node}\" && "
        "rm -f /home/node/.openclaw/models.json ; "
        "rm -f /home/node/.openclaw/agents/main/agent/models.json ; "
        "echo '--- OPENCLAW CONFIG ---' ; "
        "node -e \"const fs=require('fs'); const c=JSON.parse(fs.readFileSync('/home/node/.openclaw/openclaw.json')); if(c.models?.providers?.xiaomi?.apiKey) c.models.providers.xiaomi.apiKey='[REDACTED]'; if(c.models?.providers?.openai?.apiKey) c.models.providers.openai.apiKey='[REDACTED]'; console.log(JSON.stringify(c));\" ; "
        "echo '--- MODELS CACHE 1 ---' ; "
        "cat /home/node/.openclaw/models.json 2>/dev/null || true ; "
        "echo '--- MODELS CACHE 2 ---' ; "
        "cat /home/node/.openclaw/agents/main/agent/models.json 2>/dev/null || true ; "
        "echo '--- END DEBUG ---' ; "
        "cd /workspace && "
        "rm -rf /home/node/.openclaw/workspace ; "
        "ln -sfn /workspace /home/node/.openclaw/workspace ; "
        "find /home/node/.openclaw -name '*.lock' -delete ; "
        f"openclaw agent --agent main -m '{safe_prompt}'"
    )

    docker_bin = os.getenv("DOCKER_BIN") or shutil.which("docker")
    if not docker_bin:
        possible_paths = [
            "/usr/local/bin/docker", "/opt/homebrew/bin/docker",
            os.path.expanduser("~/.docker/bin/docker"), "/Applications/Docker.app/Contents/Resources/bin/docker"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                docker_bin = p
                break
                
    if not docker_bin:
        yield sse_pack({"choices": [{"delta": {"content": "\n\n> **[Docker Error]**\n> Docker CLI not found."}}]})
        yield sse_done()
        return

    docker_args = [docker_bin, "exec", "openclaw-sandbox", "sh", "-lc", setup_cmd]

    engine_str = "duckduckgo" if search_successful else "none"
    model_display = MODELS.get(model_key, {}).get("display", "OpenClaw Sandbox")
    
    yield sse_pack({
        "nexus_meta": { "model_name": model_display, "complexity": complexity, "search_engine": engine_str },
        "choices": [{"delta": {"content": "\n\n> **[Clawdbot Execution Terminal]**\n> "}}]
    })
    
    yield sse_pack({"choices": [{"delta": {"content": "🚀 **Task Accepted:** Booting agent workspace...\n> "}}]})

    if search_successful:
        yield sse_pack({"choices": [{"delta": {"content": "✅ **Web Context Injected:** Live search results successfully loaded into Agent Brain.\n> "}}]})
    else:
        yield sse_pack({"choices": [{"delta": {"content": "ℹ️ **Web Search:** Skipped (No active web context injected).\n> "}}]})

    try:
        process = await asyncio.create_subprocess_exec(
            *docker_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
    
        agent_transcript = ""
        skip_debug_output = False 

        while True:
            line = await process.stdout.readline()
            if not line:
                break

            decoded_line = line.decode("utf-8")
            agent_transcript += decoded_line

            if "--- OPENCLAW CONFIG ---" in decoded_line:
                skip_debug_output = True
                continue
            if "--- END DEBUG ---" in decoded_line:
                skip_debug_output = False
                continue
            if skip_debug_output:
                continue
        
            if "--- MODELS CACHE" in decoded_line:
                continue

            if "Gateway agent failed; falling back to embedded" in decoded_line:
                continue

            if any(tag in decoded_line for tag in ["[diagnostic]", "[tools]", "[agent]"]):
                reasoning_line = f"<think>{decoded_line.strip()}</think>\n"
                yield sse_pack({"choices": [{"delta": {"content": reasoning_line}}]})
                continue

            text_line = decoded_line.replace('\n', '\n> ')
            yield sse_pack({"choices": [{"delta": {"content": text_line}}]})

        await process.wait()
        print(f">> [OPENCLAW] exit_code={process.returncode}")
        
        try:
            workspace_dir = os.path.abspath("./workspace")
            if os.path.exists(workspace_dir):
                files = [os.path.join(workspace_dir, f) for f in os.listdir(workspace_dir) 
                         if os.path.isfile(os.path.join(workspace_dir, f)) and not f.startswith('.')]
                
                if files:
                    newest_file = max(files, key=os.path.getmtime)
                    filename = os.path.basename(newest_file)
                    
                    valid_exts = ['.txt', '.md', '.py', '.json', '.html', '.csv', '.js']
                    if (time.time() - os.path.getmtime(newest_file) < 60) and any(filename.endswith(e) for e in valid_exts):
                        with open(newest_file, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read(1500)
                            
                        ellipsis = "\n...[Content Truncated]" if len(content) == 1500 else ""
                        safe_content = content.replace('\n', '\n> ') 
                        
                        preview = (
                            f"\n> \n> 📄 **Auto-Import [{filename}]:**\n"
                            f"> \n> {safe_content}{ellipsis}\n"
                        )
                        yield sse_pack({"choices": [{"delta": {"content": preview}}]})
        except Exception as e:
            pass 

        yield sse_pack({"choices": [{"delta": {"content": "\n> \n> ⏳ **Synthesizing Executive Summary...**\n"}}]})
        
        try:
            summary_prompt = (
                "You are an AI assistant. Summarize the following agent execution transcript in 3-4 concise bullet points. "
                "Highlight exactly what the agent did, what files it created/modified, and the final outcome/findings. "
                "Output ONLY the bullet points, no extra chat.\n\nTranscript:\n" + agent_transcript[-5000:]
            )
            
            res = await app.state.ollama.post(
                "/generate", 
                json={
                    "model": MODELS["router"]["name"], 
                    "prompt": summary_prompt, 
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 300}
                },
                timeout=45.0
            )
            
            summary_text = res.json().get("response", "").strip()
            
            if summary_text:
                final_report = (
                    f"\n\n---\n### 📊 Executive Summary of Agent Findings\n"
                    f"{summary_text}\n\n"
                    f"✅ **Task Complete.** The agent has closed the workspace.\n"
                )
                yield sse_pack({"choices": [{"delta": {"content": final_report}}]})
            else:
                yield sse_pack({"choices": [{"delta": {"content": "\n> \n> ✅ **Task Complete.** The agent has closed the workspace.\n"}}]})
                
        except Exception as e:
            yield sse_pack({"choices": [{"delta": {"content": "\n> \n> ✅ **Task Complete.** The agent has closed the workspace.\n"}}]})

        yield sse_pack({"nexus_stats": { "tokens": 0, "time": 0 }})
        
    except Exception as e:
        yield sse_pack({"choices": [{"delta": {"content": f"\n\n> **[Docker Error]**\n> {str(e)}"}}]})
         
    yield sse_done()

async def delayed_restart():
    await asyncio.sleep(0.8)
    os.execv(sys.executable, [sys.executable] + sys.argv)

async def background_task_manager():
    """Watches a single tasks.txt file and executes tasks sequentially in the background."""
    bg_dir = os.path.abspath("./workspace_bg")
    tasks_file = os.path.join(bg_dir, "tasks.txt")
    output_dir = os.path.join(bg_dir, "completed_output")
    
    os.makedirs(bg_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    docker_bin = shutil.which("docker")
    if not docker_bin:
        possible_paths = [
            "/usr/local/bin/docker",
            "/opt/homebrew/bin/docker",
            os.path.expanduser("~/.docker/bin/docker"),
            "/Applications/Docker.app/Contents/Resources/bin/docker"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                docker_bin = p
                break
                
    if not docker_bin:
        print(">> [BG AGENT] ⚠️ Docker CLI not found. Background tasks disabled.")
        return 
    
    # Create the template file if it doesn't exist
    if not os.path.exists(tasks_file):
        with open(tasks_file, "w", encoding="utf-8") as f:
            f.write("# BACKGROUND AGENT TO-DO LIST\n")
            f.write("# Add tasks below using the format: [ ] Your task here\n")
            f.write("# The agent will output files to the 'completed_output' folder.\n\n")
            f.write("[ ] Write a python script to calculate the first 100 Fibonacci numbers and save it as fib.py\n")

    # Give the server 10 seconds to fully boot before checking
    await asyncio.sleep(10)
    
    while True:
        try:
            if os.path.exists(tasks_file):
                with open(tasks_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                updated = False
                task_to_run = None
                
                # Scan for the first incomplete task
                for i, line in enumerate(lines):
                    if line.strip().startswith("[ ]"):
                        task_to_run = line.strip()[3:].strip() # Extract the text after [ ]
                        lines[i] = line.replace("[ ]", "[RUNNING]")
                        updated = True
                        break
                
                if updated and task_to_run:
                    # 1. Update the text file to show it is currently working
                    with open(tasks_file, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                        
                    print(f"\n>> [BG AGENT] 🚀 Starting Task: {task_to_run[:50]}...")
                    
                    # 2. Execute the task in the background container
                    safe_prompt = task_to_run.replace("'", "'\\''")
                    
                    cmd = (
                        "cd /workspace/completed_output && "
                        "rm -rf /home/node/.openclaw/workspace && "
                        "ln -sfn /workspace/completed_output /home/node/.openclaw/workspace && "
                        "find /home/node/.openclaw -name '*.lock' -delete && "
                        f"openclaw agent --agent main -m '{safe_prompt}'"
                    )
                    
                    proc = await asyncio.create_subprocess_exec(
                        docker_bin, "exec", "openclaw-bg-sandbox", "sh", "-lc", cmd
                    )
                    await proc.wait() # This will wait as long as the agent takes (up to 24hrs)
                
                    with open(tasks_file, "r", encoding="utf-8") as f:
                        fresh_lines = f.readlines()
                        
                    for j, f_line in enumerate(fresh_lines):
                        if "[RUNNING]" in f_line and task_to_run in f_line:
                            fresh_lines[j] = f_line.replace("[RUNNING]", "[x] DONE:")
                            break
                            
                    with open(tasks_file, "w", encoding="utf-8") as f:
                        f.writelines(fresh_lines)
                        
                    print(f">> [BG AGENT] ✅ Task Complete!")
                    
        except Exception as e:
            print(f">> [BG AGENT] Error: {e}")
            
        # Check the document for new tasks every 15 seconds
        await asyncio.sleep(15)
async def setup_docker_sandbox():
    """Boots TWO Docker containers and locks the background one to the Local Giant."""
    print(">> [BOOTSTRAP] Refreshing OpenClaw Sandbox Containers...")
    
    cwd = os.getcwd()
    
    # Foreground Agent Folders (For UI Chat)
    dot_openclaw = os.path.join(cwd, ".openclaw")
    workspace = os.path.join(cwd, "workspace")
    
    # Background Agent Folders (For tasks.txt loop)
    dot_openclaw_bg = os.path.join(cwd, ".openclaw_bg")
    workspace_bg = os.path.join(cwd, "workspace_bg")
    
    # Ensure all local sync folders exist
    for folder in [dot_openclaw, workspace, dot_openclaw_bg, workspace_bg]:
        os.makedirs(folder, exist_ok=True)
    
    # Hunt down the Docker executable securely
    docker_bin = shutil.which("docker")
    if not docker_bin:
        possible_paths = [
            "/usr/local/bin/docker",
            "/opt/homebrew/bin/docker",
            os.path.expanduser("~/.docker/bin/docker"),
            "/Applications/Docker.app/Contents/Resources/bin/docker"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                docker_bin = p
                break
                
    if not docker_bin:
        print(">> [BOOTSTRAP] ⚠️ Docker CLI not found. Agent features will be disabled.")
        return

    # 1. Forcibly remove the old, dead containers (ADDED SEARXNG TO THIS LIST)
    for name in ["openclaw-sandbox", "openclaw-bg-sandbox", "searxng"]:
        try:
            proc = await asyncio.create_subprocess_exec(
                docker_bin, "rm", "-f", name, 
                stdout=asyncio.subprocess.DEVNULL, 
                stderr=asyncio.subprocess.DEVNULL
            )
            await proc.wait()
        except Exception:
            pass
            
    await asyncio.sleep(1) # Give the Docker daemon a second to breathe

    try:
        proc = await asyncio.create_subprocess_exec(
            docker_bin, "run", "-d", "--name", "searxng", "-p", "8080:8080",
            "-e", "SEARXNG_BASE_URL=http://127.0.0.1:8080",
            "searxng/searxng",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait() 
        
        await asyncio.sleep(3) 
        
        proc_sed = await asyncio.create_subprocess_exec(
            docker_bin, "exec", "searxng", "sed", "-i", "s/- html/- html\\n    - json/g", "/etc/searxng/settings.yml",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc_sed.wait()
        
        proc_restart = await asyncio.create_subprocess_exec(
            docker_bin, "restart", "searxng",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc_restart.wait()
        
        print(">> [BOOTSTRAP] ✅ SearxNG Meta-Search Engine Online (Port 8080, JSON Unlocked).")
    except Exception as e:
        print(f">> [BOOTSTRAP] ⚠️ SearxNG launch error: {e}")


    # 2. Boot the Foreground Container (Dynamic Routing)
    try:
        proc = await asyncio.create_subprocess_exec(
            docker_bin, "run", "-d", "--name", "openclaw-sandbox", "--network", "host", "--entrypoint", "sh",
            "-v", f"{dot_openclaw}:/home/node/.openclaw", "-v", f"{workspace}:/workspace",
            "ghcr.io/openclaw/openclaw", "-c", "sleep infinity",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait() 
    except Exception as e:
        print(f">> [BOOTSTRAP] ⚠️ Foreground launch error: {e}")

    # 3. Boot the Background Container (Locked to Local)
    try:
        proc = await asyncio.create_subprocess_exec(
            docker_bin, "run", "-d", "--name", "openclaw-bg-sandbox", "--network", "host", "--entrypoint", "sh",
            "-v", f"{dot_openclaw_bg}:/home/node/.openclaw", "-v", f"{workspace_bg}:/workspace",
            "ghcr.io/openclaw/openclaw", "-c", "sleep infinity",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait() 
    except Exception as e:
        print(f">> [BOOTSTRAP] ⚠️ Background launch error: {e}")
        
    await asyncio.sleep(1)
        
    # 4. FORCIBLY INJECT THE LOCAL GIANT CONFIG INTO THE BACKGROUND CONTAINER
    minified_bg_node = (
        "const fs=require('fs'); const p='/home/node/.openclaw/openclaw.json'; let c={}; "
        "c.models={providers:{openai:{api:'openai-responses', baseUrl:'http://host.docker.internal:11434/v1', apiKey:'ollama', models:[{id:'qwen3.5:9b', name:'qwen3.5:9b'}]}}}; "
        "c.agents={defaults:{model:{primary:'openai/qwen3.5:9b'}, timeoutSeconds: 86400}}; "
        "fs.writeFileSync(p, JSON.stringify(c));"
    )
    
    bg_setup_cmd = f"mkdir -p /home/node/.openclaw && node -e \"{minified_bg_node}\""
    
    try:
        proc = await asyncio.create_subprocess_exec(
            docker_bin, "exec", "openclaw-bg-sandbox", "sh", "-lc", bg_setup_cmd,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait() 
        print(">> [BOOTSTRAP] ✅ Dual OpenClaw Agents Online (Foreground & Locked BG).")
    except Exception as e:
        print(f">> [BOOTSTRAP] ⚠️ Config injection error: {e}")

   # 5. INJECT THE SECURED WEB SURFER TOOL AND VISION TOOL INTO BOTH WORKSPACES
    # We dynamically inject the server's global NEXUS_TOKEN so the agent can authenticate
    surfer_code = f"""import sys, urllib.request, urllib.parse, json
if len(sys.argv) < 2:
    print("Usage: python web_surfer.py <URL>")
    sys.exit(1)
url = sys.argv[1]
try:
    api_url = f"http://127.0.0.1:8000/v1/agent/read_url?url={{urllib.parse.quote(url)}}"
    # Attach the Nexus Token to the request headers
    req = urllib.request.Request(api_url, headers={{"X-Nexus-Token": "{NEXUS_TOKEN}"}})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        print(data.get('text', 'No content extracted.'))
except Exception as e:
    print(f"Failed to fetch: {{e}}")
"""

    vision_code = f"""import sys, urllib.request, urllib.parse, json
if len(sys.argv) < 2:
    print("Usage: python vision_tool.py <image_filename>")
    sys.exit(1)
filename = sys.argv[1]
try:
    api_url = f"http://127.0.0.1:8000/v1/agent/vision?filename={{urllib.parse.quote(filename)}}"
    req = urllib.request.Request(api_url, headers={{"X-Nexus-Token": "{NEXUS_TOKEN}"}})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        print(data.get('text', 'No content extracted.'))
except Exception as e:
    print(f"Failed to analyze image: {{e}}")
"""
    
    try:
        # Drop tools into the Foreground Workspace
        with open(os.path.join(workspace, "web_surfer.py"), "w", encoding="utf-8") as f:
            f.write(surfer_code)
        with open(os.path.join(workspace, "vision_tool.py"), "w", encoding="utf-8") as f:
            f.write(vision_code)
            
        # Drop tools into the Background Workspace
        with open(os.path.join(workspace_bg, "web_surfer.py"), "w", encoding="utf-8") as f:
            f.write(surfer_code)
        with open(os.path.join(workspace_bg, "vision_tool.py"), "w", encoding="utf-8") as f:
            f.write(vision_code)
            
        print(">> [BOOTSTRAP] ✅ Secured Web Surfer and Vision tools injected into Agent workspaces.")
    except Exception as e:
        print(f">> [BOOTSTRAP] ⚠️ Failed to inject tool scripts: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nexus_token = NEXUS_TOKEN
    app.state.rate = {}
    app.state.model_last_used = {}
    app.state.spec_cache = {} 
    app.state.ollama = httpx.AsyncClient(base_url=OLLAMA_API_BASE, timeout=120.0)
    
    await load_sessions(app)
    await get_budget(app)
    
    # 1. Setup basic Docker sandboxes
    await setup_docker_sandbox()

    print(">> [BOOTSTRAP] Synchronizing Telegram Config...")
    
    # This script checks if 'channels' exists and injects your token if it doesn't
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    fix_script = (
        "const fs=require('fs'); const f='/home/node/.openclaw/openclaw.json'; "
        "let c=JSON.parse(fs.readFileSync(f,'utf8')); "
        "c.gateway=c.gateway||{}; c.gateway.mode='local'; "
        "c.channels=c.channels||{}; "
        f"c.channels.telegram={{enabled:true, botToken:'{telegram_token}', dmPolicy:'pairing'}}; "
        "fs.writeFileSync(f,JSON.stringify(c,null,2));"
    )
    
    fix_cmd = ["docker", "exec", "-u", "node", "openclaw-bg-sandbox", "node", "-e", fix_script]
    
    try:
        subprocess.run(fix_cmd, check=True, capture_output=True)
        subprocess.run(["docker", "exec", "-u", "node", "openclaw-bg-sandbox", "openclaw", "gateway", "stop"], capture_output=True)
        subprocess.run(["docker", "exec", "-d", "-u", "node", "openclaw-bg-sandbox", "openclaw", "gateway"], check=True)
        print(">> [BOOTSTRAP] ✅ Telegram Background Agent (@Isaac_Dear_Bot) is ONLINE.")
    except Exception as e:
        print(f">> [WARNING] Telegram Auto-Start failed: {e}")

    # 4. Standard background loops
    asyncio.create_task(system_stats_loop(app))
    asyncio.create_task(sessions_cleanup_loop(app))
    asyncio.create_task(cache_cleanup_loop(app))
    asyncio.create_task(background_task_manager())
    
    yield
    await app.state.ollama.aclose()

app = FastAPI(lifespan=lifespan)
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://your-deployment-domain.com", 
]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=ALLOWED_ORIGINS, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

@app.get("/v1/budget")
async def api_budget(request: Request):
    if not _rate_allow(request.app, "budget", _client_ip(request), BUDGET_RPM):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return await get_budget(request.app)

@app.get("/v1/session/history")
async def api_history(session_id: str, request: Request):
    _require_token(request) 
    sess = get_or_create_session(request.app, session_id)
    return {"messages": sess.get("messages", [])}

@app.post("/v1/session/clear")
async def api_clear(request: Request):
    data = await request.json()
    sid = data.get("session_id")
    if sid in request.app.state.sessions:
        del request.app.state.sessions[sid]
        await save_sessions(request.app)
    return {"status": "ok"}

@app.post("/v1/admin/restart")
async def api_restart(request: Request):
    _require_token(request)
    asyncio.create_task(delayed_restart())
    return {"status": "restarting"}

@app.get("/v1/sessions")
async def api_get_all_sessions(request: Request):
    """Returns a list of all chat sessions for the UI sidebar."""
    _require_token(request)
    
    sessions_list = []
    for sid, data in request.app.state.sessions.items():
        summary = data.get("summary", "New Conversation")
        if not summary and data.get("messages"):
             first_msg = next((m["content"] for m in data["messages"] if m["role"] == "user"), "New Conversation")
             summary = first_msg[:40] + "..." if len(first_msg) > 40 else first_msg
             
        sessions_list.append({
            "id": sid,
            "summary": summary,
            "updated": data.get("updated", 0)
        })
        
    sessions_list.sort(key=lambda x: x["updated"], reverse=True)
    return {"sessions": sessions_list}
from fastapi import UploadFile, File

@app.post("/v1/document/parse")
async def api_parse_document(request: Request, file: UploadFile = File(...)):
    _require_token(request)
    
    ext = file.filename.split('.')[-1].lower()
    content = await file.read()
    
    workspace_dir = os.path.abspath("./workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    physical_path = os.path.join(workspace_dir, file.filename)
    
    with open(physical_path, "wb") as f:
        f.write(content)
    
    
    try:
        if ext in ['csv', 'json']:
            file_stats = f"File: {file.filename}\nType: Structured {ext.upper()} Data\nSize: {len(content)} bytes\n"
            
            if ext == 'csv':
                import pandas as pd
                from io import BytesIO
                try:
                    df = pd.read_csv(BytesIO(content), on_bad_lines='skip')
                    file_stats += f"Rows: {len(df)}\nColumns: {list(df.columns)}\nData Sample:\n{df.head(3).to_markdown()}"
                except Exception: pass
            elif ext == 'json':
                file_stats += "Use your Python tools to load and query this JSON structure."

            return {
                "filename": file.filename,
                "content": f"[SYSTEM: {ext.upper()} Bypass Engaged. File saved to /workspace/{file.filename} for Agent access.]\n\n{file_stats}"
            }

        # 2. STANDARD TEXT/PDF EXTRACTION
        full_text = ""
        success_msg = ""
        
        if ext in ['txt', 'md', 'py', 'html', 'xml']:
            full_text = content.decode('utf-8', errors='replace')
            success_msg = f"[SYSTEM: '{file.filename}' ingested via Text Reader.]\n"
            
        elif ext == 'pdf':
            print(">> [PDF] Initiating Advanced Tabular Hybrid Extraction...")
            import pdfplumber
            from io import BytesIO
            
            extracted_pages = []
            
            with pdfplumber.open(BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    
                    # 1. Extract raw text and apply The Paragraph Healer
                    raw_text = page.extract_text() or ""
                    import re
                    # Fix words split across lines by hyphens (e.g. "manu-\nfacturing" -> "manufacturing")
                    healed_text = re.sub(r'([a-zA-Z]+)-\n([a-zA-Z]+)', r'\1\2', raw_text)
                    # Unwrap hard line breaks (single newlines) into spaces so sentences stay intact
                    page_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', healed_text)
                    
                    # 2. Explicitly detect tables and convert them to strict Markdown
                    tables = page.extract_tables()
                    md_tables = []
                    
                    for table in tables:
                        if not table: continue
                        # Clean out None values and replace inner newlines with spaces
                        clean_table = [[str(cell).replace('\n', ' ').strip() if cell else "" for cell in row] for row in table]
                        
                        if len(clean_table) > 0:
                            headers = clean_table[0]
                            # Build Markdown Header
                            md_table = "| " + " | ".join(headers) + " |\n"
                            md_table += "|" + "|".join(["---" for _ in headers]) + "|\n"
                            # Build Markdown Rows
                            for row in clean_table[1:]:
                                md_table += "| " + " | ".join(row) + " |\n"
                            md_tables.append(md_table)
                    
                    # 3. Combine layout-preserved text with perfect Markdown tables
                    combined_page_data = page_text
                    if md_tables:
                        combined_page_data += "\n\n" + "\n\n".join(md_tables)
                        
                    # 4. Check if page has data, or if we need to fall back to Vision OCR for scanned images
                    if combined_page_data and len(combined_page_data.strip()) > 50:
                        extracted_pages.append(combined_page_data)
                    else:
                        try:
                            if _P2T_AVAILABLE and p2t is not None:
                                print(f">> [OCR] Page {i+1} is an image. Booting Vision...")
                                page_imgs = convert_from_bytes(content, dpi=120, first_page=i+1, last_page=i+1)
                                
                                outs = p2t.recognize(page_imgs[0])
                                vision_text = "\n".join([out.get('text', '') for out in outs if isinstance(out, dict)])
                                extracted_pages.append(vision_text)
                            else:
                                extracted_pages.append("[Image - Vision OCR Unavailable]")
                        except Exception as e:
                            print(f">> [OCR ERROR] Vision failed on page {i+1}: {e}")
                            
            full_text = "\n\n".join(extracted_pages)
            success_msg = f"[SYSTEM: '{file.filename}' ingested via Advanced Tabular Reader.]\n"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # 3. AGGRESSIVE MINIFICATION
        clean_text = minify_text(full_text)

        global_context = ""
        if len(clean_text) > 100:
            print(f">> [CONTEXTUAL RAG] Generating global DNA for {file.filename}...")
            # Grab the first ~4000 characters (usually title, intro, abstract) to figure out the subject
            intro_text = clean_text[:4000]
            context_prompt = (
                f"You are an expert data cataloger. Read the beginning of this document and write a "
                f"single, dense sentence explaining exactly what this document is about, who it involves, "
                f"and its main subject matter. DO NOT use introductory phrases.\n\n"
                f"DOCUMENT TITLE: {file.filename}\n\nTEXT:\n{intro_text}"
            )
            try:
                ctx_res = await request.app.state.ollama.post("/generate", json={
                    "model": MODELS["router"]["name"],
                    "prompt": context_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 100}
                }, timeout=15.0)
                global_context = ctx_res.json().get("response", "").strip()
                print(f">> [CONTEXTUAL RAG] DNA Generated: {global_context}")
            except Exception as e:
                print(f">> [CONTEXTUAL RAG Error] {e}. Proceeding without global context.")
                global_context = f"Document titled: {file.filename}"
        else:
            global_context = f"Document titled: {file.filename}"

        # 4. SEMANTIC CHUNKING WITH CONTEXT INJECTION
        raw_chunks = semantic_chunking(clean_text, max_chunk_len=800)
        
        # Inject the global DNA into every single chunk!
        chunks = []
        for chunk in raw_chunks:
            contextualized_chunk = f"[Document Context: {global_context}]\n{chunk}"
            chunks.append(contextualized_chunk)
            
        # 5. INJECT CONTEXTUAL LEAF NODES INTO CHROMADB
        if chunks and doc_collection is not None:
            import uuid
            # We change the ID and type to reflect the new contextual DNA
            ids = [f"{file.filename}_ctx_{uuid.uuid4().hex[:8]}" for _ in chunks]
            metadatas = [{"source": file.filename, "type": "contextual_text", "chunk": i} for i in range(len(chunks))]
            doc_collection.add(documents=chunks, metadatas=metadatas, ids=ids)
            
            # 6. FIRE THE BACKGROUND RAPTOR TASK
            # This runs asynchronously so the UI updates instantly while the AI works in the background
            asyncio.create_task(background_raptor_summaries(request.app, chunks, file.filename))
            
        return {
            "filename": file.filename, 
            "content": success_msg + f" [{len(chunks)} semantic chunks embedded. RAPTOR background summarization initiated. You may now ask questions.]"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failure: {str(e)}")

async def generate_training_data_with_xiaomi(user_text: str):
    """Background task: Uses Xiaomi to generate perfect router training data and saves it to a JSONL file."""
    
    # We use a slightly condensed system prompt for the final training data
    system_instructions = "You are an expert AI traffic router. Score user prompts based on complexity (0-100). Output ONLY valid JSON."
    
    # This matches the exact prompt your local router uses
    full_prompt = f"""<system_instructions>
You are an expert AI traffic router. Your job is to score user prompts based on complexity so they can be routed to the correct model size.
You MUST output ONLY valid JSON. No explanations, no markdown formatting outside of the JSON block.
CRITICAL RULE: DO NOT use <think> tags. DO NOT output any reasoning or internal monologue. Start your response IMMEDIATELY with the opening JSON brace {{.
</system_instructions>

<complexity_scale>
0-15: Casual greetings, simple facts, one-sentence questions. (e.g., "Hello", "What is the capital of France?")
16-35: Basic tasks, professional emails, summarizing short text, simple lists. (e.g., "Write an email to my boss about a delay.")
36-60: Standard coding, logic puzzles, data formatting, multi-step instructions. (e.g., "Write a python script to scrape a website.")
61-85: Complex debugging, advanced system design, deep analytical reasoning. (e.g., "Why is my React useEffect causing an infinite loop here?")
86-100: Frontier AI research, highly abstract math, massive context synthesis.
</complexity_scale>

<user_message>
{user_text}
</user_message>

RETURN JSON:"""

    api_url = API_URLS.get("xiaomi")
    api_key = API_KEYS.get("xiaomi")
    model_name = MODELS["xiaomi"]["name"]
    
    if not api_key: 
        return # Skip if Xiaomi isn't configured
        
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name, 
        "messages": [{"role": "user", "content": full_prompt}], 
        "temperature": 0.0 # Strict 0.0 for consistent routing logic
    }
    
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(api_url, headers=headers, json=payload, timeout=30.0)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"]
                
                # Verify it is valid JSON before saving it!
                clean_json = safe_json_loads(content, None)
                
                if clean_json:
                    # Create the OpenAI ChatML format (The industry standard for fine-tuning)
                    training_row = {
                        "messages": [
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": json.dumps(clean_json)}
                        ]
                    }
                    
                    # Append it securely to the JSONL file in your Obsidian folder
                    async with budget_io_lock: # Reusing a lock to prevent file corruption
                        with open("router_training_data.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps(training_row) + "\n")
                            
                    print(">> [DATA COLLECTOR] Successfully saved Xiaomi routing label for future training.")
    except Exception as e:
        print(f">> [DATA COLLECTOR] Silently failed to generate training data: {e}")

@app.post("/v1/chat/speculate")
async def api_speculate(request: Request):
    """Silently predicts the user's prompt and pre-fetches heavy context."""
    _require_token(request)
    data = await request.json()
    session_id = data.get("session_id")
    partial_text = data.get("partial_text", "").strip()
    
    if len(partial_text) < 15: 
        return {"status": "too_short"}

    try:
        prompt = f"The user is currently typing this query: '{partial_text}'. Predict their full, completed sentence or question. Output ONLY the predicted text."
        
        async with httpx.AsyncClient() as client:
            res = await client.post(
                API_URLS.get("xiaomi"),
                headers={"Authorization": f"Bearer {API_KEYS.get('xiaomi')}"},
                json={"model": MODELS["xiaomi"]["name"], "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=8.0
            )
            predicted_query = res.json()["choices"][0]["message"]["content"].strip()

        print(f">> [SPECULATION] User typing: '{partial_text}' -> Predicted: '{predicted_query}'")

        # Pre-fetch the heaviest bottleneck: Live Web Search
        search_ctx = await execute_search(request.app, predicted_query, "duckduckgo")

        request.app.state.spec_cache[session_id] = {
            "predicted_query": predicted_query,
            "search_context": search_ctx,
            "timestamp": time.time()
        }
        
        return {"status": "speculating", "prediction": predicted_query}
    except Exception as e:
        print(f">> [SPECULATION ERROR] Background fetch failed! Reason: {str(e)}") 
        return {"status": "failed", "error": str(e)}

@app.post("/v1/chat/completions")
async def api_chat(request: Request):
    _require_token(request)
    _enforce_payload_limits(request)
    
    if not _rate_allow(request.app, "chat", _client_ip(request), CHAT_RPM):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    budget = await get_budget(request.app)
    if budget.get("spent", 0) >= budget.get("daily_limit", 2.00):
        raise HTTPException(status_code=402, detail="Daily budget limit reached. Please increase your limit or try again tomorrow.")

    data = await request.json()
    msg = _validate_message_shape(data.get("message", {}))
    session_id = data.get("session_id", "default")
    user_text = msg.get("content", "")
    
    raw_query = user_text
    if "User Query:" in user_text:
        raw_query = user_text.split("User Query:")[-1].strip()
    elif len(raw_query) > 1500:
        raw_query = raw_query[-1500:] 
        
    # --- 2. FRONTIER UPGRADE: MEMORY-AWARE QUERY REFORMULATION ---
    # Ask Xiaomi to look at the isolated query and rewrite "it" or "that" using chat history
    query_for_tools = await reformulate_search_query(request.app, session_id, raw_query)

    # --- NEW: ROUTER BYPASS LOGIC ---
    req_chat_model = data.get("chat_model", "auto")
    req_agent_model = data.get("agent_model", "auto")
    req_search_pref = data.get("search_preference", "auto")
    
    is_agent_request = request.headers.get("X-Internal-Agent") == "true" or user_text.strip().lower().startswith("openclaw")
    
    # --- NEW: UNIFIED DROPDOWN ROUTER BYPASS ---
    requested_model = req_agent_model if is_agent_request else req_chat_model
    
    if requested_model != "auto":
        print(f">> [ROUTER BYPASS] Manual override engaged. Routing directly to {requested_model}.")
        eval_data = {"complexity": 99, "needs_search": False, "forced_model": requested_model}
    else:
        eval_data = await evaluate_prompt_with_router(request.app, query_for_tools)
        
    force_search = request.headers.get("X-Force-Search") == "true"
    
    import re
    search_patterns = [
        r'\b(weather|temperature)\b',                               
        r'\b(latest|current|recent|today\'s)\b',                    
        r'\b(search|look up|research|find info on)\b',              
        r'\bwho (won|is winning)\b',                                                                       
    ]
    
    # If the regex matches, we override the router
    if any(re.search(pattern, user_text.lower()) for pattern in search_patterns):
        eval_data["needs_search"] = True
        force_search = True
        print(">> [ROUTER OVERRIDE] Live-data intent detected. Forcing web search.")
    
    if req_search_pref == "disabled":
        eval_data["needs_search"] = False
        force_search = False
    elif req_search_pref in ["ddg", "sonar"]:
        eval_data["needs_search"] = True
        force_search = True

    # Only collect data for normal chats, skip agent commands to keep the dataset clean
    if not is_agent_request:
        asyncio.create_task(generate_training_data_with_xiaomi(user_text))

    engine_used = "none"
    
    selected_model_key = select_model_routing(request.app, eval_data, is_agent_request)
    if eval_data.get("is_vision") and MODELS[selected_model_key]["type"] == "local" and selected_model_key != "giant":
        selected_model_key = "giant"

    # Liquid Context Window
    if selected_model_key == "opus":
        dynamic_top_k = 40
        sieve_bypass = True
    elif selected_model_key in ["glm", "xiaomi", "search"]:
        dynamic_top_k = 20
        sieve_bypass = True 
    else:
        dynamic_top_k = 6
        sieve_bypass = False

    # Define Task 1: Document RAG
    async def fetch_rag() -> str:
        if doc_collection is None or doc_collection.count() == 0: return ""
        try:
            print(f">> [HyDE] Generating hypothetical target vector via Xiaomi...")
            hyde_prompt = (
                f"You are an expert analyst. Write a highly detailed, hypothetical paragraph answering the following query. "
                f"Do not apologize or explain yourself. Just invent a plausible, fact-dense answer using professional jargon.\n\n"
                f"USER QUERY: {query_for_tools}"
            )
            
            search_queries = [query_for_tools]
            try:
                xiaomi_api_url = API_URLS.get("xiaomi")
                xiaomi_api_key = API_KEYS.get("xiaomi")
                
                headers = {"Authorization": f"Bearer {xiaomi_api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": MODELS["xiaomi"]["name"],
                    "messages": [{"role": "user", "content": hyde_prompt}],
                    "temperature": 0.3
                }
                
                async with httpx.AsyncClient() as client:
                    hyde_res = await client.post(xiaomi_api_url, headers=headers, json=payload, timeout=15.0)
                    if hyde_res.status_code == 200:
                        hypothetical_answer = hyde_res.json()["choices"][0]["message"]["content"].strip()
                        if len(hypothetical_answer) > 20:
                            search_queries.append(hypothetical_answer)
                            print(f">> [HyDE] Target vector generated successfully.")
            except Exception as e:
                print(f">> [HyDE Error] {e}. Falling back to standard query.")
                hypothetical_answer = hyde_res.json().get("response", "").strip()
                if len(hypothetical_answer) > 20:
                    search_queries.append(hypothetical_answer)
                    print(f">> [HyDE] Target vector generated successfully.")
            except Exception as e:
                print(f">> [HyDE Error] {e}. Falling back to standard query.")

            def _search_fuse_and_rerank():
                safe_n_results = min(dynamic_top_k * 3, doc_collection.count())
                if safe_n_results == 0: return None
                
                search_results = doc_collection.query(query_texts=search_queries, n_results=safe_n_results)
                if not search_results or not search_results.get('documents'): return None
                    
                rrf_scores = {}
                k = 60 
                for doc_list in search_results['documents']:
                    if not doc_list: continue
                    for rank, doc in enumerate(doc_list):
                        if doc not in rrf_scores: rrf_scores[doc] = 0.0
                        rrf_scores[doc] += 1.0 / (k + rank + 1)
                        
                import re
                query_terms = set(re.findall(r'\w+', query_for_tools.lower()))
                for doc in rrf_scores.keys():
                    doc_lower = doc.lower()
                    match_count = sum(1 for term in query_terms if len(term) > 4 and term in doc_lower)
                    rrf_scores[doc] *= (1.0 + (match_count * 0.15))
                    
                fused_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
                top_fused_chunks = fused_docs[:(dynamic_top_k * 2)]
                if not top_fused_chunks: return None
                
                if _CROSS_ENCODER_AVAILABLE and cross_encoder is not None:
                    pairs = [[query_for_tools, chunk] for chunk in top_fused_chunks]
                    scores = cross_encoder.predict(pairs)
                    scored_chunks = list(zip(scores, top_fused_chunks))
                    scored_chunks.sort(key=lambda x: x[0], reverse=True)
                    
                    top_k_chunks = [chunk for score, chunk in scored_chunks[:dynamic_top_k] if score > 0.01]
                    
                    if not top_k_chunks:
                        print(">> [SNIPER RAG] No relevant context found in vault. Proceeding with base knowledge.")
                        return None
                        
                    top_k_chunks.reverse()
                    litm_chunks = []
                    for i, chunk in enumerate(top_k_chunks):
                        if i % 2 == 0: litm_chunks.append(chunk)
                        else: litm_chunks.insert(0, chunk)
                            
                    print(f">> [SNIPER RAG] Re-ranked & LitM-shaped the optimal top {dynamic_top_k} chunks for {selected_model_key}.")
                    return litm_chunks
                    
                return top_fused_chunks[:dynamic_top_k]

            best_docs = await run_in_thread(_search_fuse_and_rerank)
            
            if best_docs:
                raw_context = "\n\n---\n\n".join(best_docs)
                if sieve_bypass:
                    print(f">> [RAG] Bypassing Local Sieve. Feeding {len(raw_context)} chars directly to {selected_model_key}.")
                    return f"<document type=\"raw_rag\">\n{raw_context}\n</document>"

                sieve_prompt = (
                    f"You are a strict data extraction tool. Read the following reference text and extract ONLY "
                    f"the facts necessary to answer this user query: '{query_for_tools}'. "
                    f"If the text does not contain the answer, output exactly 'NO RELEVANT DATA'.\n\n"
                    f"REFERENCE TEXT:\n{raw_context}"
                )
                print(f">> [CLOUD SIEVE] Compressing {len(raw_context)} chars of raw context using Xiaomi...")
                
                try:
                    xiaomi_api_url = API_URLS.get("xiaomi")
                    xiaomi_api_key = API_KEYS.get("xiaomi")
                    xiaomi_model_name = MODELS["xiaomi"]["name"]
                    
                    if not xiaomi_api_key:
                        raise ValueError("Xiaomi API Key not configured.")
                        
                    headers = {"Authorization": f"Bearer {xiaomi_api_key}", "Content-Type": "application/json"}
                    payload = {
                        "model": xiaomi_model_name,
                        "messages": [{"role": "user", "content": sieve_prompt}],
                        "temperature": 0.0
                    }
                    
                    async with httpx.AsyncClient() as client:
                        sieve_res = await client.post(xiaomi_api_url, headers=headers, json=payload, timeout=15.0)
                        
                        if sieve_res.status_code == 200:
                            distilled_facts = sieve_res.json()["choices"][0]["message"]["content"].strip()
                        else:
                            raise ValueError(f"API Error {sieve_res.status_code}: {sieve_res.text}")
                            
                    if distilled_facts and "NO RELEVANT DATA" not in distilled_facts.upper():
                        print(f">> [CLOUD SIEVE] SUCCESS! Context compressed to {len(distilled_facts)} chars.")
                        return f"<document type=\"distilled_rag\">\n{distilled_facts}\n</document>"
                    else:
                        print(">> [CLOUD SIEVE] Sieve rejected context. Falling back to raw chunks to be safe.")
                        return f"<document type=\"raw_rag\">\n{raw_context}\n</document>"
                        
                except Exception as e:
                    # If Xiaomi fails or times out, safely fallback to the raw context without crashing the chat
                    print(f">> [CLOUD SIEVE ERROR] {e}. Falling back to raw chunks.")
                    return f"<document type=\"raw_rag\">\n{raw_context}\n</document>"
        except Exception as e:
            print(f">> [LOCAL RAG ERROR] {e}.")
            return f"<document type=\"raw_rag\">\n{raw_context}\n</document>" if 'raw_context' in locals() else ""
        return ""

    # Define Task 2: Long-Term Memory
    async def fetch_memory() -> str:
        if memory_collection is not None and len(user_text) > 10:
            def _search_memory(): return memory_collection.query(query_texts=[user_text], n_results=3)
            mem_results = await run_in_thread(_search_memory)
            if mem_results and mem_results.get("documents") and mem_results["documents"][0]:
                retrieved_docs = "\n---\n".join(mem_results["documents"][0])
                return f"[System Alert: Retrieved from Long-Term Memory]\n{retrieved_docs}"
        return ""

    async def fetch_search() -> str:
        nonlocal engine_used 
        if eval_data.get("needs_search", False) or force_search:
            
            # --- THE ZERO-LATENCY QUALITY CHECK ---
            spec_data = request.app.state.spec_cache.get(session_id)
            if spec_data and (time.time() - spec_data["timestamp"] < 30):
                pred_lower = spec_data["predicted_query"].lower()
                actual_lower = query_for_tools.lower()
                
                if actual_lower in pred_lower or pred_lower[:20] in actual_lower:
                    print(">> [SPECULATION HIT] Prediction was accurate! Injecting zero-latency pre-fetched data.")
                    engine_used = "duckduckgo (pre-fetched)"
                    del request.app.state.spec_cache[session_id] 
                    return f"[System Alert: Live Web Context]\n{spec_data['search_context']}"

            # --- FALLBACK: Standard real-time search ---
            print(">> [SPECULATION MISS] Executing standard search...")
            if req_search_pref in ["ddg", "sonar"]:
                engine = req_search_pref
            else:
                engine = "duckduckgo" if is_agent_request or eval_data.get("complexity", 30) < 80 else "sonar"
                
            search_ctx = await execute_search(request.app, query_for_tools, engine)
            if search_ctx and "yielded no immediate results" not in search_ctx:
                 engine_used = engine
                 return f"[System Alert: Live Web Context]\n{search_ctx}"
        return ""

    # Define Task 4: Local GraphRAG Retrieval
    async def fetch_graph() -> str:
        if obsidian_graph.number_of_nodes() == 0: return ""
        
        # 1. Use Xiaomi to extract the core entities the user is asking about
        entity_prompt = f"Extract the core subject entities from this query. Output ONLY a JSON list of strings. Query: {user_text}"
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    API_URLS.get("xiaomi"),
                    headers={"Authorization": f"Bearer {API_KEYS.get('xiaomi')}"},
                    json={"model": MODELS["xiaomi"]["name"], "messages": [{"role": "user", "content": entity_prompt}], "temperature": 0.0},
                    timeout=5.0
                )
                query_entities = safe_json_loads(res.json()["choices"][0]["message"]["content"], [])
        except Exception:
            return ""

        if not isinstance(query_entities, list) or not query_entities: return ""

        # 2. Traverse the local NetworkX graph to find relationships
        graph_context = []
        for q_ent in query_entities:
            q_ent_upper = str(q_ent).strip().upper()
            
            # Fuzzy matching: check if the queried entity exists in our graph nodes
            matched_nodes = [n for n in obsidian_graph.nodes() if q_ent_upper in n or n in q_ent_upper]
            
            for node in matched_nodes:
                # Get direct neighbors (1-hop traversal)
                neighbors = list(obsidian_graph.neighbors(node))
                for neighbor in neighbors[:5]:
                    edge_data = obsidian_graph.get_edge_data(node, neighbor)
                    rel = edge_data.get('relation', 'is related to')
                    src_file = edge_data.get('source_file', 'unknown')
                    graph_context.append(f"[{src_file}] {node} --({rel})--> {neighbor}")

        if graph_context:
            formatted_graph = "\n".join(list(set(graph_context)))
            print(f">> [GraphRAG] Retrieved {len(graph_context)} relational connections.")
            return f"<document type=\"knowledge_graph_relations\">\n{formatted_graph}\n</document>"
        
        return ""

    if is_agent_request:
        clean_text = re.sub(r'^openclaw:?\s*', '', user_text, flags=re.IGNORECASE).strip()
        search_query = re.sub(r'^openclaw:?\s*', '', query_for_tools, flags=re.IGNORECASE).strip()
        search_successful = False 
        
        print(">> [AGENT PRE-FLIGHT] Gathering external intelligence for OpenClaw...")
        
        # Fire our hyper-optimized RAG and Search tasks concurrently for the agent!
        rag_context, search_context = await asyncio.gather(
            fetch_rag(), 
            fetch_search()
        )
        
        agent_intelligence_brief = ""
        
        if search_context:
            search_successful = True
            agent_intelligence_brief += f"\n[LIVE WEB SEARCH DATA]\n{search_context}\n"
            
        if rag_context:
            agent_intelligence_brief += f"\n[BLACK LLAB DATABASE FACTS]\n{rag_context}\n"
            
        # Inject the pre-computed intelligence into the agent's brain before it boots
        if agent_intelligence_brief:
            clean_text = (
                f"SYSTEM DIRECTIVE: You have access to a vision tool. To read an image in your workspace, run: `python /workspace/vision_tool.py <filename>`\n"
                f"You have also been provided with pre-computed live intelligence. "
                f"DO NOT write scripts to search for this data. Instead, combine the facts below WITH your short-term memory of the previous chat messages to execute the user's task. If the user asks for a comparison to a previous topic (e.g., using words like 'that' or 'the other one'), explicitly integrate both sources of knowledge.\n"
                f"{agent_intelligence_brief}\n\n"
                f"USER TASK:\n{clean_text}"
            )
        else:
            clean_text = (
                f"SYSTEM DIRECTIVE: You have access to a vision tool. To read text, math formulas, "
                f"or tables from an image file in your workspace, run: `python /workspace/vision_tool.py <filename>`\n\n"
                f"USER TASK:\n{clean_text}"
            )

        await session_append_and_maybe_summarize(request.app, session_id, msg)
        
        agent_model_key = selected_model_key
        agent_complexity = eval_data.get("complexity", 100)
        
        async def openclaw_wrapper():
            full_agent_text = ""
            async for chunk in stream_real_openclaw(request.app, clean_text, search_successful, agent_model_key, agent_complexity):
                try:
                    chunk_str = chunk.decode("utf-8")
                    for line in chunk_str.split('\n'):
                        if line.startswith("data: ") and "[DONE]" not in line:
                            parsed = json.loads(line[6:])
                            full_agent_text += parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                except Exception: pass
                yield chunk
                
            if full_agent_text:
                clean_history_text = full_agent_text.replace("> ", "").replace(">", "").strip()
                await session_append_and_maybe_summarize(request.app, session_id, {"role": "assistant", "content": clean_history_text})
                
                asyncio.create_task(update_spend(request.app, agent_model_key, user_text, clean_history_text))
                
        return StreamingResponse(openclaw_wrapper(), media_type="text/event-stream")

    rag_context, memory_context, search_context, graph_context_data = await asyncio.gather(
        fetch_rag(),
        fetch_memory(),
        fetch_search(),
        fetch_graph()
    )

    # 5. History Building
    sess = await session_append_and_maybe_summarize(request.app, session_id, msg)
    llm_messages = [dict(m) for m in sess.get("messages", [])]

    ordered_context_parts = []
    
    if memory_context: ordered_context_parts.append(memory_context)
    if graph_context_data: ordered_context_parts.append(graph_context_data) 
    if rag_context: ordered_context_parts.append(rag_context)
    if search_context: ordered_context_parts.append(search_context)

    if ordered_context_parts and llm_messages:
        xml_context = ""
        for i, ctx in enumerate(ordered_context_parts):
            xml_context += f"<document index=\"{i+1}\">\n{ctx}\n</document>\n\n"
            
        llm_messages[-1]["content"] = (
            f"Analyze the provided context documents to answer the user's query. "
            f"If the data contains weather, stock prices, or news, act as if you have just looked it up yourself in real-time.\n\n"
            f"<context>\n{xml_context}</context>\n\n"
            f"<user_query>\n{user_text}\n</user_query>"
        )

    model_msgs = build_messages_for_model(sess.get("summary", ""), llm_messages, eval_data)
    model_type = MODELS[selected_model_key]["type"]

    prefill_text = ""
    complexity = eval_data.get("complexity", 30)

    if model_type == "local":
        if eval_data.get("is_code"):
            prefill_text = "I have analyzed the constraints. Here is the optimal, production-ready solution:\n\n```python"
            model_msgs.append({"role": "assistant", "content": prefill_text})
        elif complexity > 80:
            prefill_text = "<think>"
            model_msgs.append({"role": "assistant", "content": prefill_text})

    complexity = eval_data.get("complexity", 30)

    if complexity > 55 and requested_model == "auto" and not is_agent_request:
        print(">> [SYSTEM 2] Frontier routing engaged. Spawning MiMo Best-of-N cluster...")
        selected_model_key = "xiaomi"
        MODELS[selected_model_key]["display"] = "Xiaomi System 2 (Cluster)"
        stream_gen = stream_system2_xiaomi(request.app, model_msgs, user_text)
        
    elif model_type == "local":
        stream_gen = stream_ollama(request.app, selected_model_key, model_msgs)
    elif model_type == "openai_compat":
        stream_gen = stream_openai_compat(request.app, selected_model_key, model_msgs)
    elif model_type == "anthropic":
        stream_gen = stream_anthropic(request.app, model_msgs)
    else:
        raise HTTPException(status_code=500, detail="Unknown model routing type")

    async def stream_wrapper():
        yield sse_pack({
            "nexus_meta": {
                "model_name": MODELS[selected_model_key]["display"],
                "model_type": MODELS[selected_model_key]["type"], 
                "complexity": eval_data.get("complexity", 0),
                "search_engine": engine_used
            },
            "choices": [{"delta": {"content": ""}}]
        })

        full_response = ""
        buffer = ""
        
        if prefill_text:
            full_response += prefill_text
            yield sse_pack({"choices": [{"delta": {"content": prefill_text}}]})
        
        async for chunk in stream_gen:
            try:
                chunk_str = chunk.decode("utf-8")
                buffer += chunk_str
            
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line.startswith("data: ") and "[DONE]" not in line:
                        try:
                            parsed = json.loads(line[6:])
                            full_response += parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        except json.JSONDecodeError:
                            pass 
            except Exception as e:
                print(f"Stream decode error: {e}")
                
            yield chunk
            
        if full_response:
            assistant_msg = {"role": "assistant", "content": full_response.strip()}
            await session_append_and_maybe_summarize(request.app, session_id, assistant_msg)
            
        asyncio.create_task(update_spend(request.app, selected_model_key, user_text, full_response))

    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")


from fastapi.responses import FileResponse

import urllib.parse
import ipaddress
@app.get("/v1/agent/vision")
async def api_agent_vision(request: Request, filename: str):
    """Secured endpoint for the OpenClaw Agent to request OCR on an image."""
    _require_token(request)

    workspace_dir = os.path.abspath("./workspace")
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(workspace_dir, safe_filename)
    
    if not os.path.exists(file_path):
        return {"text": f"[VISION ERROR] File '{safe_filename}' not found in workspace."}
        
    if not _P2T_AVAILABLE or p2t is None:
        return {"text": "[VISION ERROR] The Pix2Text engine is currently offline on the host machine."}
        
    try:
        print(f">> [AGENT VISION] Analyzing {safe_filename}...")
        outs = await run_in_thread(lambda: p2t.recognize(file_path))
        
        vision_text = "\n".join([out.get('text', '') for out in outs if isinstance(out, dict)])
        
        if not vision_text.strip():
             return {"text": "[VISION RESULT] No readable text, math, or tables found in this image."}
             
        return {"text": f"[VISION RESULT - {safe_filename}]:\n{vision_text}"}
    except Exception as e:
        return {"text": f"[VISION ERROR] Failed to process image: {str(e)}"}

@app.get("/v1/agent/read_url")
async def api_agent_read_url(request: Request, url: str):
    """Secured endpoint for the OpenClaw Docker sandbox to request deep-scrapes."""
    # 1. ENFORCE AUTHENTICATION
    _require_token(request)
    
    # 2. SSRF FIREWALL: Prevent the agent from scanning your local network
    try:
        parsed_url = urllib.parse.urlparse(url)
        hostname = parsed_url.hostname
        
        # Block obvious local hosts
        if hostname in ["localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"]:
            return {"text": "[SECURITY BLOCK] Access to local hostnames is forbidden."}
            
        # Block internal IP ranges (192.168.x.x, 10.x.x.x, etc.)
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback:
            return {"text": "[SECURITY BLOCK] Access to internal network IPs is forbidden."}
    except ValueError:
        pass 
    except Exception:
        return {"text": "[SECURITY BLOCK] Invalid URL format."}

    # 3. IF SAFE, EXECUTE SCRAPE
    content = await smart_scrape_async(url)
    if not content:
        return {"text": "Error: Could not extract text from this URL. It may have heavy anti-bot protection."}
    return {"text": content}

@app.get("/")
async def serve_ui():
    return FileResponse("index.html")

@app.get("/Logo.jpg")
async def serve_logo():
    import os
    if os.path.exists("Logo.jpg"):
        return FileResponse("Logo.jpg")
    return {"error": "Logo file not found on server"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)