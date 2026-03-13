"""
ShopTalk – AI Shopping Assistant (Streamlit UI)

A modern chat-based interface for the ShopTalk AI-powered shopping assistant.
Connects to the RAG pipeline (hybrid search + LLM generation) built in notebooks 03-04.

Usage:
    streamlit run app/streamlit_app.py

Prerequisites:
    - NB03/NB04 artifacts in data/ directory (rag_products.pkl, embeddings, config)
    - .env file with OPENAI_API_KEY and/or GROQ_API_KEY
    - Product images in data/images/small/ (optional, for product cards)
"""

import os
import sys
import io
import json
import time
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
import httpx

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ShopTalk AI Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global */
    .stApp { background-color: #fafafa; }

    /* Chat container */
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }

    /* Product card styling */
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 12px;
        transition: transform 0.2s;
        border: 1px solid #eee;
    }
    .product-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .product-title {
        font-weight: 600;
        font-size: 14px;
        color: #1a1a2e;
        margin-bottom: 6px;
        line-height: 1.3;
    }
    .product-meta {
        font-size: 12px;
        color: #666;
        margin-bottom: 4px;
    }
    .product-price {
        font-size: 18px;
        font-weight: 700;
        color: #e63946;
        margin-top: 8px;
    }
    .product-category {
        display: inline-block;
        background: #e8f4f8;
        color: #1a759f;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
    }

    /* Status messages */
    .status-msg {
        text-align: center;
        color: #666;
        font-style: italic;
        padding: 8px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #fff !important;
        font-weight: 500;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Header */
    .app-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .app-header h1 {
        color: #1a1a2e;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .app-header p {
        color: #666;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Data & Model Loading (cached for performance)
# ===========================================================================

@st.cache_resource(show_spinner="Loading product catalog...")
def load_data():
    """Load product data and embedding indexes from NB03/NB04 artifacts."""
    candidates = [
        Path("data"),
        Path("../data"),
        Path("."),
        Path("/kaggle/working"),
    ]

    data_dir = None
    for d in candidates:
        if (d / "rag_products.pkl").exists():
            data_dir = d
            break
        # Also check for products_with_prices.pkl (NB04 output)
        if (d / "products_with_prices.pkl").exists():
            data_dir = d
            break

    if data_dir is None:
        st.error("Cannot find product data. Run notebooks 03-04 first, "
                 "then place artifacts in data/ directory.")
        st.stop()

    # Prefer products with prices (NB04) over raw products (NB03)
    if (data_dir / "products_with_prices.pkl").exists():
        df = pd.read_pickle(data_dir / "products_with_prices.pkl")
    else:
        df = pd.read_pickle(data_dir / "rag_products.pkl")

    text_index = np.load(data_dir / "rag_text_index.npy")
    image_index = np.load(data_dir / "rag_image_index.npy")

    config = {}
    if (data_dir / "rag_config.json").exists():
        with open(data_dir / "rag_config.json") as f:
            config = json.load(f)

    # Load fine-tuned embeddings if available
    ft_path = data_dir / "finetuned_text_index.npy"
    if ft_path.exists():
        text_index = np.load(ft_path)

    return df, text_index, image_index, config, data_dir


@st.cache_resource(show_spinner="Loading search models (ST + CLIP)...")
def load_search_models(config: dict, data_dir: Path):
    """Load SentenceTransformer and CLIP at startup (once, then cached).

    Loading everything upfront means all queries are instant. CLIP is
    best-effort: if it fails the app falls back to text-only search.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
        "cpu"
    )

    st_model_id = config.get("text_model_id", "all-MiniLM-L6-v2")
    clip_model_id = config.get("image_model_id", "openai/clip-vit-base-patch32")

    ft_env_path = os.getenv("FINETUNED_MODEL_PATH", "").strip()
    if ft_env_path and Path(ft_env_path).exists():
        st_model_source = ft_env_path
    else:
        default_ft_path = data_dir / "models" / "finetuned-shoptalk-emb"
        st_model_source = str(default_ft_path) if default_ft_path.exists() else st_model_id

    st_model = SentenceTransformer(st_model_source, device=device)

    clip_model, clip_processor = None, None
    try:
        import logging as _logging
        # Suppress noisy-but-harmless CLIP load messages:
        #   "unexpected keys" (position_ids), "layers not sharded", HF Hub auth hint
        for _noisy_logger in (
            "transformers.modeling_utils",
            "transformers.models.clip.modeling_clip",
            "huggingface_hub.file_download",
            "huggingface_hub.utils._headers",
        ):
            _logging.getLogger(_noisy_logger).setLevel(_logging.ERROR)
        clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        for _noisy_logger in (
            "transformers.modeling_utils",
            "transformers.models.clip.modeling_clip",
            "huggingface_hub.file_download",
            "huggingface_hub.utils._headers",
        ):
            _logging.getLogger(_noisy_logger).setLevel(_logging.WARNING)
    except Exception as e:
        st.warning(
            f"⚠️ CLIP model could not be loaded ({type(e).__name__}). "
            "Falling back to **text-only search** — all features still work.",
            icon="ℹ️",
        )

    return st_model, clip_model, clip_processor, device


def _make_ssl_httpx_client():
    """Return an httpx.Client configured with the corporate SSL cert if present.

    httpx does not automatically read SSL_CERT_FILE when instantiated inside
    third-party libraries (OpenAI, Groq). We inject the cert explicitly so
    that API calls work through a corporate HTTPS proxy.
    """
    import ssl as _ssl
    import httpx as _httpx

    cert_path = (
        os.getenv("SSL_CERT_FILE", "")
        or os.getenv("REQUESTS_CA_BUNDLE", "")
    ).strip()

    if cert_path and Path(cert_path).exists():
        ctx = _ssl.create_default_context(cafile=cert_path)
        return _httpx.Client(verify=ctx, timeout=30.0)

    return _httpx.Client(timeout=30.0)


@st.cache_resource(show_spinner="Connecting to LLM...")
def load_llm(model_key: str):
    """Load LLM client based on model selection.

    model_key format:
        "ollama/<model>"     → local Ollama
        "gpt-4o-mini"        → OpenAI
        "groq/<model>"       → Groq cloud
    """
    from dotenv import load_dotenv
    load_dotenv()

    if model_key.startswith("ollama/"):
        try:
            from langchain_ollama import ChatOllama
            ollama_model = model_key.split("/", 1)[1]
            return ChatOllama(model=ollama_model, temperature=0.3, num_predict=512)
        except Exception:
            return None

    elif model_key.startswith("gpt"):
        try:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                return None
            return ChatOpenAI(
                model=model_key, api_key=api_key,
                temperature=0.3, max_tokens=512, request_timeout=30,
                http_client=_make_ssl_httpx_client(),
            )
        except Exception:
            return None

    elif model_key.startswith("groq/"):
        try:
            from langchain_groq import ChatGroq
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                return None
            groq_model = model_key.split("/", 1)[1]
            return ChatGroq(
                model=groq_model, api_key=api_key,
                temperature=0.3, max_tokens=512,
                http_client=_make_ssl_httpx_client(),
            )
        except Exception:
            return None

    return None


def detect_available_llms() -> Dict[str, str]:
    """Detect which LLMs are available and return {display_name: model_key}."""
    options = {}

    # 1. Ollama (local, free)
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            for m in models[:3]:  # Show up to 3 Ollama models
                options[f"Ollama: {m} (local, free)"] = f"ollama/{m}"
    except Exception:
        pass

    # 2. OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        options["GPT-4o-mini (OpenAI)"] = "gpt-4o-mini"

    # 3. Groq
    if os.getenv("GROQ_API_KEY"):
        options["Llama-3.3-70B (Groq cloud)"] = "groq/llama-3.3-70b-versatile"

    return options


# ===========================================================================
# Hybrid Search Pipeline — imported from src/search.py (single source of truth)
# ===========================================================================

# Add project root to path so we can import the shared search module
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.search import hybrid_search as _hybrid_search, l2_normalize


def run_search(
    query: str,
    df: pd.DataFrame,
    text_index: np.ndarray,
    image_index: np.ndarray,
    st_model,
    clip_model,
    clip_processor,
    device: str,
    top_k: int = 5,
    price_max: float = None,
    category: str = None,
) -> pd.DataFrame:
    """Thin wrapper: creates query encoders and delegates to src/search.hybrid_search.

    Falls back to text-only (alpha=1.0) if clip_model is None.
    """
    import torch

    def encode_text(q: str) -> np.ndarray:
        return st_model.encode([q], show_progress_bar=False,
                               normalize_embeddings=True).astype(np.float32)

    if clip_model is not None and clip_processor is not None:
        def encode_clip_fn(q: str) -> np.ndarray:
            inputs = clip_processor(text=[q], return_tensors="pt",
                                    padding=True, truncation=True).to(device)
            with torch.no_grad():
                feats = clip_model.get_text_features(**inputs)
            # transformers ≥4.x may return a model-output object instead of a
            # plain tensor depending on the model config's return_dict setting.
            # Use explicit None checks — 'or' triggers bool(tensor) which raises.
            if not isinstance(feats, torch.Tensor):
                t = getattr(feats, "text_embeds", None)
                if t is None:
                    t = getattr(feats, "pooler_output", None)
                if t is None:
                    t = feats.last_hidden_state[:, 0, :]
                feats = t
            return l2_normalize(feats.cpu().numpy().astype(np.float32))
        alpha = None  # dynamic alpha (text + image)
    else:
        # Text-only fallback: zero CLIP vector, alpha=1.0 ignores image scores
        clip_dim = image_index.shape[1] if image_index is not None else 512

        def encode_clip_fn(q: str) -> np.ndarray:  # noqa: F811
            return np.zeros((1, clip_dim), dtype=np.float32)
        alpha = 1.0  # pure text search

    return _hybrid_search(
        query=query,
        df=df,
        text_index=text_index,
        image_index=image_index,
        encode_text_fn=encode_text,
        encode_clip_fn=encode_clip_fn,
        top_k=top_k,
        alpha=alpha,
        price_max=price_max,
        category=category,
    )


# ===========================================================================
# LLM / RAG Generation
# ===========================================================================

SYSTEM_PROMPT = """You are ShopTalk, a friendly and knowledgeable AI shopping assistant.
Your job is to help customers find products from our catalog and answer questions about them.

RULES:
1. ONLY recommend products from the provided context — never invent products or details.
2. Be concise and helpful: aim for 2-4 sentences per recommendation.
3. Highlight relevant features that match the customer's query (color, material, brand, etc.).
4. When showing multiple products, briefly explain WHY each is relevant.
5. If no products match, honestly say so and suggest different keywords.
6. Use a warm, conversational tone — like a helpful store associate.
7. When prices are available, mention them.
8. Keep responses under 150 words unless asked for details."""


def format_context(results: pd.DataFrame, max_products: int = 5) -> str:
    """Format products into LLM context string."""
    if results.empty:
        return "No products found."
    lines = []
    for i, (_, row) in enumerate(results.head(max_products).iterrows()):
        parts = [f"Product {i+1}: {row.get('item_name_flat', 'Unknown')}"]
        parts.append(f"  ID: {row.get('item_id', 'N/A')}")
        if pd.notna(row.get("brand_flat")) and str(row["brand_flat"]).strip():
            parts.append(f"  Brand: {row['brand_flat']}")
        if pd.notna(row.get("product_type_flat")):
            parts.append(f"  Category: {row['product_type_flat']}")
        if pd.notna(row.get("color_flat")) and str(row["color_flat"]).strip():
            parts.append(f"  Color: {row['color_flat']}")
        if pd.notna(row.get("price")):
            parts.append(f"  Price: ${row['price']:.2f}")
        if pd.notna(row.get("bullet_point_flat")):
            parts.append(f"  Features: {str(row['bullet_point_flat'])[:300]}")
        if pd.notna(row.get("image_caption")) and str(row["image_caption"]).strip():
            parts.append(f"  Appearance: {row['image_caption']}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def rag_generate(
    query: str,
    results: pd.DataFrame,
    llm,
    session_history: list,
) -> str:
    """Generate LLM response from search results and conversation history."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    context = format_context(results, max_products=5)
    user_msg = (
        f"Here are the top matching products:\n\n{context}\n\n---\n"
        f"Customer query: {query}\n\n"
        f"Provide a helpful, concise recommendation."
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in (session_history or [])[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_msg))

    response = llm.invoke(messages)
    return response.content


# ===========================================================================
# Product Card Rendering
# ===========================================================================

ABO_IMAGE_BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small"


def render_product_card(row: pd.Series, data_dir: Path, col, key_prefix: str = ""):
    """Render a product card in a Streamlit column."""
    with col:
        # Image: try local file first, then ABO public S3 URL, then placeholder
        image_src = None
        img_path_val = row.get("path")
        if pd.notna(img_path_val) and str(img_path_val).strip():
            rel_path = str(img_path_val).strip()
            for base in [data_dir / "images" / "small", data_dir / ".." / "data" / "images" / "small"]:
                candidate = base / rel_path
                if candidate.exists():
                    image_src = str(candidate)
                    break
            if image_src is None:
                # Fall back to public ABO S3 URL (no local images needed)
                image_src = f"{ABO_IMAGE_BASE_URL}/{rel_path}"

        if image_src:
            st.image(image_src, width="stretch")
        else:
            st.markdown(
                '<div style="background:#f0f0f0; border-radius:8px; height:150px; '
                'display:flex; align-items:center; justify-content:center; color:#999;">'
                '📷 No Image</div>',
                unsafe_allow_html=True,
            )

        # Title
        title = str(row.get("item_name_flat", "Unknown Product"))[:80]
        st.markdown(f'<div class="product-title">{title}</div>', unsafe_allow_html=True)

        # Category badge
        cat = row.get("product_type_flat", "")
        if pd.notna(cat) and str(cat).strip():
            st.markdown(f'<span class="product-category">{cat}</span>', unsafe_allow_html=True)

        # Brand
        brand = row.get("brand_flat", "")
        if pd.notna(brand) and str(brand).strip():
            st.markdown(f'<div class="product-meta">Brand: {brand}</div>', unsafe_allow_html=True)

        # Color
        color = row.get("color_flat", "")
        if pd.notna(color) and str(color).strip():
            st.markdown(f'<div class="product-meta">Color: {color}</div>', unsafe_allow_html=True)

        # Price
        price = row.get("price")
        if pd.notna(price):
            st.markdown(f'<div class="product-price">${price:.2f}</div>', unsafe_allow_html=True)

        # Caption
        caption = row.get("image_caption", "")
        if pd.notna(caption) and str(caption).strip():
            st.caption(f"📸 {caption}")

        # Add to Cart button (mock)
        st.button("🛒 Add to Cart", key=f"cart_{key_prefix}{row.get('item_id', id(row))}", type="secondary")


def render_product_grid(results: pd.DataFrame, data_dir: Path, key_prefix: str = ""):
    """Render a grid of product cards."""
    if results.empty:
        return

    n_products = min(len(results), 5)
    n_cols = min(n_products, 3)
    cols = st.columns(n_cols)

    for i in range(n_products):
        row = results.iloc[i]
        col = cols[i % n_cols]
        render_product_card(row, data_dir, col, key_prefix=key_prefix)


# ===========================================================================
# Voice: STT (Whisper) + TTS (gTTS)
# ===========================================================================

@st.cache_resource(show_spinner="Loading Whisper STT model...")
def load_whisper():
    """Load Whisper model for local STT. Returns None if unavailable."""
    try:
        import whisper
        model = whisper.load_model("base")
        return model
    except Exception:
        return None


def local_transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio locally using Whisper."""
    import tempfile
    whisper_model = load_whisper()
    if whisper_model is None:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    try:
        result = whisper_model.transcribe(temp_path, language="en", fp16=False)
        return result.get("text", "").strip()
    except Exception:
        return ""
    finally:
        Path(temp_path).unlink(missing_ok=True)


def local_tts(text: str) -> tuple[bytes, str]:
    """Convert text to speech. Returns (audio_bytes, format_string).

    Strategy (offline-first):
      1. macOS built-in `say` + `afconvert` to WAV  — no network, instant
      2. gTTS (Google TTS)                           — requires internet
      3. Returns (b"", "") if both fail
    """
    import subprocess, tempfile, os, platform

    # Truncate very long responses to avoid excessive audio
    tts_text = text[:600] + ("..." if len(text) > 600 else "")

    # --- Option 1: macOS offline TTS (say + afconvert) ---
    if platform.system() == "Darwin":
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                aiff_path = os.path.join(tmp_dir, "out.aiff")
                wav_path  = os.path.join(tmp_dir, "out.wav")
                subprocess.run(
                    ["say", "-o", aiff_path, tts_text],
                    timeout=15, check=True, capture_output=True,
                )
                subprocess.run(
                    ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff_path, wav_path],
                    timeout=10, check=True, capture_output=True,
                )
                with open(wav_path, "rb") as f:
                    return f.read(), "audio/wav"
        except Exception:
            pass  # fall through to gTTS

    # --- Option 2: gTTS (online) ---
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        tts = gTTS(text=tts_text, lang="en", slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read(), "audio/mp3"
    except Exception:
        pass

    return b"", ""


# ===========================================================================
# Backend API Client (thin-client mode for Docker/EC2 deployment)
# ===========================================================================

BACKEND_URL = os.getenv("BACKEND_URL", "")


def _backend_available() -> bool:
    """Check if the FastAPI backend is reachable."""
    if not BACKEND_URL:
        return False
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _backend_chat(query: str, top_k: int, price_max: float = None,
                  category: str = None, session_history: list = None) -> dict:
    """Call the backend /api/v1/chat endpoint."""
    payload = {
        "query_text": query,
        "filters": {},
    }
    if price_max is not None:
        payload["filters"]["price_max"] = price_max
    if category:
        payload["filters"]["category"] = category
    if session_history:
        payload["session_context"] = [
            {"role": m["role"], "content": m["content"]}
            for m in session_history[-6:]
        ]

    r = httpx.post(f"{BACKEND_URL}/api/v1/chat", json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()


def _backend_voice(audio_bytes: bytes, price_max: float = None,
                   category: str = None) -> dict:
    """Call the backend /api/v1/voice/query endpoint."""
    files = {"audio": ("recording.wav", audio_bytes, "audio/wav")}
    data = {}
    if price_max is not None:
        data["price_max"] = str(price_max)
    if category:
        data["category"] = category

    r = httpx.post(f"{BACKEND_URL}/api/v1/voice/query",
                   files=files, data=data, timeout=30.0)
    r.raise_for_status()
    return r.json()


def _backend_search(query: str, top_k: int, price_max: float = None,
                    category: str = None) -> dict:
    """Call the backend /api/v1/search endpoint."""
    payload = {
        "query_text": query,
        "top_k": top_k,
        "filters": {},
    }
    if price_max is not None:
        payload["filters"]["price_max"] = price_max
    if category:
        payload["filters"]["category"] = category

    r = httpx.post(f"{BACKEND_URL}/api/v1/search", json=payload, timeout=15.0)
    r.raise_for_status()
    return r.json()


# ===========================================================================
# Main App
# ===========================================================================

def main():
    # --- Detect mode: thin-client (backend API) vs standalone (local models) ---
    USE_BACKEND = _backend_available()

    if USE_BACKEND:
        # Thin-client mode: backend handles search + LLM
        df = pd.DataFrame()  # Placeholder
        data_dir = Path(".")
        device = "backend"
        # Fetch product metadata from backend health
        try:
            health = httpx.get(f"{BACKEND_URL}/health", timeout=3).json()
            product_count = health.get("product_count", 0)
        except Exception:
            product_count = 0
    else:
        # Standalone mode: load models locally
        df, text_index, image_index, config, data_dir = load_data()
        st_model, clip_model, clip_processor, device = load_search_models(config, data_dir)
        product_count = len(df)

    # Early session-state init (needed by sidebar audio widget)
    if "voice_input_nonce" not in st.session_state:
        st.session_state.voice_input_nonce = 0

    # ==================================================================
    # Sidebar
    # ==================================================================
    with st.sidebar:
        st.markdown("## 🛍️ ShopTalk")
        st.markdown("*AI Shopping Assistant*")
        st.divider()

        # Model selection
        st.markdown("### ⚙️ Settings")
        if USE_BACKEND:
            st.success(f"Connected to backend")
            selected_model = "backend"
        else:
            llm_options = detect_available_llms()
            if not llm_options:
                st.warning("No LLM available! Install Ollama or set API keys in .env")
                llm_options = {"None (search only)": "none"}

            selected_name = st.selectbox(
                "LLM Model",
                options=list(llm_options.keys()),
                index=0,
            )
            selected_model = llm_options[selected_name]

        st.divider()

        # Filters
        st.markdown("### 🔍 Filters")
        price_max = None  # ABO dataset has no prices

        # Category filter
        if not USE_BACKEND and not df.empty:
            categories = sorted(df["product_type_flat"].dropna().unique().tolist())
        else:
            categories = []
        category = st.selectbox(
            "Category",
            options=["All Categories"] + categories,
            index=0,
        )
        if category == "All Categories":
            category = None

        # Top-K
        top_k = st.slider("Results to show", 1, 10, 5)

        st.divider()

        # Stats
        st.markdown("### 📊 Catalog Stats")
        st.markdown(f"- **Products:** {product_count:,}")
        if not USE_BACKEND and not df.empty:
            st.markdown(f"- **Categories:** {df['product_type_flat'].nunique()}")
        st.markdown(f"- **Mode:** {'Backend API' if USE_BACKEND else f'Local ({device})'}")

        # Voice settings
        st.divider()
        st.markdown("### 🎙️ Voice")
        voice_enabled = st.toggle("Enable voice input", value=False)
        tts_enabled = st.toggle("Read responses aloud", value=False)

        _sidebar_audio_data = None
        if voice_enabled:
            _voice_key = f"voice_input_{st.session_state.voice_input_nonce}"
            _sidebar_audio_data = st.audio_input(
                "🎙️ Record your question", key=_voice_key,
            )

        # Clear chat
        st.divider()
        if st.button("🗑️ Clear Chat", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.products = {}
            st.rerun()

    # ==================================================================
    # Initialize Session State
    # ==================================================================
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "products" not in st.session_state:
        st.session_state.products = {}  # Maps message index to product results
    # ==================================================================
    # Header
    # ==================================================================
    st.markdown(
        '<div class="app-header">'
        '<h1>🛍️ ShopTalk AI Assistant</h1>'
        '<p>Ask me about any product — I\'ll find the best matches for you!</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ==================================================================
    # Chat History
    # ==================================================================
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show product cards for assistant messages
            if msg["role"] == "assistant" and i in st.session_state.products:
                products_df = st.session_state.products[i]
                if not products_df.empty:
                    st.markdown("---")
                    render_product_grid(products_df, data_dir, key_prefix=f"h{i}_")

    # ==================================================================
    # Voice Input Processing (mic widget lives in sidebar)
    # ==================================================================
    if voice_enabled and _sidebar_audio_data is not None:
        audio_bytes = _sidebar_audio_data.read()
        audio_hash = hash(audio_bytes) if audio_bytes else None
        already_processed = audio_hash and audio_hash == st.session_state.get("_last_voice_hash")

        if audio_bytes and not already_processed:
            st.session_state._last_voice_hash = audio_hash
            with st.status("🎙️ Transcribing audio...", expanded=True) as stt_status:
                if USE_BACKEND:
                    try:
                        voice_result = _backend_voice(
                            audio_bytes, price_max=price_max, category=category,
                        )
                        transcript = voice_result.get("transcript", "")
                        stt_status.write(f"✅ Heard: \"{transcript}\"")

                        if transcript:
                            st.session_state.messages.append({
                                "role": "user",
                                "content": f"🎙️ {transcript}",
                            })
                            st.session_state._voice_result = voice_result
                            st.session_state._voice_transcript = transcript
                            st.session_state.voice_input_nonce += 1
                            stt_status.update(label=f"Transcribed: \"{transcript}\"", state="complete")
                            st.rerun()
                        else:
                            stt_status.update(label="Could not understand audio", state="error")
                    except Exception as e:
                        stt_status.write(f"❌ Error: {str(e)[:100]}")
                        stt_status.update(label="Voice error", state="error")
                else:
                    transcript = local_transcribe(audio_bytes)
                    if transcript:
                        stt_status.write(f"✅ Heard: \"{transcript}\"")
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"🎙️ {transcript}",
                        })
                        st.session_state._voice_transcript = transcript
                        st.session_state.voice_input_nonce += 1
                        stt_status.update(label=f"Transcribed: \"{transcript}\"", state="complete")
                        st.rerun()
                    else:
                        stt_status.update(label="Could not understand audio", state="error")

    # Process pending voice result (after rerun from voice input)
    if "_voice_transcript" in st.session_state:
        prompt = st.session_state._voice_transcript
        voice_result = st.session_state.pop("_voice_result", None)
        del st.session_state._voice_transcript

        with st.chat_message("user"):
            st.markdown(f"🎙️ {prompt}")

        with st.chat_message("assistant"):
            status = st.status("Processing voice query...", expanded=True)
            response_text = ""
            results = pd.DataFrame()

            if USE_BACKEND and voice_result:
                # We already have the result from the voice endpoint
                response_text = voice_result.get("response_text", "")
                if voice_result.get("products"):
                    results = pd.DataFrame(voice_result["products"])
                    rename_map = {"title": "item_name_flat", "id": "item_id",
                                  "category": "product_type_flat", "brand": "brand_flat",
                                  "color": "color_flat", "image_url": "path"}
                    results = results.rename(columns={k: v for k, v in rename_map.items() if k in results.columns})
                status.update(label="Done", state="complete")

                # Play TTS audio if available
                if tts_enabled and voice_result.get("audio_base64"):
                    audio_bytes_tts = base64.b64decode(voice_result["audio_base64"])
                    st.audio(audio_bytes_tts, format="audio/mp3")
            else:
                # Standalone mode: run RAG pipeline locally
                status.write("🔍 Searching...")
                t0 = time.time()
                results = run_search(
                    query=prompt, df=df,
                    text_index=text_index, image_index=image_index,
                    st_model=st_model, clip_model=clip_model,
                    clip_processor=clip_processor, device=device,
                    top_k=top_k, price_max=price_max, category=category,
                )
                search_time = time.time() - t0
                status.write(f"✅ Found {len(results)} products ({search_time:.2f}s)")

                if results.empty:
                    response_text = "I couldn't find products matching that query."
                else:
                    llm = load_llm(selected_model) if selected_model != "none" else None
                    if llm:
                        try:
                            response_text = rag_generate(query=prompt, results=results, llm=llm, session_history=[])
                        except Exception as e:
                            response_text = "Here are the matching products:"
                            status.write(f"⚠️ LLM error: {str(e)[:80]}")
                    else:
                        response_text = "Here are the matching products:"

                status.update(label="Done", state="complete")

                # Play TTS locally
                if tts_enabled and response_text:
                    tts_audio, tts_fmt = local_tts(response_text)
                    if tts_audio:
                        st.audio(tts_audio, format=tts_fmt)

            st.markdown(response_text)

            msg_idx = len(st.session_state.messages)
            if not results.empty:
                st.session_state.products[msg_idx] = results
                st.markdown("---")
                render_product_grid(results, data_dir, key_prefix=f"v{msg_idx}_")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
            })

    # ==================================================================
    # Chat Input (text)
    # ==================================================================
    if prompt := st.chat_input("What are you looking for? (e.g., 'red shoes for women')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            status = st.status("Processing your query...", expanded=True)

            gen_time = 0.0
            response_text = ""
            results = pd.DataFrame()

            if USE_BACKEND:
                # ---- THIN-CLIENT MODE: call backend API ----
                status.write("🔍 Searching via backend API...")
                t0 = time.time()
                try:
                    session_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    api_result = _backend_chat(
                        query=prompt, top_k=top_k,
                        price_max=price_max, category=category,
                        session_history=session_history,
                    )
                    response_text = api_result.get("response_text", "")
                    total_time = time.time() - t0

                    # Convert API products to DataFrame for product cards
                    if api_result.get("products"):
                        results = pd.DataFrame(api_result["products"])
                        # Rename fields for product card renderer
                        rename_map = {"title": "item_name_flat", "id": "item_id",
                                      "category": "product_type_flat", "brand": "brand_flat",
                                      "color": "color_flat", "image_url": "path"}
                        results = results.rename(columns={k: v for k, v in rename_map.items() if k in results.columns})

                    status.write(f"✅ Done ({total_time:.2f}s)")
                    status.update(label=f"Done ({total_time:.2f}s)", state="complete")
                except Exception as e:
                    response_text = "Something went wrong. Please try again."
                    status.write(f"❌ Error: {str(e)[:100]}")
                    status.update(label="Error", state="error")

            else:
                # ---- STANDALONE MODE: local models ----
                status.write("🔍 Searching product catalog...")

                t0 = time.time()
                results = run_search(
                    query=prompt, df=df,
                    text_index=text_index, image_index=image_index,
                    st_model=st_model, clip_model=clip_model,
                    clip_processor=clip_processor, device=device,
                    top_k=top_k, price_max=price_max, category=category,
                )
                search_time = time.time() - t0
                status.write(f"✅ Found {len(results)} products ({search_time:.2f}s)")

                if results.empty:
                    response_text = (
                        "I couldn't find any products matching that query. "
                        "Try different keywords or adjust your filters."
                    )
                    status.update(label="No results found", state="complete")
                else:
                    status.write("💬 Generating recommendation...")

                    llm = load_llm(selected_model) if selected_model != "none" else None
                    if llm is None:
                        response_text = "Here are the top matching products:"
                        for i, (_, row) in enumerate(results.head(3).iterrows()):
                            title = row.get("item_name_flat", "Unknown")
                            price = row.get("price")
                            price_str = f" - ${price:.2f}" if pd.notna(price) else ""
                            response_text += f"\n\n{i+1}. **{title}**{price_str}"
                    else:
                        try:
                            session_history = [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages[:-1]
                            ]
                            t1 = time.time()
                            response_text = rag_generate(
                                query=prompt, results=results,
                                llm=llm, session_history=session_history,
                            )
                            gen_time = time.time() - t1
                            status.write(f"✅ Response generated ({gen_time:.2f}s)")
                        except Exception as e:
                            response_text = "Something went wrong. Please try again."
                            status.write(f"❌ Error: {str(e)[:100]}")

                    status.update(
                        label=f"Done ({search_time + gen_time:.2f}s)",
                        state="complete",
                    )

            # Display response
            st.markdown(response_text)

            # TTS: Read response aloud (text chat mode)
            if tts_enabled and response_text:
                tts_audio, tts_fmt = local_tts(response_text)
                if tts_audio:
                    st.audio(tts_audio, format=tts_fmt)

            # Store and display products
            msg_idx = len(st.session_state.messages)
            if not results.empty:
                st.session_state.products[msg_idx] = results
                st.markdown("---")
                render_product_grid(results, data_dir, key_prefix=f"t{msg_idx}_")

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
            })


if __name__ == "__main__":
    main()
