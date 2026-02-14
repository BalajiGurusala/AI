"""
ShopTalk Backend — RAG Pipeline Service.

Combines hybrid search + LLM generation into a single service.
"""

import time
import logging
from typing import Optional, List, Dict, Any

import pandas as pd

from backend.src.services.embeddings import embedding_service
from backend.src.models.schemas import Product, Filters

logger = logging.getLogger(__name__)

# Import search logic from shared module
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from src.search import hybrid_search


# ======================================================================
# System Prompt
# ======================================================================

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

USER_TEMPLATE = """Here are the top matching products:

{product_context}

---
Customer query: {query}

Provide a helpful, concise recommendation."""


# ======================================================================
# LLM Manager
# ======================================================================

class LLMManager:
    """Manages LLM connections. Tries Ollama first, then fallbacks."""

    def __init__(self):
        self._llm = None
        self._name = "none"

    @property
    def is_available(self) -> bool:
        return self._llm is not None

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, ollama_url: str, ollama_model: str,
                   openai_key: str = None, groq_key: str = None,
                   default: str = "ollama"):
        """Try to connect to an LLM. Priority: Ollama → OpenAI → Groq."""

        # 1. Ollama
        try:
            from langchain_ollama import ChatOllama
            import httpx
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                model = ollama_model if any(ollama_model in m for m in models) else (models[0] if models else None)
                if model:
                    self._llm = ChatOllama(
                        model=model, base_url=ollama_url,
                        temperature=0.3, num_predict=512,
                    )
                    self._name = f"ollama/{model}"
                    logger.info(f"LLM: {self._name}")
                    return
        except Exception as e:
            logger.info(f"Ollama not available: {e}")

        # 2. OpenAI
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model="gpt-4o-mini", api_key=openai_key,
                    temperature=0.3, max_tokens=512, request_timeout=30,
                )
                self._name = "gpt-4o-mini"
                logger.info(f"LLM: {self._name}")
                return
            except Exception as e:
                logger.warning(f"OpenAI failed: {e}")

        # 3. Groq
        if groq_key:
            try:
                from langchain_groq import ChatGroq
                self._llm = ChatGroq(
                    model="llama-3.3-70b-versatile", api_key=groq_key,
                    temperature=0.3, max_tokens=512,
                )
                self._name = "groq/llama-3.3-70b"
                logger.info(f"LLM: {self._name}")
                return
            except Exception as e:
                logger.warning(f"Groq failed: {e}")

        logger.warning("No LLM available — generation will be disabled")

    def generate(self, messages: list) -> str:
        """Generate a response from messages."""
        if not self._llm:
            raise RuntimeError("No LLM available")
        response = self._llm.invoke(messages)
        return response.content


# Singleton
llm_manager = LLMManager()


# ======================================================================
# RAG Service
# ======================================================================

def _format_context(results: pd.DataFrame, max_products: int = 5) -> str:
    """Format products for LLM context."""
    if results.empty:
        return "No products found."
    lines = []
    for i, (_, row) in enumerate(results.head(max_products).iterrows()):
        parts = [f"Product {i+1}: {row.get('item_name_flat', 'Unknown')}"]
        parts.append(f"  ID: {row.get('item_id', 'N/A')}")
        for field, label in [("brand_flat", "Brand"), ("product_type_flat", "Category"),
                             ("color_flat", "Color")]:
            v = row.get(field)
            if pd.notna(v) and str(v).strip():
                parts.append(f"  {label}: {v}")
        if pd.notna(row.get("price")):
            parts.append(f"  Price: ${row['price']:.2f}")
        if pd.notna(row.get("bullet_point_flat")):
            parts.append(f"  Features: {str(row['bullet_point_flat'])[:300]}")
        if pd.notna(row.get("image_caption")) and str(row.get("image_caption", "")).strip():
            parts.append(f"  Appearance: {row['image_caption']}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _row_to_product(row, image_base: str = "/images") -> Product:
    """Convert a DataFrame row to a Product schema."""
    image_url = None
    if pd.notna(row.get("path")):
        image_url = f"{image_base}/{row['path']}"

    return Product(
        id=str(row.get("item_id", "")),
        title=str(row.get("item_name_flat", "Unknown")),
        description=str(row.get("bullet_point_flat", "")) if pd.notna(row.get("bullet_point_flat")) else None,
        price=float(row["price"]) if pd.notna(row.get("price")) else None,
        category=str(row.get("product_type_flat", "")) if pd.notna(row.get("product_type_flat")) else None,
        brand=str(row.get("brand_flat", "")) if pd.notna(row.get("brand_flat")) else None,
        color=str(row.get("color_flat", "")) if pd.notna(row.get("color_flat")) else None,
        image_url=image_url,
        image_caption=str(row.get("image_caption", "")) if pd.notna(row.get("image_caption")) else None,
    )


def search_products(
    query: str,
    top_k: int = 5,
    filters: Filters = None,
) -> pd.DataFrame:
    """Run hybrid search with filters."""
    svc = embedding_service
    return hybrid_search(
        query=query,
        df=svc.df,
        text_index=svc.text_index,
        image_index=svc.image_index,
        encode_text_fn=svc.encode_text,
        encode_clip_fn=svc.encode_clip,
        top_k=top_k,
        price_max=filters.price_max if filters else None,
        category=filters.category if filters else None,
    )


def rag_query(
    query: str,
    top_k: int = 5,
    filters: Filters = None,
    session_context: list = None,
) -> Dict[str, Any]:
    """Full RAG pipeline: search → context → LLM → response."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    result = {
        "response_text": "",
        "product_ids": [],
        "products": [],
        "status": "ok",
    }

    # Stage 1: Search
    t0 = time.time()
    try:
        search_results = search_products(query, top_k=top_k, filters=filters)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        result["status"] = "pipeline_error"
        result["response_text"] = "Something went wrong with the search. Please try again."
        return result

    if search_results.empty:
        result["status"] = "no_results"
        result["response_text"] = "No products match that query. Try different keywords or broaden your filters."
        return result

    result["product_ids"] = search_results["item_id"].astype(str).tolist()[:top_k]
    result["products"] = [_row_to_product(row) for _, row in search_results.head(top_k).iterrows()]

    # Stage 2: Generate (if LLM available)
    if not llm_manager.is_available:
        result["response_text"] = "Here are the matching products."
        return result

    context = _format_context(search_results, max_products=top_k)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if session_context:
        for msg in session_context[-6:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

    messages.append(HumanMessage(content=USER_TEMPLATE.format(
        product_context=context, query=query,
    )))

    try:
        result["response_text"] = llm_manager.generate(messages)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        result["status"] = "pipeline_error"
        result["response_text"] = "Something went wrong. Please try again."

    return result
