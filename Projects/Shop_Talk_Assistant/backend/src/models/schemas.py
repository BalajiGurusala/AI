"""
ShopTalk Backend — Pydantic models per api-openapi.yaml.

All data exchange uses pydantic (constitution.md §4).
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ======================================================================
# Request Models
# ======================================================================

class Filters(BaseModel):
    price_max: Optional[float] = None
    category: Optional[str] = None


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    query_text: str
    filters: Optional[Filters] = None
    session_context: Optional[List[ChatMessage]] = None


class SearchRequest(BaseModel):
    query_text: str
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Filters] = None


# ======================================================================
# Response Models
# ======================================================================

class Product(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    image_url: Optional[str] = None
    image_caption: Optional[str] = None


class ChatResponse(BaseModel):
    response_text: str
    product_ids: List[str] = []
    products: List[Product] = []
    status: str = "ok"  # ok | no_results | pipeline_error


class SearchResponse(BaseModel):
    product_ids: List[str] = []
    products: List[Product] = []
    total: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    embedding_loaded: bool = False
    chroma_connected: bool = False
    llm_available: bool = False
    product_count: int = 0
