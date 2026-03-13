"""
ShopTalk Backend — Pydantic Schemas.

All data exchange models per data-model.md and api-openapi.yaml.
Uses pydantic v2 (constitution.md: "Use pydantic models for all data exchange").
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class Filters(BaseModel):
    """Shared filter parameters for search and chat endpoints."""
    price_max: Optional[float] = None
    category: Optional[str] = None


class ChatMessage(BaseModel):
    """A single message in session context for follow-up queries."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class Product(BaseModel):
    """Flattened product from ABO dataset."""
    id: str
    title: str
    description: Optional[str] = None
    bullet_points: Optional[str] = None
    keywords: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    image_caption: Optional[str] = None
    detection_confidence: Optional[float] = None


class ChatRequest(BaseModel):
    """POST /api/v1/chat request body."""
    query_text: str
    filters: Optional[Filters] = None
    session_context: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    """POST /api/v1/chat response body."""
    response_text: str
    product_ids: List[str] = Field(default_factory=list)
    products: List[Product] = Field(default_factory=list)
    status: str = "ok"


class SearchRequest(BaseModel):
    """POST /api/v1/search request body."""
    query_text: str
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Filters] = None


class SearchResponse(BaseModel):
    """POST /api/v1/search response body."""
    product_ids: List[str] = Field(default_factory=list)
    products: List[Product] = Field(default_factory=list)
    total: int = 0


class VoiceQueryResponse(BaseModel):
    """POST /api/v1/voice/query response body."""
    transcript: Optional[str] = None
    response_text: str
    product_ids: List[str] = Field(default_factory=list)
    products: List[Product] = Field(default_factory=list)
    status: Optional[str] = None
    audio_base64: Optional[str] = None


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str = "ok"
    embedding_loaded: bool = False
    chroma_connected: bool = False
    llm_available: bool = False
    stt_available: bool = False
    tts_available: bool = False
    product_count: int = 0
