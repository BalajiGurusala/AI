"""
ShopTalk Backend — API Routes per api-openapi.yaml.

GET  /health             → HealthResponse
POST /api/v1/chat        → ChatResponse  (text → RAG → response)
POST /api/v1/search      → SearchResponse (text → product list)
POST /api/v1/voice/query → VoiceQueryResponse (audio → STT → RAG → TTS)
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from backend.src.models.schemas import (
    ChatRequest, ChatResponse,
    SearchRequest, SearchResponse,
    HealthResponse, Product,
    VoiceQueryResponse,
)
from backend.src.services.embeddings import embedding_service
from backend.src.services.rag import llm_manager, rag_query, search_products, _row_to_product
from backend.src.services.voice import voice_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check — reports component status."""
    return HealthResponse(
        status="ok" if embedding_service.is_loaded else "degraded",
        embedding_loaded=embedding_service.is_loaded,
        chroma_connected=False,  # Using in-memory for now
        llm_available=llm_manager.is_available,
        stt_available=voice_service.stt_available,
        tts_available=voice_service.tts_available,
        product_count=len(embedding_service.df) if embedding_service.df is not None else 0,
    )


@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Primary text flow: query → RAG → natural language response + product IDs."""
    if not embedding_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    result = rag_query(
        query=request.query_text,
        top_k=5,
        filters=request.filters,
        session_context=request.session_context,
    )

    return ChatResponse(
        response_text=result["response_text"],
        product_ids=result["product_ids"],
        products=result["products"],
        status=result["status"],
    )


@router.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Pure search: query → product list (no LLM generation)."""
    if not embedding_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        results = search_products(
            query=request.query_text,
            top_k=request.top_k,
            filters=request.filters,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=503, detail="Search pipeline error")

    products = [_row_to_product(row) for _, row in results.iterrows()]
    return SearchResponse(
        product_ids=[p.id for p in products],
        products=products,
        total=len(products),
    )


@router.post("/api/v1/voice/query", response_model=VoiceQueryResponse)
async def voice_query(
    audio: UploadFile = File(...),
    price_max: float = Form(None),
    category: str = Form(None),
):
    """Voice flow: audio → Whisper STT → RAG → TTS audio.

    Accepts multipart/form-data with audio file + optional filters.
    Returns transcript, RAG response text, products, and optional TTS audio.
    """
    if not embedding_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    if not voice_service.stt_available:
        raise HTTPException(status_code=503, detail="Whisper STT not available")

    # --- Step 1: STT (Whisper) ---
    audio_bytes = await audio.read()
    transcript, stt_time = voice_service.transcribe(audio_bytes)

    if not transcript:
        return VoiceQueryResponse(
            transcript="",
            response_text="I couldn't understand the audio. Please try again.",
            status="stt_failed",
        )

    logger.info(f"Voice query: '{transcript}' (STT {stt_time:.2f}s)")

    # --- Step 2: RAG pipeline (same as /api/v1/chat) ---
    from backend.src.models.schemas import Filters
    filters = Filters(price_max=price_max, category=category)

    result = rag_query(
        query=transcript,
        top_k=5,
        filters=filters,
    )

    # --- Step 3: TTS (gTTS) ---
    tts_audio = None
    if voice_service.tts_available and result.get("response_text"):
        tts_audio = voice_service.synthesize_base64(result["response_text"])

    return VoiceQueryResponse(
        transcript=transcript,
        response_text=result["response_text"],
        product_ids=result["product_ids"],
        products=result["products"],
        status=result["status"],
        audio_base64=tts_audio,
    )
