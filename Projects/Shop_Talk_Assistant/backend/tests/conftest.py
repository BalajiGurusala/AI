"""
Shared test fixtures for ShopTalk backend tests.

Provides synthetic product data, embedding indexes, and mock services
so tests can run without real ML models or data files.
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Synthetic product catalog ───────────────────────────────────────

NUM_PRODUCTS = 20
EMB_DIM_TEXT = 384
EMB_DIM_CLIP = 512

SAMPLE_PRODUCTS = [
    {"item_id": f"P{i:03d}", "item_name_flat": name,
     "product_type_flat": cat, "brand_flat": brand,
     "color_flat": color, "price": price,
     "bullet_point_flat": desc, "image_caption": caption,
     "enriched_text": f"{name} {desc}", "path": f"images/{i}.jpg"}
    for i, (name, cat, brand, color, price, desc, caption) in enumerate([
        ("Red Running Shoes", "SHOES", "Nike", "Red", 89.99,
         "Lightweight mesh running shoes with cushioned sole", "a pair of red athletic shoes"),
        ("Blue Denim Jacket", "OUTERWEAR", "Levi's", "Blue", 120.00,
         "Classic denim jacket with button closure", "a blue denim jacket on a mannequin"),
        ("Women's White Sneakers", "SHOES", "Adidas", "White", 79.99,
         "Comfortable everyday sneakers for women", "white sneakers on a wooden floor"),
        ("Black Leather Wallet", "ACCESSORIES", "Coach", "Black", 49.99,
         "Slim bifold wallet with card slots", "a black leather wallet"),
        ("Green Cotton T-Shirt", "T_SHIRT", "Uniqlo", "Green", 19.99,
         "Soft cotton crew neck t-shirt", "a green t-shirt folded neatly"),
        ("Men's Brown Boots", "SHOES", "Timberland", "Brown", 159.99,
         "Waterproof leather boots for men", "brown leather boots"),
        ("Pink Yoga Mat", "FITNESS", "Gaiam", "Pink", 29.99,
         "Non-slip yoga mat, 6mm thick", "a pink yoga mat rolled up"),
        ("Silver Watch", "WATCH", "Fossil", "Silver", 199.99,
         "Stainless steel analog watch", "a silver wristwatch"),
        ("Purple Backpack", "LUGGAGE", "JanSport", "Purple", 45.00,
         "Large capacity school backpack", "a purple backpack"),
        ("Yellow Table Lamp", "LIGHTING", "IKEA", "Yellow", 34.99,
         "Adjustable desk lamp with LED bulb", "a yellow desk lamp"),
        ("Navy Polo Shirt", "SHIRT", "Ralph Lauren", "Navy", 89.00,
         "Classic fit polo shirt for men", "a navy polo shirt"),
        ("Floral Pillow Cover", "HOME_BED_AND_BATH", "Threshold", "Multi", 15.99,
         "Decorative floral print pillow cover", "a colorful floral pillow"),
        ("Charcoal Ottoman", "OTTOMAN", "Wayfair", "Grey", 129.00,
         "Round storage ottoman with tufted top", "a grey ottoman in a living room"),
        ("Rose Gold Phone Case", "CELLULAR_PHONE_CASE", "OtterBox", "Rose Gold", 29.99,
         "Protective phone case for iPhone", "a rose gold phone case"),
        ("3D Printer PLA Filament", "THERMOPLASTIC_FILAMENT", "Hatchbox", "White", 24.99,
         "1.75mm PLA filament, 1kg spool", "a spool of white 3d printer filament"),
        ("Drawer Slide Hardware", "HARDWARE", "Liberty", "Silver", 12.99,
         "Ball bearing drawer slides, 18 inch", "metal drawer slides"),
        ("Kids Blue Sandals", "SHOES", "Crocs", "Blue", 34.99,
         "Comfortable kids sandals with strap", "blue kids sandals"),
        ("Coral Curtain Panel", "HOME_BED_AND_BATH", "Threshold", "Coral", 22.99,
         "Blackout curtain panel 84 inch", "a coral colored curtain"),
        ("Men's Black Dress Shoes", "SHOES", "Cole Haan", "Black", 149.99,
         "Oxford dress shoes with leather sole", "black oxford shoes"),
        ("Striped Beach Towel", "HOME_BED_AND_BATH", "Sun Squad", "Multi", 14.99,
         "Cotton beach towel with colorful stripes", "a striped beach towel"),
    ])
]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Synthetic product DataFrame matching ABO-flattened schema."""
    return pd.DataFrame(SAMPLE_PRODUCTS)


@pytest.fixture
def text_index() -> np.ndarray:
    """Random normalised text embedding index (NUM_PRODUCTS x 384)."""
    rng = np.random.default_rng(42)
    idx = rng.standard_normal((NUM_PRODUCTS, EMB_DIM_TEXT)).astype(np.float32)
    norms = np.linalg.norm(idx, axis=1, keepdims=True)
    return idx / np.where(norms == 0, 1, norms)


@pytest.fixture
def image_index() -> np.ndarray:
    """Random normalised CLIP embedding index (NUM_PRODUCTS x 512)."""
    rng = np.random.default_rng(99)
    idx = rng.standard_normal((NUM_PRODUCTS, EMB_DIM_CLIP)).astype(np.float32)
    norms = np.linalg.norm(idx, axis=1, keepdims=True)
    return idx / np.where(norms == 0, 1, norms)


@pytest.fixture
def encode_text_fn(text_index):
    """Mock text encoder: returns a fixed vector from the index."""
    def _encode(query: str) -> np.ndarray:
        rng = np.random.default_rng(hash(query) % 2**31)
        vec = rng.standard_normal((1, EMB_DIM_TEXT)).astype(np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.where(norms == 0, 1, norms)
    return _encode


@pytest.fixture
def encode_clip_fn(image_index):
    """Mock CLIP encoder: returns a fixed vector."""
    def _encode(query: str) -> np.ndarray:
        rng = np.random.default_rng(hash(query) % 2**31)
        vec = rng.standard_normal((1, EMB_DIM_CLIP)).astype(np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.where(norms == 0, 1, norms)
    return _encode
