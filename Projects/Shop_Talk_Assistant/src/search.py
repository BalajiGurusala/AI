"""
ShopTalk – Hybrid Search Pipeline (Shared Module)

Two-stage hybrid search: vector retrieval + production reranking.
Used by NB03, NB04, NB05, and the Streamlit app.

This module is the SINGLE SOURCE OF TRUTH for search logic —
no more copy-pasting between notebooks.
"""

import re
from typing import Optional, Dict, Set

import numpy as np
import pandas as pd


# ======================================================================
# Constants & Configuration
# ======================================================================

ALPHA_DEFAULT = 0.6

# Rerank weights (tuned in NB03)
LEXICAL_WEIGHT = 0.16
TITLE_WEIGHT = 0.10
TYPE_WEIGHT = 0.06
HEAD_NOUN_MISS_PENALTY = 0.50   # multiplicative
GENDER_MISS_PENALTY = 0.40      # multiplicative
COLOR_MISS_PENALTY  = 0.45      # multiplicative — product has a clearly different color
TYPE_MISS_PENALTY   = 0.45      # multiplicative — product_type doesn't match query intent
QUALIFIER_MISS_PENALTY = 0.40   # multiplicative — qualifying terms absent from title
LOW_CONFIDENCE_CUTOFF = 0.30

# Color families: query token → canonical family, family → all member tokens
# A product is penalized when its color field contains tokens from a DIFFERENT
# family than the one the user requested.
COLOR_FAMILIES: Dict[str, Set[str]] = {
    "red":    {"red", "crimson", "scarlet", "burgundy", "ruby", "cherry",
               "mandarina", "cardinal", "garnet", "wine", "bordeaux"},
    "blue":   {"blue", "navy", "cobalt", "azure", "royal", "indigo", "denim",
               "sapphire", "teal", "turquoise", "aqua"},
    "green":  {"green", "olive", "emerald", "sage", "lime", "mint", "forest",
               "khaki", "hunter", "moss"},
    "black":  {"black", "ebony", "charcoal", "onyx", "jet"},
    "white":  {"white", "ivory", "cream", "pearl", "snow", "vanilla"},
    "brown":  {"brown", "tan", "beige", "caramel", "chocolate", "taupe",
               "cognac", "mocha", "chestnut", "sand"},
    "pink":   {"pink", "rose", "blush", "magenta", "fuchsia", "flamingo",
               "bubblegum", "mauve"},
    "purple": {"purple", "violet", "lavender", "lilac", "plum", "grape",
               "orchid", "amethyst"},
    "yellow": {"yellow", "gold", "amber", "mustard", "lemon", "canary"},
    "orange": {"orange", "coral", "peach", "rust", "terracotta", "apricot",
               "pumpkin"},
    "grey":   {"grey", "gray", "silver", "ash", "slate", "pewter"},
}

QUERY_STOPWORDS = {
    "for", "with", "and", "the", "a", "an", "in", "on", "to", "of", "by",
    "from", "best", "good", "new", "comfortable", "great", "nice", "high",
    "quality", "s", "under", "below", "above", "less", "than", "more",
    "about", "show", "me", "find", "get", "want", "need", "looking", "search",
}

FEMALE_TOKENS = {
    "women", "woman", "female", "ladies", "lady", "girls", "girl", "womens",
}
MALE_TOKENS = {
    "men", "man", "male", "mens", "boys", "boy",
}

ABO_CATEGORY_HINTS: Dict[str, Set[str]] = {
    "shoe": {"SHOES"}, "shoes": {"SHOES"},
    "sneaker": {"SHOES"}, "sneakers": {"SHOES"},
    "boot": {"SHOES", "BOOT"}, "boots": {"SHOES", "BOOT"},
    "sandal": {"SHOES", "SANDAL"}, "sandals": {"SHOES", "SANDAL"},
    "filament": {"THERMOPLASTIC_FILAMENT", "MECHANICAL_COMPONENTS"},
    "pla": {"THERMOPLASTIC_FILAMENT"}, "abs": {"THERMOPLASTIC_FILAMENT"},
    "3d": {"THERMOPLASTIC_FILAMENT"}, "printer": {"THERMOPLASTIC_FILAMENT"},
    "phone": {"CELLULAR_PHONE_CASE"}, "case": {"CELLULAR_PHONE_CASE"},
    "cover": {"CELLULAR_PHONE_CASE"},
    "drawer": {"HARDWARE"}, "slides": {"HARDWARE"}, "slide": {"HARDWARE"},
    "handle": {"HARDWARE_HANDLE"},
    "hardware": {"HARDWARE", "HARDWARE_HANDLE"},
    "shirt": {"SHIRT", "T_SHIRT"}, "tshirt": {"SHIRT", "T_SHIRT"},
    "t-shirt": {"SHIRT", "T_SHIRT"}, "polo": {"SHIRT"},
    "pillow": {"HOME_BED_AND_BATH", "HOME"},
    "sheet": {"HOME_BED_AND_BATH"}, "curtain": {"HOME_BED_AND_BATH"},
    "towel": {"HOME_BED_AND_BATH"},
    "ottoman": {"OTTOMAN", "FURNITURE"}, "chair": {"CHAIR", "FURNITURE"},
    "table": {"TABLE", "FURNITURE"}, "lamp": {"LIGHTING", "LAMP"},
    "watch": {"WATCH"}, "backpack": {"LUGGAGE"},
}

VISUAL_CUES = {
    "colorful", "patterned", "floral", "striped", "printed", "design",
    "aesthetic", "stylish", "cute", "pretty", "beautiful",
}
TECHNICAL_CUES = {
    "inch", "mm", "kg", "watt", "volt", "capacity", "specs",
    "compatible", "mount", "gauge", "thread", "count",
}


# ======================================================================
# Helper Functions
# ======================================================================

def tokenize(text: str) -> list:
    """Lowercase tokenization, strip punctuation, remove stopwords."""
    text = re.sub(r"[^a-zA-Z0-9\-\s]", " ", str(text).lower())
    return [t for t in text.split() if t and t not in QUERY_STOPWORDS]


def field_overlap(query: str, text: str) -> float:
    """Fraction of query tokens found in text."""
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    return len(q_tokens & set(tokenize(text))) / max(1, len(q_tokens))


def expand_compound_tokens(tokens: set) -> set:
    """Expand hyphenated tokens (e.g. t-shirt -> {t-shirt, shirt})."""
    expanded = set(tokens)
    for tok in tokens:
        if "-" in tok:
            for part in tok.split("-"):
                if part and part in ABO_CATEGORY_HINTS:
                    expanded.add(part)
    return expanded


def extract_head_nouns(query: str) -> set:
    """Query tokens that match ABO category vocabulary."""
    direct = {t for t in tokenize(query) if t in ABO_CATEGORY_HINTS}
    return expand_compound_tokens(direct)


def infer_expected_types(query: str) -> set:
    """Map query tokens to expected ABO product_type values."""
    expected = set()
    for t in tokenize(query):
        expected |= ABO_CATEGORY_HINTS.get(t, set())
    return expected


def extract_gender_intent(query: str) -> Optional[str]:
    """Detect gender intent from query tokens."""
    q_tokens = set(tokenize(query))
    female_hit = bool(q_tokens & FEMALE_TOKENS)
    male_hit = bool(q_tokens & MALE_TOKENS)
    if female_hit and not male_hit:
        return "female"
    if male_hit and not female_hit:
        return "male"
    return None


def extract_color_intent(query: str) -> Optional[str]:
    """Return the canonical color family name if the query mentions a color.

    Returns e.g. 'red', 'blue', 'black', or None if no color term found.
    Checks both direct family names and all synonym tokens.
    """
    q_tokens = set(tokenize(query))
    for family, synonyms in COLOR_FAMILIES.items():
        if q_tokens & synonyms:
            return family
    return None


def extract_qualifier_tokens(query: str) -> set:
    """Return query tokens that are not stopwords, head nouns, gender, or color.

    These are the descriptive/qualifying terms that narrow intent within a
    category (e.g. 'sports' in 'sports shoes', 'cotton' in 'cotton shirt').
    """
    q_tokens = set(tokenize(query))
    head = extract_head_nouns(query)
    color_tok: set = set()
    for _syns in COLOR_FAMILIES.values():
        color_tok |= _syns
    return q_tokens - head - MALE_TOKENS - FEMALE_TOKENS - color_tok


def compute_dynamic_alpha(query: str) -> float:
    """Shift alpha: visual queries lower it, technical queries raise it."""
    q_tokens = set(tokenize(query))
    visual = len(q_tokens & VISUAL_CUES)
    technical = len(q_tokens & TECHNICAL_CUES)
    has_nums = bool(re.search(r"\d", query))

    alpha = ALPHA_DEFAULT
    if visual > 0:
        alpha -= 0.15 * min(visual, 2)
    if technical > 0 or has_nums:
        alpha += 0.10 * min(technical + int(has_nums), 2)
    return max(0.2, min(0.9, alpha))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return x / norms


# ======================================================================
# Stage 2: Production Reranker
# ======================================================================

def apply_rerank(
    results: pd.DataFrame,
    query: str,
    top_k: int,
) -> pd.DataFrame:
    """Generic production reranker.

    Penalties applied (all multiplicative):
      - head-noun miss   : product text lacks the queried category term
      - gender miss       : product targets opposite gender
      - color miss        : product explicitly states a different color
      - type miss         : product_type doesn't match expected types
      - qualifier miss    : qualifying terms absent from product *title*

    Operates on any DataFrame that has 'hybrid_score', 'item_name_flat',
    'enriched_text', 'color_flat', and 'product_type_flat' columns.
    """
    if results.empty:
        return results

    head_nouns = extract_head_nouns(query)
    expected_types = infer_expected_types(query)
    gender_intent = extract_gender_intent(query)
    color_intent = extract_color_intent(query)
    qualifiers = extract_qualifier_tokens(query)
    target_color_tokens = COLOR_FAMILIES.get(color_intent, set()) if color_intent else set()
    other_color_tokens: Set[str] = set()
    if color_intent:
        for fam, synonyms in COLOR_FAMILIES.items():
            if fam != color_intent:
                other_color_tokens |= synonyms

    adjusted = results.copy()

    # Additive features
    adjusted["lexical_overlap"] = adjusted.apply(
        lambda r: field_overlap(query, " ".join([
            str(r.get("item_name_flat", "")),
            str(r.get("enriched_text", "")),
            str(r.get("product_type_flat", "")),
        ])), axis=1,
    )
    adjusted["title_overlap"] = adjusted.apply(
        lambda r: field_overlap(query, str(r.get("item_name_flat", ""))), axis=1,
    )
    adjusted["type_overlap"] = adjusted.apply(
        lambda r: field_overlap(query, str(r.get("product_type_flat", ""))), axis=1,
    )

    adjusted["final_score"] = (
        adjusted["hybrid_score"]
        + LEXICAL_WEIGHT * adjusted["lexical_overlap"]
        + TITLE_WEIGHT * adjusted["title_overlap"]
        + TYPE_WEIGHT * adjusted["type_overlap"]
    )

    # Multiplicative penalties
    hn_list, gn_list, color_list, type_list, qual_list = [], [], [], [], []
    for _, row in adjusted.iterrows():
        hay = " ".join([
            str(row.get("item_name_flat", "")),
            str(row.get("enriched_text", "")),
            str(row.get("product_type_flat", "")),
        ]).lower()
        hay_tokens = set(tokenize(hay))
        hay_expanded = expand_compound_tokens(hay_tokens)

        title_tokens = set(tokenize(str(row.get("item_name_flat", "")).lower()))
        ptype = str(row.get("product_type_flat", "")).upper()

        # Head-noun penalty
        hn_mult = 1.0
        if head_nouns and not (head_nouns & hay_expanded):
            if not (expected_types and any(t in ptype for t in expected_types)):
                hn_mult = 1.0 - HEAD_NOUN_MISS_PENALTY

        # Gender penalty
        gn_mult = 1.0
        if gender_intent == "female" and (hay_tokens & MALE_TOKENS) and not (hay_tokens & FEMALE_TOKENS):
            gn_mult = 1.0 - GENDER_MISS_PENALTY
        elif gender_intent == "male" and (hay_tokens & FEMALE_TOKENS) and not (hay_tokens & MALE_TOKENS):
            gn_mult = 1.0 - GENDER_MISS_PENALTY

        # Color penalty
        color_mult = 1.0
        if color_intent:
            product_color = " ".join([
                str(row.get("color_flat", "")),
                str(row.get("item_name_flat", "")),
            ]).lower()
            color_tokens = set(tokenize(product_color))
            has_target  = bool(color_tokens & target_color_tokens)
            has_other   = bool(color_tokens & other_color_tokens)
            if has_other and not has_target:
                color_mult = 1.0 - COLOR_MISS_PENALTY

        # Product-type mismatch penalty (independent of head-noun text check).
        # Fires when the product's type doesn't match any expected type AND
        # the query doesn't explicitly mention the product's actual type.
        type_mult = 1.0
        if expected_types and ptype:
            if not any(t in ptype for t in expected_types):
                if ptype not in expected_types:
                    type_mult = 1.0 - TYPE_MISS_PENALTY

        # Qualifier miss penalty: descriptive terms (e.g. 'sports', 'cotton',
        # 'leather') must appear in the product TITLE (not enriched_text which
        # can include misleading image captions).
        qual_mult = 1.0
        if qualifiers and not (qualifiers & title_tokens):
            qual_mult = 1.0 - QUALIFIER_MISS_PENALTY

        hn_list.append(hn_mult)
        gn_list.append(gn_mult)
        color_list.append(color_mult)
        type_list.append(type_mult)
        qual_list.append(qual_mult)

    adjusted["hn_mult"] = hn_list
    adjusted["gn_mult"] = gn_list
    adjusted["color_mult"] = color_list
    adjusted["type_mult"] = type_list
    adjusted["qual_mult"] = qual_list
    adjusted["final_score"] *= (
        adjusted["hn_mult"] * adjusted["gn_mult"] * adjusted["color_mult"]
        * adjusted["type_mult"] * adjusted["qual_mult"]
    )

    adjusted = adjusted.sort_values("final_score", ascending=False).reset_index(drop=True)
    strong = adjusted[adjusted["final_score"] >= LOW_CONFIDENCE_CUTOFF]
    out = strong.head(top_k).copy() if len(strong) >= top_k else adjusted.head(top_k).copy()
    out["hybrid_score"] = out["final_score"]
    out["_rank"] = np.arange(1, len(out) + 1)
    return out


# ======================================================================
# Stage 1: Vector Retrieval Backends
# ======================================================================

def retrieve_inmemory(
    q_text: np.ndarray,
    q_clip: np.ndarray,
    text_index: np.ndarray,
    image_index: np.ndarray,
    df: pd.DataFrame,
    alpha: float,
    n_fetch: int,
) -> pd.DataFrame:
    """In-memory NumPy dot-product retrieval (fast, no external DB)."""
    text_sim = (text_index @ q_text.T).squeeze()
    image_sim = (image_index @ q_clip.T).squeeze()
    scores = alpha * text_sim + (1.0 - alpha) * image_sim

    top_indices = np.argsort(scores)[::-1][:n_fetch]
    results = df.iloc[top_indices].copy()
    results["text_sim"] = text_sim[top_indices]
    results["image_sim"] = image_sim[top_indices]
    results["hybrid_score"] = scores[top_indices]
    return results.reset_index(drop=True)


def retrieve_chroma(
    q_text: np.ndarray,
    q_clip: np.ndarray,
    text_collection,
    image_collection,
    df: pd.DataFrame,
    item_id_to_idx: dict,
    alpha: float,
    n_fetch: int,
) -> pd.DataFrame:
    """ChromaDB-backed retrieval (persistent, production-ready)."""
    text_res = text_collection.query(
        query_embeddings=q_text.tolist(), n_results=n_fetch, include=["distances"],
    )
    image_res = image_collection.query(
        query_embeddings=q_clip.tolist(), n_results=n_fetch, include=["distances"],
    )

    score_map = {}
    for pid, dist in zip(text_res["ids"][0], text_res["distances"][0]):
        score_map.setdefault(pid, {"text_sim": 0.0, "image_sim": 0.0})
        score_map[pid]["text_sim"] = 1.0 - float(dist)
    for pid, dist in zip(image_res["ids"][0], image_res["distances"][0]):
        score_map.setdefault(pid, {"text_sim": 0.0, "image_sim": 0.0})
        score_map[pid]["image_sim"] = 1.0 - float(dist)

    if not score_map:
        return pd.DataFrame(columns=list(df.columns) + ["text_sim", "image_sim", "hybrid_score"])

    scored = [
        (pid, p["text_sim"], p["image_sim"],
         alpha * p["text_sim"] + (1.0 - alpha) * p["image_sim"])
        for pid, p in score_map.items()
    ]
    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[:n_fetch]

    top_idx = [item_id_to_idx[pid] for pid, *_ in top if pid in item_id_to_idx]
    results = df.iloc[top_idx].copy()
    lookup = {pid: (ts, ims, hs) for pid, ts, ims, hs in top}
    results["text_sim"] = results["item_id"].astype(str).map(lambda x: lookup.get(x, (0, 0, 0))[0])
    results["image_sim"] = results["item_id"].astype(str).map(lambda x: lookup.get(x, (0, 0, 0))[1])
    results["hybrid_score"] = results["item_id"].astype(str).map(lambda x: lookup.get(x, (0, 0, 0))[2])
    return results.reset_index(drop=True)


# ======================================================================
# Main Search Entry Point
# ======================================================================

def hybrid_search(
    query: str,
    df: pd.DataFrame,
    text_index: np.ndarray,
    image_index: np.ndarray,
    encode_text_fn,
    encode_clip_fn,
    top_k: int = 5,
    alpha: float = None,
    price_max: float = None,
    category: str = None,
    text_collection=None,
    image_collection=None,
    item_id_to_idx: dict = None,
    rerank: bool = True,
) -> pd.DataFrame:
    """Production hybrid search.

    Uses ChromaDB if collections are provided, otherwise falls back to
    in-memory NumPy retrieval.

    Args:
        query: User search query.
        df: Product DataFrame.
        text_index: Pre-normalised text embeddings (n_products x dim).
        image_index: Pre-normalised image embeddings (n_products x dim).
        encode_text_fn: Callable(str) -> np.ndarray (1 x text_dim).
        encode_clip_fn: Callable(str) -> np.ndarray (1 x clip_dim).
        top_k: Number of results to return.
        alpha: Text/image weight. None = dynamic.
        price_max: Optional price filter.
        category: Optional category filter.
        text_collection: ChromaDB text collection (optional).
        image_collection: ChromaDB image collection (optional).
        item_id_to_idx: Mapping item_id -> df index (for ChromaDB).
        rerank: Whether to apply Stage-2 reranking.

    Returns:
        DataFrame of top-K products with hybrid_score and _rank columns.
    """
    if alpha is None:
        alpha = compute_dynamic_alpha(query) if rerank else ALPHA_DEFAULT

    # Encode query
    q_text = encode_text_fn(query)
    q_clip = encode_clip_fn(query)
    n_fetch = min(len(df), max(top_k * 12, 80))

    # Stage 1: Retrieve
    if text_collection is not None and image_collection is not None and item_id_to_idx:
        results = retrieve_chroma(
            q_text, q_clip, text_collection, image_collection,
            df, item_id_to_idx, alpha, n_fetch,
        )
    else:
        results = retrieve_inmemory(
            q_text, q_clip, text_index, image_index, df, alpha, n_fetch,
        )

    # Metadata filters
    if price_max is not None and "price" in results.columns:
        results = results[results["price"] <= price_max].reset_index(drop=True)
    if category and "product_type_flat" in results.columns:
        mask = results["product_type_flat"].str.upper().str.contains(
            category.upper(), na=False
        )
        results = results[mask].reset_index(drop=True)

    if results.empty:
        return results

    results.attrs["alpha_used"] = alpha

    # Stage 2: Rerank
    if rerank:
        results = apply_rerank(results, query=query, top_k=top_k)
    else:
        results = results.sort_values("hybrid_score", ascending=False).head(top_k).copy()
        results["_rank"] = np.arange(1, len(results) + 1)

    results.attrs["alpha_used"] = alpha
    return results
