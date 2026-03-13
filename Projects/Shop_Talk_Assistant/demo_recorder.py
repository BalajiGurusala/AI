"""
Automated video demo recorder for ShopTalk AI Assistant.

Records a browser session interacting with the Streamlit app,
producing a .webm video and per-scene screenshots.

Usage:
    pip install playwright
    playwright install chromium
    python demo_recorder.py [--url http://<EC2-IP>:8501]

Output:
    demo_recording/shoptalk_demo.webm
    demo_recording/*.png  (per-scene screenshots)
"""

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

DEFAULT_URL = "http://localhost:8501"
OUTPUT_DIR = Path("demo_recording")

DEMO_QUERIES = [
    {
        "query": "red shoes for women",
        "label": "text_search_red_shoes",
        "wait": 12,
        "description": "Basic text search demonstrating color + gender relevance",
    },
    {
        "query": "modern desk lamp for home office",
        "label": "cross_category_lamp",
        "wait": 12,
        "description": "Cross-category search showing hybrid text+image retrieval",
    },
    {
        "query": "sports shoes for men",
        "label": "sports_shoes_reranking",
        "wait": 12,
        "description": "Demonstrates type-miss and qualifier reranking penalties",
    },
    {
        "query": "do you have any in blue?",
        "label": "multi_turn_blue",
        "wait": 12,
        "description": "Multi-turn follow-up using session context",
    },
    {
        "query": "recommend a good backpack for travel",
        "label": "backpack_recommendation",
        "wait": 12,
        "description": "RAG recommendation with LLM-generated summary",
    },
]


async def wait_for_streamlit_ready(page, timeout_s=60):
    """Wait for Streamlit to finish loading (spinner gone, chat input visible)."""
    try:
        await page.wait_for_selector(
            'textarea[data-testid="stChatInputTextArea"]',
            state="visible",
            timeout=timeout_s * 1000,
        )
    except Exception:
        print("  ⚠ Timed out waiting for chat input — app may still be loading models")


async def wait_for_processing_done(page, timeout_s=45):
    """Wait until the processing status shows 'Done' or disappears."""
    await asyncio.sleep(2)
    try:
        await page.wait_for_function(
            """() => {
                const statuses = document.querySelectorAll('[data-testid="stStatusWidget"]');
                if (statuses.length === 0) return true;
                const last = statuses[statuses.length - 1];
                const text = last.textContent || '';
                return text.includes('Done') || text.includes('complete') || text.includes('Error');
            }""",
            timeout=timeout_s * 1000,
        )
    except Exception:
        print("  ⚠ Processing did not complete within timeout — continuing anyway")
    await asyncio.sleep(3)


async def submit_query(page, query: str):
    """Type a query into the Streamlit chat input and submit it."""
    chat_input = page.locator('textarea[data-testid="stChatInputTextArea"]')
    await chat_input.click()
    await chat_input.fill("")
    for char in query:
        await chat_input.type(char, delay=50)
    await asyncio.sleep(0.3)
    await chat_input.press("Enter")


async def screenshot(page, name: str, full_page=False):
    """Save a screenshot to the output directory."""
    path = OUTPUT_DIR / f"{name}.png"
    await page.screenshot(path=str(path), full_page=full_page)
    return path


async def main(base_url: str):
    OUTPUT_DIR.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(OUTPUT_DIR),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        # ── Scene 1: Load the application ────────────────────────────
        print("━" * 60)
        print("Scene 1 — Loading application")
        print("━" * 60)
        await page.goto(base_url, wait_until="networkidle", timeout=90_000)
        await wait_for_streamlit_ready(page)
        await asyncio.sleep(3)
        await screenshot(page, "01_app_loaded")
        print("  ✓ Application loaded")

        # Capture sidebar state
        await screenshot(page, "01_sidebar", full_page=False)

        # ── Scenes 2–6: Run demo queries ─────────────────────────────
        for i, scene in enumerate(DEMO_QUERIES, start=2):
            query = scene["query"]
            label = scene["label"]
            wait = scene["wait"]

            print(f"\nScene {i} — {scene['description']}")
            print(f"  Query: \"{query}\"")

            await submit_query(page, query)
            print(f"  → Submitted, waiting for results...")

            await wait_for_processing_done(page, timeout_s=wait + 20)

            # Screenshot: top of results (LLM response)
            await screenshot(page, f"{i:02d}_{label}_response")

            # Scroll to bottom to capture product cards
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            await screenshot(page, f"{i:02d}_{label}_cards", full_page=True)

            print(f"  ✓ Captured")

            # Scroll back up for next query
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(1)

        # ── Scene 7: Demonstrate sidebar filter change ────────────────
        print("\nScene 7 — Sidebar filter change")
        top_k_slider = page.locator('div[data-testid="stSlider"]').first
        if await top_k_slider.count() > 0:
            await top_k_slider.scroll_into_view_if_needed()
            await asyncio.sleep(1)
        await screenshot(page, "07_sidebar_filters")
        print("  ✓ Sidebar captured")

        # ── Scene 8: Backend health endpoint ──────────────────────────
        print("\nScene 8 — Backend health endpoint")
        backend_url = base_url.replace("8501", "8000")
        health_page = await context.new_page()
        try:
            await health_page.goto(
                f"{backend_url}/health",
                wait_until="networkidle",
                timeout=10_000,
            )
            await asyncio.sleep(2)
            await health_page.screenshot(
                path=str(OUTPUT_DIR / "08_health_endpoint.png")
            )
            print("  ✓ Health endpoint captured")
        except Exception as e:
            print(f"  ⚠ Could not reach backend: {e}")
        finally:
            await health_page.close()

        # ── Finish ────────────────────────────────────────────────────
        print("\nSaving video...")
        await page.close()
        await context.close()
        await browser.close()

    # Rename the auto-generated video file
    videos = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if videos:
        final = OUTPUT_DIR / "shoptalk_demo.webm"
        if final.exists():
            final.unlink()
        videos[-1].rename(final)
        print(f"\n{'━' * 60}")
        print(f"✅ Video:       {final}")
        print(f"📸 Screenshots: {OUTPUT_DIR}/")
        print(f"{'━' * 60}")
        print(f"\nConvert to MP4:")
        print(f"  ffmpeg -i {final} -c:v libx264 -crf 23 {OUTPUT_DIR}/shoptalk_demo.mp4")
    else:
        print("⚠ No video file found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record ShopTalk demo video")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Streamlit app URL (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.url))
