# ShopTalk AI Assistant — Video Demo Script

> A step-by-step guide for recording a working video walkthrough of the ShopTalk application on EC2.

---

## Table of Contents

- [Part 1 — Recording Setup](#part-1--recording-setup)
  - [Option A: Record from Your Local Browser (Recommended)](#option-a-record-from-your-local-browser-recommended)
  - [Option B: Automated Browser Recording with Playwright](#option-b-automated-browser-recording-with-playwright)
- [Part 2 — Pre-Demo Checklist](#part-2--pre-demo-checklist)
- [Part 3 — Demo Script (Scene by Scene)](#part-3--demo-script-scene-by-scene)
  - [Scene 1: Application Overview (1 min)](#scene-1-application-overview-1-min)
  - [Scene 2: Text Search — Basic Query (2 min)](#scene-2-text-search--basic-query-2-min)
  - [Scene 3: Hybrid Search — Image + Text (2 min)](#scene-3-hybrid-search--image--text-2-min)
  - [Scene 4: Multi-Turn Conversation (1.5 min)](#scene-4-multi-turn-conversation-15-min)
  - [Scene 5: Filters and Category Browsing (1.5 min)](#scene-5-filters-and-category-browsing-15-min)
  - [Scene 6: Voice Input (1.5 min)](#scene-6-voice-input-15-min)
  - [Scene 7: Text-to-Speech Output (1 min)](#scene-7-text-to-speech-output-1-min)
  - [Scene 8: LLM Provider Switching (1 min)](#scene-8-llm-provider-switching-1-min)
  - [Scene 9: Backend Health & Architecture (1 min)](#scene-9-backend-health--architecture-1-min)
  - [Scene 10: Wrap-Up (0.5 min)](#scene-10-wrap-up-05-min)
- [Part 4 — Playwright Automation Script](#part-4--playwright-automation-script)
- [Part 5 — Post-Recording Tips](#part-5--post-recording-tips)

---

## Part 1 — Recording Setup

### Option A: Record from Your Local Browser (Recommended)

Access the Streamlit app from your local machine's browser and use a screen recorder. This gives the best quality and lets you use voice narration.

**macOS (QuickTime):**
```bash
# 1. Open QuickTime Player
# 2. File → New Screen Recording
# 3. Click the dropdown arrow next to the Record button
#    - Select your microphone for voice narration
#    - Choose "Record Selected Portion" to capture just the browser window
# 4. Open Chrome/Safari → navigate to http://<EC2-PUBLIC-IP>:8501
# 5. Click Record, perform the demo, then click Stop
# 6. File → Export As → 1080p (mov)
```

**macOS (OBS Studio — free, more control):**
```bash
# Install OBS
brew install --cask obs

# Launch OBS, configure:
# - Sources: Add "Window Capture" → select your browser window
# - Audio: Enable desktop audio + microphone
# - Output: Settings → Output → Recording Path, Format: mp4
# - Video: 1920x1080, 30fps
# Record → perform demo → Stop
```

**Chrome built-in (simplest, no install):**
```
# Chrome DevTools has a built-in recorder (no audio):
# 1. Open http://<EC2-PUBLIC-IP>:8501
# 2. Cmd+Shift+P → "Show Recorder" → Start recording
# Note: no audio — use for quick visual demos only
```

### Option B: Automated Browser Recording with Playwright

Run headless Playwright on EC2 to produce a video programmatically. Good for repeatable, scripted demos without manual interaction.

```bash
# On EC2
pip install playwright
playwright install chromium

# Run the automated demo script (see Part 4 below)
python demo_recorder.py
```

---

## Part 2 — Pre-Demo Checklist

Run these checks on your EC2 instance **before** starting the recording.

```bash
# 1. Verify all containers are running
docker compose ps
# Expected: backend, frontend, ollama all "Up (healthy)" or "Up"

# 2. Verify backend health
curl -s http://localhost:8000/health | python3 -m json.tool
# Expected: {"status":"ok","product_count":9190,"llm":"ollama/llama3.2",...}

# 3. Verify Streamlit is accessible
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501
# Expected: 200

# 4. Verify Ollama model is pulled
docker exec shoptalk-ollama ollama list
# Expected: llama3.2 listed

# 5. (From local machine) Verify external access
curl -s -o /dev/null -w "%{http_code}" http://<EC2-PUBLIC-IP>:8501
# Expected: 200

# 6. Clear any previous chat state (optional — just refresh the browser page)
```

**Browser prep:**
- Use Chrome or Firefox in a clean window (no distracting bookmarks bar)
- Set zoom to 100% (Cmd+0)
- Window size: ~1400x900 or full screen on a 1080p display
- Close all other tabs

---

## Part 3 — Demo Script (Scene by Scene)

Total estimated time: **~12 minutes**

---

### Scene 1: Application Overview (1 min)

**Action:** Open `http://<EC2-PUBLIC-IP>:8501` in browser. Wait for full load.

**Show:**
- The ShopTalk AI Assistant header with the tagline
- The left sidebar with settings, filters, and catalog stats
- The clean chat interface with the text input box at the bottom

**Talking Points:**
> "This is ShopTalk, an AI-powered shopping assistant built with a Retrieval-Augmented Generation pipeline. The application uses a hybrid search engine combining text embeddings and image embeddings to find products, then generates natural language recommendations using an LLM."
>
> "On the left sidebar, you can see the settings panel — the LLM model selector, category filter, a slider to control how many results to show, and catalog statistics. The system currently indexes over 9,000 products from the Amazon Berkeley Objects dataset."
>
> "Let me show you how it works."

---

### Scene 2: Text Search — Basic Query (2 min)

**Action:** Type in the chat input box.

**Query 1:** `red shoes for women`

**Wait for:**
1. The "Processing your query..." status to expand
2. "Searching product catalog..." message
3. "Found X products" message
4. "Generating recommendation..." message
5. The LLM response text to appear
6. Product cards with images to render below

**Show:**
- The streaming status updates during processing
- The natural-language recommendation from the LLM
- The product cards with images, titles, categories, colors, brands
- Hover over a card to show the subtle elevation effect

**Talking Points:**
> "I'll search for 'red shoes for women'. The system performs a hybrid search — it encodes this query with both a fine-tuned SentenceTransformer model and a CLIP vision-language model, then combines text similarity and image similarity scores."
>
> "The results go through a reranking pipeline that applies penalties for gender mismatch, color mismatch, and product type mismatch. You can see the results are highly relevant — all red, all women's shoes."
>
> "The LLM — in this case Ollama running Llama 3.2 locally — takes these search results as context and generates a conversational recommendation, highlighting key features of each product."

---

### Scene 3: Hybrid Search — Image + Text (2 min)

**Action:** Type another query to demonstrate different product categories.

**Query 2:** `modern desk lamp for home office`

**Wait for results, then:**

**Query 3:** `sports shoes for men`

**Talking Points:**
> "Let me try a completely different category — 'modern desk lamp for home office'. Notice how the hybrid search handles this well, finding lamps that match both the text description and visual characteristics."
>
> "Now let me search for 'sports shoes for men'. The reranking engine specifically penalizes results that don't match the qualifier 'sports' — so you won't see leather dress shoes mixed in with athletic footwear."

---

### Scene 4: Multi-Turn Conversation (1.5 min)

**Action:** Follow up on the previous query without changing context.

**Query 4:** `do you have any in blue?`

**Talking Points:**
> "One powerful feature is multi-turn conversation. I can follow up with 'do you have any in blue?' — the LLM remembers the previous context was about sports shoes for men, and the search automatically refines."
>
> "This session context is maintained throughout the conversation, making it feel like a natural shopping experience."

**Query 5:** `what about under 50 dollars?`

> "The assistant maintains context across turns, providing a conversational shopping experience rather than isolated search queries."

---

### Scene 5: Filters and Category Browsing (1.5 min)

**Action:** Use the sidebar filters.

1. Click the **Category** dropdown → select a specific category (e.g., "SHIRT" or "WATCH")
2. Change **Results to show** slider from 5 → 3
3. Type a new query: `best options available`

**Talking Points:**
> "The sidebar provides additional filtering. I can narrow down to a specific category — let me select 'WATCH' — and reduce the results to 3."
>
> "Now when I search, the results are filtered to only watches. This is useful for browsing within a known category."

4. Reset category back to "All Categories"
5. Set results back to 5

---

### Scene 6: Voice Input (1.5 min)

**Action:**
1. Toggle **"Enable voice input"** ON in the sidebar
2. The microphone widget appears in the sidebar
3. Click the mic icon → speak: "show me leather bags"
4. Wait for transcription → search results

**Talking Points:**
> "ShopTalk supports voice input. I'll enable it from the sidebar — you can see the microphone widget appear."
>
> "I'll click the mic and say 'show me leather bags'."
>
> *[Record audio]*
>
> "The speech is transcribed using Whisper — you can see the transcription appear, and then the same hybrid search pipeline processes the query. Voice and text queries go through the exact same RAG pipeline."

---

### Scene 7: Text-to-Speech Output (1 min)

**Action:**
1. Toggle **"Read responses aloud"** ON in the sidebar
2. Type a query: `recommend a good backpack for travel`
3. Wait for the audio player to appear and play the response

**Talking Points:**
> "I'll enable text-to-speech. Now when the assistant responds, it also reads the answer aloud."
>
> *[Let audio play for a few seconds]*
>
> "This uses platform-native speech synthesis for fast, offline text-to-speech. Combined with voice input, this creates a fully voice-driven shopping experience."

4. Toggle TTS and Voice Input OFF (reset for cleaner recording)

---

### Scene 8: LLM Provider Switching (1 min)

**Action:** (Only applicable in standalone mode; on EC2 with Docker the backend handles LLM selection.)

*If running standalone (Option A), show the LLM dropdown:*
1. Show the **LLM Model** dropdown in the sidebar
2. Point out available options: Ollama (local/free), OpenAI, Groq

**Talking Points:**
> "The system supports multiple LLM backends. On this EC2 deployment, we're using Ollama with Llama 3.2, which runs entirely locally — no API costs."
>
> "In standalone mode, you can also switch to OpenAI GPT-4o-mini or Groq's Llama-3.3-70B through the sidebar dropdown, depending on your needs for quality vs. cost vs. latency."

*If using Docker backend, show the health endpoint instead:*

---

### Scene 9: Backend Health & Architecture (1 min)

**Action:** Open a new browser tab → navigate to `http://<EC2-PUBLIC-IP>:8000/health`

**Show:** The JSON health response with product count, LLM info, device.

**Talking Points:**
> "Under the hood, this runs as a microservice architecture. Let me show the backend health endpoint."
>
> "The FastAPI backend reports the product count, active LLM, device type, and model loading status. The backend handles embedding, hybrid search, RAG generation, and voice processing."
>
> "The fine-tuned SentenceTransformer — trained with triplet loss on product data — is loaded automatically. You can see it reported here."

**Action:** Switch back to the Streamlit tab.

---

### Scene 10: Wrap-Up (0.5 min)

**Action:** Click **"Clear Chat"** in the sidebar to reset.

**Talking Points:**
> "To summarize, ShopTalk AI Assistant combines:
> - Hybrid retrieval using fine-tuned text and CLIP image embeddings
> - A genre-aware reranking pipeline for relevance
> - RAG-based natural language generation
> - Multi-turn conversational context
> - Voice input and text-to-speech
> - Multiple LLM provider support
>
> All deployed as containerized microservices on EC2 with Docker Compose. Thank you for watching."

---

## Part 4 — Playwright Automation Script

Save this as `demo_recorder.py` in the project root. It automates browser interactions and records a video.

```python
"""
Automated video demo recorder for ShopTalk AI Assistant.

Usage:
    pip install playwright
    playwright install chromium
    python demo_recorder.py

Output:
    demo_recording/shoptalk_demo.webm
"""

import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright

EC2_URL = "http://localhost:8501"  # Change to EC2 public IP if running remotely
OUTPUT_DIR = Path("demo_recording")
OUTPUT_DIR.mkdir(exist_ok=True)

QUERIES = [
    ("red shoes for women", 8),
    ("modern desk lamp for home office", 8),
    ("sports shoes for men", 8),
    ("do you have any in blue?", 8),
    ("recommend a good backpack for travel", 8),
]


async def wait_for_processing(page, timeout_s=30):
    """Wait for Streamlit to finish processing (status widget disappears or completes)."""
    await asyncio.sleep(2)
    try:
        await page.wait_for_function(
            """() => {
                const statuses = document.querySelectorAll('[data-testid="stStatusWidget"]');
                if (statuses.length === 0) return true;
                const last = statuses[statuses.length - 1];
                return last.textContent.includes('Done') || last.textContent.includes('complete');
            }""",
            timeout=timeout_s * 1000,
        )
    except Exception:
        pass
    await asyncio.sleep(2)


async def slow_type(page, selector, text, delay_ms=80):
    """Type text character by character for a realistic demo effect."""
    element = page.locator(selector)
    await element.click()
    for char in text:
        await element.type(char, delay=delay_ms)


async def main():
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

        # ---- Scene 1: Load the app ----
        print("[Scene 1] Loading application...")
        await page.goto(EC2_URL, wait_until="networkidle", timeout=60_000)
        await asyncio.sleep(5)  # Let models load

        # Take a screenshot of initial state
        await page.screenshot(path=str(OUTPUT_DIR / "01_initial.png"))
        print("  ✓ App loaded")

        # ---- Scenes 2–5: Run queries ----
        for i, (query, wait_s) in enumerate(QUERIES, start=2):
            print(f"[Scene {i}] Query: '{query}'")

            chat_input = page.locator('textarea[data-testid="stChatInputTextArea"]')
            await chat_input.click()
            await chat_input.fill("")

            # Type the query with realistic speed
            for char in query:
                await chat_input.type(char, delay=60)
            await asyncio.sleep(0.5)

            # Press Enter to submit
            await chat_input.press("Enter")
            print(f"  → Submitted, waiting {wait_s}s...")

            await wait_for_processing(page, timeout_s=wait_s + 15)
            await asyncio.sleep(3)  # Extra buffer for rendering

            # Screenshot
            await page.screenshot(
                path=str(OUTPUT_DIR / f"{i:02d}_{query[:20].replace(' ', '_')}.png"),
                full_page=True,
            )
            print(f"  ✓ Results captured")

            # Scroll down to show product cards
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            await page.screenshot(
                path=str(OUTPUT_DIR / f"{i:02d}_{query[:20].replace(' ', '_')}_cards.png"),
                full_page=True,
            )

            # Scroll back to top for next query
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(1)

        # ---- Scene: Show sidebar filters ----
        print("[Scene] Demonstrating sidebar...")
        sidebar = page.locator('[data-testid="stSidebar"]')
        await sidebar.scroll_into_view_if_needed()
        await asyncio.sleep(2)
        await page.screenshot(path=str(OUTPUT_DIR / "sidebar.png"))

        # ---- Scene: Backend health endpoint ----
        print("[Scene] Backend health check...")
        health_page = await context.new_page()
        await health_page.goto(
            EC2_URL.replace("8501", "8000") + "/health",
            wait_until="networkidle",
        )
        await asyncio.sleep(2)
        await health_page.screenshot(path=str(OUTPUT_DIR / "health_endpoint.png"))
        await health_page.close()

        # ---- Cleanup ----
        print("[Done] Saving video...")
        await page.close()
        await context.close()
        await browser.close()

    # Rename video
    videos = list(OUTPUT_DIR.glob("*.webm"))
    if videos:
        final_path = OUTPUT_DIR / "shoptalk_demo.webm"
        videos[0].rename(final_path)
        print(f"\n✅ Video saved: {final_path}")
        print(f"📸 Screenshots saved in: {OUTPUT_DIR}/")

    print("\nTo convert to mp4:")
    print(f"  ffmpeg -i {OUTPUT_DIR}/shoptalk_demo.webm -c:v libx264 -crf 23 {OUTPUT_DIR}/shoptalk_demo.mp4")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Part 5 — Post-Recording Tips

### Converting Video Format

```bash
# WebM → MP4 (if using Playwright)
ffmpeg -i demo_recording/shoptalk_demo.webm \
  -c:v libx264 -crf 23 -preset medium \
  -c:a aac -b:a 128k \
  demo_recording/shoptalk_demo.mp4

# Trim video (remove first 5 seconds, keep 12 minutes)
ffmpeg -i demo_recording/shoptalk_demo.mp4 \
  -ss 00:00:05 -t 00:12:00 \
  -c copy demo_recording/shoptalk_demo_trimmed.mp4

# Add title overlay (optional, requires ffmpeg with drawtext)
ffmpeg -i demo_recording/shoptalk_demo.mp4 \
  -vf "drawtext=text='ShopTalk AI Assistant Demo':fontsize=36:fontcolor=white:x=(w-text_w)/2:y=50:enable='between(t,0,3)'" \
  -c:a copy demo_recording/shoptalk_demo_titled.mp4
```

### Compressing for Sharing

```bash
# Compress to <50MB for email/Slack
ffmpeg -i demo_recording/shoptalk_demo.mp4 \
  -c:v libx264 -crf 28 -preset slow \
  -vf scale=1280:720 \
  -c:a aac -b:a 96k \
  demo_recording/shoptalk_demo_compressed.mp4
```

### Adding Voiceover (Post-Recording)

If you recorded with Playwright (no audio), add a voiceover:

```bash
# Record voiceover separately
# (Use QuickTime → New Audio Recording, save as voiceover.m4a)

# Merge video + audio
ffmpeg -i demo_recording/shoptalk_demo.mp4 \
  -i voiceover.m4a \
  -c:v copy -c:a aac -shortest \
  demo_recording/shoptalk_demo_narrated.mp4
```

### Recommended Demo Queries for Different Audiences

| Audience | Good Queries | Why |
|---|---|---|
| Technical (ML) | "sports shoes for men" → "in blue?" | Shows reranking, multi-turn, embeddings |
| Business | "gift ideas for women" → "under $50" | Shows conversational commerce |
| Product/UX | Use voice input + TTS | Shows accessibility, modern UX |
| Engineering | Show `/health` endpoint, Docker logs | Shows production architecture |

---

## Quick-Start Cheat Sheet

```bash
# 1. Ensure EC2 stack is running
ssh -i key.pem ubuntu@<EC2-IP>
docker compose ps          # all services Up
curl localhost:8000/health # status: ok

# 2. From your Mac, open browser
open "http://<EC2-IP>:8501"

# 3. Start QuickTime screen recording (with mic)

# 4. Follow Scene 1–10 from this script

# 5. Stop recording, export as 1080p
```
