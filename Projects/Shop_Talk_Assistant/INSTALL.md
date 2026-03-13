# ShopTalk AI Assistant — Installation Guide

> Step-by-step instructions for running ShopTalk locally on macOS and deploying to AWS EC2.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Part 1 — Local Install (macOS)](#part-1--local-install-macos)
  - [Option A: Standalone Streamlit (Simplest)](#option-a-standalone-streamlit-simplest)
  - [Option B: Backend + Frontend (Production-like)](#option-b-backend--frontend-production-like)
  - [Option C: Full Docker Stack Locally](#option-c-full-docker-stack-locally)
- [Part 2 — EC2 Deploy](#part-2--ec2-deploy)
  - [Step 1: Launch EC2 Instance](#step-1-launch-ec2-instance)
  - [Step 2: Run Setup Script](#step-2-run-setup-script)
  - [Step 3: Upload Artifacts](#step-3-upload-artifacts)
  - [Step 4: Clone Repo & Configure](#step-4-clone-repo--configure)
  - [Step 5: Deploy](#step-5-deploy)
  - [Step 6: Verify](#step-6-verify)
- [Environment Variables Reference](#environment-variables-reference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Data Artifacts (required before any install)

All modes need the notebook-generated artifacts in the `data/` directory.
Run notebooks in order on Kaggle, then copy the outputs to `data/`:

| File | Required | Produced By |
|---|---|---|
| `data/rag_products.pkl` | **Yes** | `03-rag-prototype.ipynb` |
| `data/rag_text_index.npy` | **Yes** | `03-rag-prototype.ipynb` |
| `data/rag_image_index.npy` | **Yes** | `03-rag-prototype.ipynb` |
| `data/rag_config.json` | **Yes** | `03-rag-prototype.ipynb` |
| `data/finetuned_text_index.npy` | Recommended | `05-fine-tuning.ipynb` |
| `data/models/finetuned-shoptalk-emb/` | Recommended | `05-fine-tuning.ipynb` |
| `data/eval_queries.json` | Optional | `04-llm-integration.ipynb` |

> **Current status:** All mandatory and recommended artifacts are already present in `data/`. You are ready to run.

### LLM — Choose one

| Option | Cost | Setup |
|---|---|---|
| **Ollama** (recommended for local) | Free | `brew install ollama && ollama pull llama3.2` |
| **OpenAI GPT-4o-mini** | Paid | Set `OPENAI_API_KEY` in `.env` |
| **Groq Llama-3.3-70B** | Free tier | Set `GROQ_API_KEY` in `.env` (get key at console.groq.com) |

---

## Part 1 — Local Install (macOS)

### Option A: Standalone Streamlit (Simplest)

Everything runs in a single process — no backend required. The app loads models and data locally.

**1. Create virtual environment**

```bash
cd /path/to/Shop_Talk_Assistant
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r app/requirements.txt
```

**3. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` for your LLM choice:

```dotenv
# Option 1 — Ollama (free, local)
# Start Ollama first: ollama serve
OLLAMA_BASE_URL=http://localhost:11434

# Option 2 — OpenAI
OPENAI_API_KEY=sk-...

# Option 3 — Groq
GROQ_API_KEY=gsk_...
```

**4. Start Ollama (if using Ollama)**

```bash
# In a separate terminal
ollama serve
```

**5. Run the app**

```bash
# From the project root, with .venv active
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

The app will automatically:
- Detect `data/rag_products.pkl`
- Load `data/finetuned_text_index.npy` (fine-tuned embeddings)
- Load `data/models/finetuned-shoptalk-emb/` as the query encoder

**Expected startup time:** 30–60 seconds for model loading on first run (CLIP + SentenceTransformer download/cache).

---

### Option B: Backend + Frontend (Production-like)

Runs FastAPI backend and Streamlit as separate processes. Streamlit acts as a thin client.

**1. Install dependencies for both services**

```bash
cd /path/to/Shop_Talk_Assistant
python3 -m venv .venv
source .venv/bin/activate

pip install -r backend/requirements.txt
pip install -r app/requirements.txt
```

**2. Configure environment**

```bash
cp .env.example .env
# Edit .env with your LLM API key or Ollama URL
```

**3. Start Ollama (if using Ollama)**

```bash
# Terminal 0 — keep running
ollama serve
```

**4. Start the backend**

```bash
# Terminal 1
source .venv/bin/activate

export DATA_DIR=./data                          # Override Docker default (/app/data)
export OLLAMA_BASE_URL=http://localhost:11434   # Override Docker default (http://ollama:11434)
# OR: export OPENAI_API_KEY=sk-...

uvicorn backend.src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

Confirm the backend is running:
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","product_count":XXXX,...}
```

**5. Start the frontend**

```bash
# Terminal 2
source .venv/bin/activate

export BACKEND_URL=http://localhost:8000
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** — sidebar shows **"Connected to backend"**.

---

### Option C: Full Docker Stack Locally

Requires Docker Desktop. Use this to replicate the exact production environment.

**1. Install Docker Desktop**

Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) and start it.

**2. Configure environment**

```bash
cp .env.example .env
# Set OPENAI_API_KEY or GROQ_API_KEY if not using Ollama
```

**3. Build and start all services**

```bash
cd /path/to/Shop_Talk_Assistant

# GPU Mac (Apple Silicon MPS not exposed to Docker — use CPU mode)
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

**4. Monitor startup**

```bash
docker compose logs -f
# Wait for: "ShopTalk Backend — Ready (XXXX products)"
# Ollama model pull takes ~2 min on first run
```

**5. Access**

| Service | URL |
|---|---|
| Frontend (Streamlit) | http://localhost:8501 |
| Backend API | http://localhost:8000/health |
| Ollama | http://localhost:11434 |

**6. Stop**

```bash
docker compose down
```

> **Note:** On Apple Silicon (M1/M2/M3), Docker containers run under Rosetta/ARM emulation. Model loading works but is slower than running natively (Option A or B).

---

## Part 2 — EC2 Deploy

### Recommended EC2 Instance Types

| Use Case | Instance | vCPU | RAM | GPU | Est. Cost |
|---|---|---|---|---|---|
| GPU (recommended) | `g4dn.xlarge` | 4 | 16 GB | T4 (16 GB) | ~$0.53/hr |
| CPU only | `t3.xlarge` | 4 | 16 GB | — | ~$0.17/hr |
| CPU budget | `t3.large` | 2 | 8 GB | — | ~$0.08/hr |

**AMI:** Ubuntu 22.04 LTS (search "ubuntu-22.04" in EC2 console)

**Storage:** 30 GB gp3 minimum (50 GB recommended for model cache)

---

### Step 1: Launch EC2 Instance

In the AWS Console:

1. Go to **EC2 → Launch Instance**
2. Select **Ubuntu Server 22.04 LTS**
3. Select instance type (see table above)
4. Create or select a key pair (`.pem` file) — save it locally
5. Under **Security Group**, open inbound ports:
   - `22` — SSH (your IP only)
   - `8501` — Streamlit frontend (0.0.0.0/0 for public access)
   - `8000` — Backend API (optional, your IP only)
6. Set storage to 30–50 GB
7. Launch

Note your **EC2 Public IP** (shown in the EC2 console).

---

### Step 2: Run Setup Script

SSH into your instance and run the one-time setup script:

```bash
# From your local machine
ssh -i /path/to/your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

```bash
# On EC2 — run setup (installs Docker, NVIDIA drivers if GPU present)
curl -fsSL https://raw.githubusercontent.com/<your-repo>/main/deploy/setup-ec2.sh | bash
# OR if you already have the repo cloned:
chmod +x deploy/setup-ec2.sh
./deploy/setup-ec2.sh
```

> **Important:** After the script completes, log out and log back in for Docker group permissions to take effect:
> ```bash
> exit
> ssh -i /path/to/your-key.pem ubuntu@<EC2-PUBLIC-IP>
> ```

---

### Step 3: Upload Artifacts

Upload the required data files from your **local machine**:

```bash
# From your local machine — upload all required artifacts
EC2_IP=<EC2-PUBLIC-IP>
KEY=/path/to/your-key.pem

# Required serving artifacts
scp -i $KEY \
  data/rag_products.pkl \
  data/rag_text_index.npy \
  data/rag_image_index.npy \
  data/rag_config.json \
  ubuntu@$EC2_IP:~/shoptalk/data/

# Recommended — fine-tuned model artifacts
scp -i $KEY \
  data/finetuned_text_index.npy \
  ubuntu@$EC2_IP:~/shoptalk/data/

# Fine-tuned model directory (recursive)
scp -i $KEY -r \
  data/models/ \
  ubuntu@$EC2_IP:~/shoptalk/data/

# Optional — evaluation artifacts
scp -i $KEY \
  data/eval_queries.json \
  data/llm_config.json \
  ubuntu@$EC2_IP:~/shoptalk/data/
```

---

### Step 4: Clone Repo & Configure

On the **EC2 instance**:

```bash
# Clone the repository
cd ~/shoptalk
git clone <your-repo-url> .
# OR copy files manually via scp if the repo is private and not on GitHub

# Configure environment
cp .env.example .env
nano .env   # or: vim .env
```

Minimum `.env` for EC2 (Ollama is default — no API key needed):

```dotenv
# Leave empty to use Ollama (started automatically by Docker Compose)
OPENAI_API_KEY=
GROQ_API_KEY=

# Fine-tuned model (auto-detected from data/models/, no change needed)
FINETUNED_MODEL_PATH=
```

If you prefer OpenAI or Groq over Ollama:

```dotenv
OPENAI_API_KEY=sk-...
# Then in docker-compose.yml, change default_llm to "openai" or "groq"
```

---

### Step 5: Deploy

```bash
cd ~/shoptalk

# CPU-only instance (t3.xlarge, t3.large, c5.xlarge):
./deploy/deploy.sh
# This is equivalent to: docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build

# GPU instance (g4dn.xlarge):
docker compose up -d --build
```

The deploy script:
1. Validates that `data/rag_text_index.npy` exists
2. Builds Docker images for backend and frontend
3. Starts all services (Ollama, backend, frontend)
4. Waits for the backend health check to pass
5. Prints access URLs

**First run takes ~5–10 minutes** — Ollama pulls the `llama3.2` model (~2 GB).

Other deploy commands:

```bash
./deploy/deploy.sh restart   # Restart containers without rebuild
./deploy/deploy.sh logs      # Stream live logs
./deploy/deploy.sh stop      # Stop all containers
./deploy/deploy.sh status    # Show running services + health
```

---

### Step 6: Verify

```bash
# Check all containers are running
docker compose ps

# Check backend health
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","product_count":XXXX,"llm":"ollama/llama3.2","device":"cuda"}
```

Access the app from your browser:

```
http://<EC2-PUBLIC-IP>:8501
```

To confirm the fine-tuned model is being used:

```bash
docker compose logs backend | grep -i "fine-tuned\|finetuned\|FINE-TUNED"
# Expected: "Loading FINE-TUNED SentenceTransformer from /app/data/models/finetuned-shoptalk-emb"
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `/app/data` | Path to artifacts directory. Override to `./data` for local runs. |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL. Override to `http://localhost:11434` for local runs. |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name. |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key for GPT-4o-mini. |
| `GROQ_API_KEY` | *(empty)* | Groq API key for Llama-3.3-70B (free at console.groq.com). |
| `FINETUNED_MODEL_PATH` | *(auto-detected)* | Explicit path to fine-tuned ST model. Leave empty — auto-detects `data/models/finetuned-shoptalk-emb/`. |
| `CHROMA_PATH` | `data/chroma` | ChromaDB persistence directory. |
| `DEFAULT_LLM` | `ollama` | Active LLM provider: `ollama`, `openai`, or `groq`. |
| `BACKEND_URL` | *(empty)* | Set in Streamlit frontend to point to backend. Empty = standalone mode. |

---

## Troubleshooting

### `Cannot find product data` on startup

The app cannot locate `rag_products.pkl`.

```bash
# Check what's in data/
ls -la data/

# Verify you're running from the project root
pwd
# Should be: .../Shop_Talk_Assistant

# For backend Option B: ensure DATA_DIR is set
export DATA_DIR=./data
```

### Backend shows base model instead of fine-tuned

```bash
# Check logs
docker compose logs backend | grep -i "model"
# OR (local)
# Look for log line at startup

# Verify the model directory exists
ls data/models/finetuned-shoptalk-emb/
```

### Ollama not reachable

```bash
# Local (Option A/B)
ollama serve   # Start in a separate terminal
curl http://localhost:11434/api/tags

# Docker — check the Ollama container
docker compose logs ollama
docker compose ps ollama
```

### Port 8501 not accessible from browser (EC2)

Check your EC2 security group inbound rules:
- Port `8501` must allow `0.0.0.0/0` (or your IP)
- Port `22` must allow your IP for SSH

```bash
# Also verify Streamlit is bound to 0.0.0.0
docker compose logs frontend | grep "You can now view"
```

### Backend container keeps restarting

```bash
docker compose logs backend --tail=50
```

Most common cause: missing `rag_text_index.npy` in `data/`. Upload artifacts per [Step 3](#step-3-upload-artifacts).

### SSL certificate error when loading models (`CERTIFICATE_VERIFY_FAILED`)

This happens on networks with a corporate SSL proxy. The `requests`/`transformers`
libraries use their own `certifi` cert bundle inside the venv, which doesn't include
your corporate CA.

**Fix — inject your corporate cert into the venv certifi bundle (one-time):**

```bash
# Locate certifi bundle inside your venv
CERTIFI_BUNDLE=$(python -c "import certifi; print(certifi.where())")
echo "Certifi bundle: $CERTIFI_BUNDLE"

# Backup and append corporate cert
cp "$CERTIFI_BUNDLE" "${CERTIFI_BUNDLE}.bak"
cat /Users/balaji.gurusala/Documents/cacert.pem >> "$CERTIFI_BUNDLE"
echo "Done — $(wc -l < $CERTIFI_BUNDLE) lines (was $(wc -l < ${CERTIFI_BUNDLE}.bak))"
```

After this, `streamlit run` will download HuggingFace models (SentenceTransformer, CLIP)
through the corporate proxy without errors.

> **Note:** If you recreate the venv (`rm -rf .venv && python3 -m venv .venv`),
> you must re-run the cert injection above.

**Graceful fallback:** The app also handles a CLIP download failure automatically —
if CLIP cannot load, it falls back to text-only search (all text queries work normally).
A yellow info banner appears in the UI on first run, and CLIP is cached for subsequent runs.

### Docker build fails with SSL certificate error

The `backend/Dockerfile` model pre-download is best-effort and non-fatal — build should succeed regardless. If it still fails:

```bash
# Skip the pre-download and let runtime load models
docker compose build --no-cache backend
```

### Out of memory (CPU instance)

Reduce model size by switching to a smaller Ollama model:

```bash
# On EC2
docker exec shoptalk-ollama ollama pull llama3.2:1b
```

Then update `.env`:
```dotenv
OLLAMA_MODEL=llama3.2:1b
```

And restart:
```bash
./deploy/deploy.sh restart
```
