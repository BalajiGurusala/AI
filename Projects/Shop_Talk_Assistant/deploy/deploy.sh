#!/bin/bash
# ============================================================
# ShopTalk — Deploy / Redeploy
# ============================================================
# Run from the project root directory.
#
# Usage:
#   ./deploy/deploy.sh          # Full deploy
#   ./deploy/deploy.sh restart  # Restart services only
#   ./deploy/deploy.sh logs     # View logs
#   ./deploy/deploy.sh stop     # Stop all
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

ACTION="${1:-deploy}"

case "$ACTION" in
    deploy)
        echo "=========================================="
        echo "ShopTalk — Deploying"
        echo "=========================================="

        # Validate data directory
        if [ ! -f "data/rag_text_index.npy" ]; then
            echo "ERROR: data/rag_text_index.npy not found!"
            echo "Upload NB03 artifacts to data/ first."
            echo "  scp data/*.pkl data/*.npy data/*.json ubuntu@<EC2>:~/shoptalk/data/"
            exit 1
        fi

        echo "[1/4] Building Docker images..."
        docker compose build

        echo "[2/4] Starting services..."
        docker compose up -d

        echo "[3/4] Waiting for Ollama to pull model (~2 min first time)..."
        sleep 10
        # Wait for backend to be healthy
        for i in $(seq 1 30); do
            if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
                echo "  Backend healthy!"
                break
            fi
            echo "  Waiting... ($i/30)"
            sleep 10
        done

        echo "[4/4] Verifying services..."
        echo ""
        
        # Health check
        HEALTH=$(curl -sf http://localhost:8000/health 2>/dev/null || echo '{"status":"unreachable"}')
        echo "Backend:  http://localhost:8000/health"
        echo "  $HEALTH"
        echo ""
        echo "Frontend: http://localhost:8501"
        echo "Ollama:   http://localhost:11434"
        echo ""
        echo "=========================================="
        echo "ShopTalk is running!"
        echo "=========================================="
        echo ""
        echo "Access the app at: http://<EC2-PUBLIC-IP>:8501"
        echo ""
        echo "Make sure EC2 security group allows:"
        echo "  - Port 8501 (Streamlit frontend)"
        echo "  - Port 8000 (API, optional)"
        ;;

    restart)
        echo "Restarting services..."
        docker compose restart
        echo "Done. Check: docker compose ps"
        ;;

    logs)
        docker compose logs -f --tail=50
        ;;

    stop)
        echo "Stopping all services..."
        docker compose down
        echo "Done."
        ;;

    status)
        docker compose ps
        echo ""
        curl -sf http://localhost:8000/health 2>/dev/null | python3 -m json.tool || echo "Backend not reachable"
        ;;

    *)
        echo "Usage: $0 {deploy|restart|logs|stop|status}"
        exit 1
        ;;
esac
