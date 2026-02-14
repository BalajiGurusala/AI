# ShopTalk Frontend — Streamlit Chat UI
FROM python:3.11-slim

WORKDIR /app

# Python deps
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ /app/app/
COPY src/ /app/src/

EXPOSE 8501

# Run Streamlit (no file watcher in production)
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
