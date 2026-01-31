FROM apache/airflow:2.8.0

# Switch to root to install build dependencies (needed for some ML libs)
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
         libhdf5-dev \
         pkg-config \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user for pip install
USER airflow

# Install ML dependencies matching the project requirements
# Note: Installing directly to avoid path issues with mounting requirements.txt
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    xgboost \
    mlflow \
    "ray[tune]" \
    joblib \
    feast \
    boto3 \
    tensorflow
