FROM apache/airflow:2.10.0-python3.11

USER root
# Install git to allow pip install from git repositories if needed
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
         libgomp1 \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
