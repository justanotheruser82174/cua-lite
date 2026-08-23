
FROM python:3.12-bookworm

ARG ONLINE_MIND2WEB_REPO=https://github.com/OSU-NLP-Group/Online-Mind2Web.git
ARG ONLINE_MIND2WEB_SHA=f0d805ee0e9e0b3ea70911e45e5264b72968f3dc

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ONLINE_MIND2WEB_HOME=/opt/Online-Mind2Web \
    PYTHONPATH=/opt/Online-Mind2Web/src:/opt/Online-Mind2Web

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      ca-certificates \
      curl \
      git \
      fonts-liberation \
      fonts-noto-cjk \
      fonts-noto-color-emoji \
      procps \
 && rm -rf /var/lib/apt/lists/*

# Keep the expensive browser layer cache-stable across upstream code changes.
RUN pip install --no-cache-dir \
      fastapi \
      "uvicorn[standard]" \
      playwright \
 && playwright install --with-deps chromium

RUN git init -q /opt/Online-Mind2Web \
 && cd /opt/Online-Mind2Web \
 && git remote add origin "${ONLINE_MIND2WEB_REPO}" \
 && git fetch -q --depth 1 origin "${ONLINE_MIND2WEB_SHA}" \
 && git checkout -q FETCH_HEAD \
 && rm -rf .git

WORKDIR /opt/Online-Mind2Web

# Upstream currently contains judge utilities only; install them for later reward
# wiring while the CUA-Lite Playwright RPC server owns browser generation.
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /tmp/online_mind2web-downloads

COPY server.py entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
