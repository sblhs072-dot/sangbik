# Sangbik — hosted server Dockerfile (primary host: Hugging Face Spaces, Docker SDK).
# Original code only. No API keys, no .env, no Studio99 code/assets are in this image.
# The server listens on $PORT (default 7860 here, the Hugging Face Docker port)
# on 0.0.0.0 — see server/index.js: app.listen(PORT, '0.0.0.0', ...).
# /api/health works WITHOUT any model loaded: it only checks that files/binaries
# exist; Whisper/NLLB models load lazily on the first workflow that needs them
# (see server/preload_models.py and DEPLOY.md for why lazy is the default).
FROM node:20-bookworm-slim

# System tools: Python (for faster-whisper / CTranslate2 / Edge TTS),
# FFmpeg + ffprobe (all video/audio work), espeak-ng (no-key Local TTS fallback).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg espeak-ng \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Node dependencies (layer-cached) ---
# Files referenced here must exist in the repo: server/package.json,
# server/package-lock.json.
COPY server/package.json server/package-lock.json ./server/
RUN npm ci --prefix server --omit=dev

# --- Python dependencies into server/.venv ---
# server/index.js expects Whisper at server/.venv/bin/python, so the venv
# MUST be at that exact path. PATH is set so `python3` also resolves to
# the venv (server/index.js checks edge-tts via `python3 -m edge_tts`).
# File referenced here must exist: server/requirements.txt.
COPY server/requirements.txt ./server/
RUN python3 -m venv /app/server/.venv \
    && /app/server/.venv/bin/pip install --no-cache-dir -r /app/server/requirements.txt
ENV PATH="/app/server/.venv/bin:$PATH"

# --- Server code ---
# Files referenced here must exist in the repo: server/index.js,
# server/whisper_local.py, server/edge_ms.py, server/nllb_ct2_translate.py,
# server/preload_models.py (optional pre-download helper, not run by default).
COPY server/index.js server/whisper_local.py server/edge_ms.py \
     server/nllb_ct2_translate.py server/preload_models.py ./server/
RUN mkdir -p /app/server/uploads /app/server/outputs /app/server/data \
    && chmod -R 755 /app/server/uploads /app/server/outputs /app/server/data

# Hugging Face Docker Spaces route to app_port (see SPACE-README.md).
# Render / other Docker hosts set PORT at runtime and override this default.
ENV PORT=7860
EXPOSE 7860

# OPTIONAL, commented out on purpose: bake Whisper small into the image.
# This downloads ~464 MB and makes the build much slower / larger, and can
# hit free-host build limits. Default is LAZY: the first workflow downloads
# the model into the runtime HF cache, with a clear log line. Uncomment
# only on a paid host with persistent disk if you want warm starts.
# RUN /app/server/.venv/bin/python /app/server/preload_models.py

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||7860)+'/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server/index.js"]
