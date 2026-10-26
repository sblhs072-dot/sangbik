---
title: Sangbik
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Sangbik (hosted server)

Original Myanmar video recap / dubbing server: offline Whisper small
transcription, Google Translate / MyMemory / offline NLLB translation,
Edge TTS voice, FFmpeg video export. No API keys are in this repo.

**Deploy instructions are in DEPLOY.md — do NOT upload
`sangbik-release.keystore` or any `.env` file to this Space.**
When you create the Space, copy this file's contents to the Space's
own `README.md` (the frontmatter above is what tells Hugging Face
this is a Docker Space on port 7860). The Dockerfile at the Space
root builds the server in `server/`.
