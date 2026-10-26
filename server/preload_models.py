#!/usr/bin/env python3
"""Optional, NOT run by default: pre-download the offline models into the
Hugging Face cache (~/.cache/huggingface).

Default Sangbik behaviour is LAZY: the first workflow run downloads
Whisper tiny (~39 MB) via faster-whisper and, if NLLB is needed, the
NLLB-200 ct2-int8 model via CTranslate2. Lazy is the default because
baking ~1 GB+ of models into the Docker image makes the first deploy
much slower and can hit host build limits.

Run this manually only if you want models baked in (e.g. a paid host
with persistent disk), then uncomment the matching RUN line in the
Dockerfile. It needs network and several minutes; it downloads, it
does not need any API key.
"""
import sys

def main():
    from faster_whisper import WhisperModel
    print("Downloading Whisper tiny (int8, CPU)...", flush=True)
    WhisperModel("tiny", device="cpu", compute_type="int8")
    print("Whisper tiny ready.", flush=True)
    # NLLB is downloaded lazily by nllb_ct2_translate.py on first use;
    # pre-downloading it here is optional and is intentionally skipped,
    # because its tokenizer/model download is large and it is only a
    # fallback after Google Translate and MyMemory.
    print("Done. (NLLB stays lazy: first NLLB use downloads it.)", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
