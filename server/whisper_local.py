#!/usr/bin/env python3
"""Sangbik offline transcription helper (original code for this app only).
FAST-v3 (2026-10-26): beam_size=1 greedy + VAD + language reuse (detect once, reuse for rest) + cpu_threads=2 for speed on Render Free.
Previous version used beam_size=5 (slow beam search).

Uses faster-whisper, model from WHISPER_MODEL env (default "tiny", multilingual, auto language detection),
CPU int8. Usage:
  python whisper_local.py --check
      Loads the model and prints MODEL_READY <model> — proves the model
      is actually loaded, not merely present on disk.
  python whisper_local.py <audio1.wav> [<audio2.wav> ...]
      Transcribes each audio file in ONE process (model loaded once) and
      prints one JSON object per line:
      {"file": "...", "model": "tiny", "language": "en",
       "segments": [{"start":..,"end":..,"text":".."}]}
No network key is needed. The model is downloaded by faster-whisper on
first load into the standard Hugging Face cache.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

MODEL = os.environ.get("WHISPER_MODEL", "tiny")


def load_model():
    from faster_whisper import WhisperModel

    return WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="*", default=[])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    try:
        model = load_model()
    except Exception as e:
        print(f"WHISPER_MODEL_LOAD_FAILED: {e}", file=sys.stderr)
        return 1
    if args.check:
        print(f"MODEL_READY {MODEL} (multilingual)")
        return 0
    if not args.audio:
        print("usage: whisper_local.py [--check] <audio.wav> [...]", file=sys.stderr)
        return 2
    detected_lang = None
    try:
        for audio in args.audio:
            # Decode with ffmpeg to 16 kHz mono float32 ourselves, instead of
            # faster-whisper's PyAV decoder: the installed PyAV (19.x) removed
            # the metadata_errors argument faster-whisper 1.2.1 passes, so
            # file-path transcription fails there. The model itself is unchanged.
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", audio, "-ac", "1", "-ar", "16000",
                 "-f", "s16le", "-"], capture_output=True, check=True).stdout
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = model.transcribe(pcm, beam_size=1, best_of=1, temperature=0, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500), language=detected_lang, condition_on_previous_text=False, word_timestamps=False) if detected_lang else model.transcribe(pcm, beam_size=1, best_of=1, temperature=0, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500), condition_on_previous_text=False, word_timestamps=False)
            if detected_lang is None: detected_lang = info.language
            out = []
            for s in segments:
                text = (s.text or "").strip()
                if text:
                    out.append({"start": round(float(s.start), 2),
                                "end": round(float(s.end), 2), "text": text})
            print(json.dumps({"file": audio, "model": MODEL,
                              "language": info.language,
                              "language_probability": round(float(info.language_probability), 3),
                              "segments": out}, ensure_ascii=False), flush=True)
        return 0
    except Exception as e:
        print(f"WHISPER_TRANSCRIBE_FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
