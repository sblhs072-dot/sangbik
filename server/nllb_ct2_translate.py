"""Offline translation with CTranslate2 NLLB-200 (JustFrederik ct2-int8).
No key, no network at run time once the model is cached. English -> Burmese
(eng_Latn -> mya_Mymr). Reads a JSON list on stdin, prints a JSON list out.
Model dir is resolved from the HF cache; override with NLLB_CT2_DIR.
"""
import sys, json, os, glob

def model_dir():
    d = os.environ.get("NLLB_CT2_DIR")
    if d:
        return d
    hits = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--JustFrederik--nllb-200-distilled-600M-ct2-int8/snapshots/*")))
    if not hits:
        raise SystemExit("NLLB_CT2_NOT_INSTALLED")
    return hits[-1]

def main():
    texts = json.load(sys.stdin)
    if not texts:
        print("[]")
        return
    import ctranslate2
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    tok.src_lang = "eng_Latn"
    tr = ctranslate2.Translator(model_dir(), device="cpu")
    out = []
    for t in texts:
        # With src_lang set, the tokenizer prepends the source-language token
        # and appends </s> — both are required by NLLB; stripping them made
        # the decoder loop and repeat (verified in an earlier run).
        ids = tok(t)["input_ids"]
        tokens = tok.convert_ids_to_tokens(ids)
        res = tr.translate_batch([tokens], target_prefix=[["mya_Mymr"]], beam_size=5, max_decoding_length=256)[0]
        out_ids = tok.convert_tokens_to_ids(res.hypotheses[0])
        decoded = tok.decode(out_ids, skip_special_tokens=True)
        out.append(decoded.strip())
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
