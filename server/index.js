/* Sangbik — server. No secrets are hard-coded; keys come from .env only.
   Honest behaviour: AI script generation runs ONLY if an AI key is configured.
   TTS runs ONLY if a provider is actually available. Local (no-key) TTS uses
   espeak-ng with its Burmese voice ("my") when that binary is installed;
   it is robotic and lower quality than Azure, and is labelled as such. */
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn, execFile, spawnSync } = require('child_process');
// No-key local TTS: true iff the espeak-ng binary is on PATH. Checked live
// (not cached at startup) so installing espeak-ng then restarting is not
// the only path — a fresh health check after install also reflects it.
function localTtsConfigured() { try { return spawnSync('which', ['espeak-ng'], { encoding: 'utf8' }).status === 0; } catch { return false; } }
// Sangbik defaults (user-chosen, 2026-10-26): offline Whisper tiny for
// transcription, Google Translate for translation, Edge TTS for voice.
// Azure/OpenAI/Gemini remain available if explicitly selected, but are NOT
// the defaults. No key is needed for the three defaults.
const WHISPER_PYTHON = path.join(__dirname, '.venv/bin/python');
const WHISPER_SCRIPT = path.join(__dirname, 'whisper_local.py');
const WHISPER_MODEL_NAME = process.env.WHISPER_MODEL || 'tiny';
function whisperInstalled() { try { return fs.existsSync(WHISPER_PYTHON) && fs.existsSync(WHISPER_SCRIPT); } catch { return false; } }
function edgeTtsInstalled() { try { return spawnSync('python3', ['-m', 'edge_tts', '--help'], { encoding: 'utf8' }).status === 0; } catch { return false; } }
function defaultTtsProvider() { return 'edge-tts'; }
async function runWhisperCheck() {
  if (!whisperInstalled()) throw new Error('Offline Whisper is not installed on this server: the server .venv (faster-whisper) or whisper_local.py is missing. Reinstall: python3 -m venv server/.venv && server/.venv/bin/pip install faster-whisper, then restart the server.');
  const { out, err } = await run(WHISPER_PYTHON, [WHISPER_SCRIPT, '--check']);
  const line = (out || '').trim();
  if (!line.startsWith('MODEL_READY')) throw new Error(`Offline Whisper model did not load: ${line || err.slice(-500)}`);
  return line;
}
async function transcribeOfflineFiles(wavFiles) {
  if (!whisperInstalled()) throw new Error('Offline Whisper is not installed on this server (server/.venv or whisper_local.py missing).');
  const { out, err } = await run(WHISPER_PYTHON, [WHISPER_SCRIPT, ...wavFiles]);
  const results = [];
  for (const line of (out || '').split('\n')) { const t = line.trim(); if (t.startsWith('{')) { try { results.push(JSON.parse(t)); } catch {} } }
  if (!results.length) throw new Error(`Offline Whisper produced no output: ${(err || '').slice(-1000)}`);
  return results;
}
// Google Translate (free endpoint, no key): batch translate. Returns the
// translated strings in the same order. Throws the exact HTTP error on
// failure (e.g. 429 rate-limit) — the caller logs it and tries the AI
// fallback, never inventing a translation.
async function googleTranslateBatch(texts, target = 'my') {
  if (!texts.length) return [];
  const params = new URLSearchParams();
  params.append('client', 'gtx'); params.append('sl', 'auto'); params.append('tl', target); params.append('dt', 't');
  for (const t of texts) params.append('q', t);
  const url = 'https://translate.googleapis.com/translate_a/single?' + params.toString();
  const r = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  const body = await r.text();
  if (!r.ok) throw new Error(`Google Translate HTTP ${r.status}: ${body.slice(0, 300)}`);
  let j; try { j = JSON.parse(body); } catch { throw new Error(`Google Translate returned unparseable output: ${body.slice(0, 300)}`); }
  const rows = j[0] || [];
  if (rows.length !== texts.length) throw new Error(`Google Translate returned ${rows.length} translations for ${texts.length} inputs.`);
  return rows.map(row => (Array.isArray(row) ? row.map(x => (typeof x === 'string' ? x : '')).join('') : '').trim());
}
// MyMemory free API (no key). MUST reject its quota-warning text: a
// responseStatus other than 200, or a translatedText starting with
// "MYMEMORY WARNING", is a failure, never a translation.
async function myMemoryTranslateBatch(texts, target = 'my') {
  const out = [];
  for (const t of texts) {
    const url = 'https://api.mymemory.translated.net/get?' + new URLSearchParams({ q: t, langpair: `en|${target}` }).toString();
    const r = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    const body = await r.text();
    let j; try { j = JSON.parse(body); } catch { throw new Error(`MyMemory returned unparseable output (HTTP ${r.status}): ${body.slice(0, 200)}`); }
    const txt = j.responseData?.translatedText || '';
    if (j.responseStatus !== 200 || !txt || txt.startsWith('MYMEMORY WARNING')) throw new Error(`MyMemory HTTP ${j.responseStatus}: ${(j.responseDetails || body).slice(0, 200)}`);
    out.push(txt);
  }
  return out;
}
// Offline NLLB-200 translation via CTranslate2 (no key, no network at run
// time, no torch needed) — only if ctranslate2 + transformers are installed
// AND the JustFrederik ct2-int8 model is cached (see nllb_ct2_translate.py).
async function nllbTranslateBatch(texts) {
  const py = fs.existsSync(WHISPER_PYTHON) ? WHISPER_PYTHON : 'python3';
  const { out, err } = await new Promise((resolve, reject) => {
    const p = spawn(py, [path.join(__dirname, 'nllb_ct2_translate.py')]);
    let o = '', e = '';
    p.stdout.on('data', d => o += d); p.stderr.on('data', d => e += d);
    p.on('error', reject);
    p.on('close', code => code === 0 ? resolve({ out: o, err: e }) : reject(new Error(e.slice(-500) || `nllb exit ${code}`)));
    p.stdin.end(JSON.stringify(texts));
  });
  const parsed = JSON.parse(out.trim().split('\n').pop());
  if (!Array.isArray(parsed) || parsed.length !== texts.length) throw new Error('NLLB returned the wrong number of translations.');
  if (parsed.some(t => !t || !t.trim())) throw new Error('NLLB returned an empty translation for one or more lines.');
  return parsed;
}
// Offline Argos Translate (no key, no network) — only if the
// argostranslate package AND an en->my package are installed on this server.
// NOTE: as of 2026-10-26 no en->my Argos package is known to exist, so this
// path is kept only as a documented dead end; NLLB above is the offline path.
async function argosTranslateBatch(texts) {
  const py = fs.existsSync(WHISPER_PYTHON) ? WHISPER_PYTHON : 'python3';
  const script = `import sys, json, argos_translate
from argos_translate import translate as _t
installed = _t.get_installed_languages()
src = next((l for l in installed if l.code == 'en'), None)
dst = next((l for l in installed if l.code == 'my'), None)
if not src or not dst: raise SystemExit('ARGOS_NO_MY_PACKAGE')
tr = src.get_translation(dst)
print(json.dumps([tr(t) for t in json.load(sys.stdin)], ensure_ascii=False))`;
  const { out, err } = await new Promise((resolve, reject) => {
    const p = spawn(py, ['-c', script]);
    let o = '', e = '';
    p.stdout.on('data', d => o += d); p.stderr.on('data', d => e += d);
    p.on('error', reject);
    p.on('close', code => code === 0 ? resolve({ out: o, err: e }) : reject(new Error(e.slice(-500) || `argos exit ${code}`)));
    p.stdin.end(JSON.stringify(texts));
  });
  const parsed = JSON.parse(out.trim().split('\n').pop());
  if (!Array.isArray(parsed) || parsed.length !== texts.length) throw new Error('Argos returned the wrong number of translations.');
  return parsed;
}
async function translateLinesWithGoogleOrFallback(lines) {
  // lines: [{original}] -> {translations, path, error}
  // Order: Google -> MyMemory (free, no key) -> NLLB offline (no key) -> Argos offline (no key, likely no en-my package) -> AI (only if configured).
  // Every step is labelled in `path`/`googleLog`; a failure never invents a translation.
  const originals = lines.map(l => l.original);
  const attempts = [];
  try {
    const t0 = Date.now();
    const translations = await googleTranslateBatch(originals, 'my');
    const ms = Date.now() - t0;
    return { translations, path: 'google', googleLog: `Google batch 1/1 ok via translate.googleapis.com (${(ms / 1000).toFixed(1)}s)`, error: null };
  } catch (ge) { attempts.push(`Google: ${ge.message}`); }
  try {
    const translations = await myMemoryTranslateBatch(originals, 'my');
    return { translations, path: 'mymemory', googleLog: 'Translated via MyMemory free API (no key) — Google failed in this sandbox, see Workflow log for the exact Google error.', error: `Google failed: ${attempts.join(' ')} — MyMemory fallback used instead.` };
  } catch (me) { attempts.push(`MyMemory: ${me.message}`); }
  try {
    const translations = await nllbTranslateBatch(originals);
    return { translations, path: 'nllb-offline', googleLog: 'Translated offline with NLLB-200 en->mya_Mymr on this server (no key, no network).', error: `Google + MyMemory failed: ${attempts.join(' ')} — NLLB offline fallback used instead.` };
  } catch (ne) { attempts.push(`NLLB offline: ${ne.message}`); }
  try {
    const translations = await argosTranslateBatch(originals);
    return { translations, path: 'argos-offline', googleLog: 'Translated offline with Argos Translate en->my on this server (no key, no network).', error: `Google + MyMemory + NLLB failed: ${attempts.join(' ')} — Argos offline fallback used instead.` };
  } catch (ae) { attempts.push(`Argos offline: ${ae.message}`); }
  // Last fallback: per-line AI translation, only if an AI key is configured.
  const aiConfigured = !!(process.env.AI_PROVIDER && (process.env.AI_API_KEY || process.env.OPENAI_API_KEY || process.env.GEMINI_API_KEY));
  if (!aiConfigured) return { translations: null, path: 'failed', googleLog: null, error: `All free translation paths failed (${attempts.join(' ')}), and no AI key is configured for fallback, so no translation was invented.` };
  try {
    const translations = [];
    for (const o of originals) translations.push((await translateTextToBurmese(o)) || '');
    if (translations.some(t => t)) return { translations, path: 'ai-fallback', googleLog: null, error: `Free translation paths failed (${attempts.join(' ')}), AI fallback used instead.` };
    return { translations: null, path: 'failed', googleLog: null, error: `All translation paths failed (${attempts.join(' ')}), AI fallback also returned nothing, so no translation was invented.` };
  } catch (aie) {
    return { translations: null, path: 'failed', googleLog: null, error: `All translation paths failed (${attempts.join(' ')}), AI fallback failed: ${aie.message}. No translation was invented.` };
  }
}


const PORT = process.env.PORT || 4000;
const UPLOAD_DIR = path.resolve(process.env.UPLOAD_DIR || './uploads');
const OUT_DIR = path.join(__dirname, 'outputs');
const DATA_FILE = path.join(__dirname, 'data', 'projects.json');
const MAX_MB = parseInt(process.env.MAX_FILE_MB || '10240', 10);
const CHUNK_SIZE = 5 * 1024 * 1024; // ~5 MB chunks, client-side split size
const CHUNK_DIR = path.join(UPLOAD_DIR, '.chunks');
const TRANSCRIBE_CONCURRENCY = Math.max(1, parseInt(process.env.TRANSCRIBE_CONCURRENCY || '4', 10));
const TRANSCRIBE_MODEL = process.env.TRANSCRIBE_MODEL || 'whisper-1';
const BLUR_SIGMA = { light: 5, medium: 15, strong: 30 };
const ZOOM_MAX = { light: 1.10, medium: 1.20, strong: 1.35 }; // zoom-every-5s max zoom factor
// Clarity enhancement (sharpen/denoise, NOT magic restoration): unsharp luma amount
// light 0.5 / medium 1.0 / strong 1.5, chroma amount kept at 0 to avoid colour artefacts.
const CLARITY_UNSHARP = { light: '5:5:0.5:5:5:0.0', medium: '5:5:1.0:5:5:0.0', strong: '5:5:1.5:5:5:0.0' };
const CLARITY_DENOISE = { light: '1:1:2:2', medium: '1.5:1.5:3:3', strong: '2:2:4:4' }; // hqdn3d, small values only
const ALLOWED_MUSIC_EXT = ['.mp3', '.wav', '.m4a', '.aac', '.ogg'];
// Subtitle fonts verified present on this server (fc-list). For Burmese text use a
// Myanmar font; DejaVu/Liberation do not contain Myanmar glyphs, so Burmese text
// in those fonts may render as boxes — the UI says so next to the choice.
const FONT_WHITELIST = ['Noto Sans Myanmar', 'Noto Serif Myanmar', 'DejaVu Sans', 'Liberation Sans'];
const ALLOWED_EXT = ['.mp4', '.mov', '.webm', '.mkv', '.avi'];
const TTS_VOICES = (process.env.TTS_VOICES || 'my-MM-NilarNeural,my-MM-ThihaNeural').split(',').map(s => s.trim()).filter(Boolean).slice(0, 5);

fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(CHUNK_DIR, { recursive: true });
fs.mkdirSync(OUT_DIR, { recursive: true });
fs.mkdirSync(path.dirname(DATA_FILE), { recursive: true });

function loadProjects() { try { return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8')); } catch { return {}; } }
function saveProjects(p) { fs.writeFileSync(DATA_FILE, JSON.stringify(p, null, 2)); }
let projects = loadProjects();
function newId() { return crypto.randomBytes(8).toString('hex'); }
function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, opts);
    let out = '', err = '';
    p.stdout.on('data', d => out += d); p.stderr.on('data', d => err += d);
    p.on('error', reject);
    p.on('close', code => code === 0 ? resolve({ out, err }) : reject(new Error(`${cmd} exited ${code}: ${err.slice(-2000)}`)));
  });
}
async function probe(file) {
  const { out } = await run('ffprobe', ['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file]);
  const j = JSON.parse(out);
  const v = j.streams.find(s => s.codec_type === 'video') || {};
  const a = j.streams.find(s => s.codec_type === 'audio') || {};
  return { duration: parseFloat(j.format.duration || 0), size: parseInt(j.format.size || 0, 10), format: j.format.format_name, videoCodec: v.codec_name, audioCodec: a.codec_name, width: v.width, height: v.height, resolution: v.width && v.height ? `${v.width}x${v.height}` : null };
}
function escapeSrt(s) { return s.replace(/\r?\n/g, ' ').trim(); }
function fmtSrt(t) { const ms = Math.floor(t * 1000); const h = String(Math.floor(ms / 3600000)).padStart(2, '0'); const m = String(Math.floor(ms / 60000) % 60).padStart(2, '0'); const s = String(Math.floor(ms / 1000) % 60).padStart(2, '0'); const r = String(ms % 1000).padStart(3, '0'); return `${h}:${m}:${s},${r}`; }
function scriptToSegments(script, totalDuration) {
  const lines = script.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  if (!lines.length) return [];
  const dur = totalDuration > 0 ? totalDuration : lines.length * 5;
  const per = dur / lines.length;
  return lines.map((text, i) => ({ start: i * per, end: (i + 1) * per, text }));
}
function writeSrt(project, segments) {
  const p = path.join(OUT_DIR, `${project.id}.srt`);
  const body = segments.map((s, i) => `${i + 1}\n${fmtSrt(s.start)} --> ${fmtSrt(s.end)}\n${escapeSrt(s.text)}\n`).join('\n');
  fs.writeFileSync(p, body); return p;
}
function fmtVtt(t) { return fmtSrt(t).replace(',', '.'); }
function writeVtt(project, segments, suffix = '') {
  const p = path.join(OUT_DIR, `${project.id}${suffix}.vtt`);
  const body = 'WEBVTT\n\n' + segments.map((s, i) => `${fmtVtt(s.start)} --> ${fmtVtt(s.end)}\n${escapeSrt(s.text)}\n`).join('\n');
  fs.writeFileSync(p, body); return p;
}
function normaliseLines(lines) {
  if (!Array.isArray(lines)) return [];
  return lines.map(l => ({ start: parseFloat(l.start || 0), end: parseFloat(l.end || 0), original: String(l.original || ''), burmese: String(l.burmese || '') })).filter(l => l.end > l.start && l.burmese);
}
function getSceneSegments(scenes, start, end) {
  const cuts = [start, ...(scenes || []).filter(t => t > start && t < end).sort((a, b) => a - b), end];
  const segs = [];
  for (let i = 0; i < cuts.length - 1; i++) if (cuts[i + 1] > cuts[i]) segs.push({ index: i, start: cuts[i], end: cuts[i + 1], duration: cuts[i + 1] - cuts[i] });
  return segs;
}
function mapTimeForOrder(order, segs, t) {
  // Map an old timestamp to the new timeline after reordering segments.
  const newStarts = {}; let acc = segs[0] ? segs[0].start : 0;
  // New timeline starts at the original start; each reordered segment keeps its duration.
  let cursor = segs.length ? segs[0].start : t;
  for (const idx of order) { newStarts[segs[idx].index] = cursor; cursor += segs[idx].duration; }
  for (const seg of segs) if (t >= seg.start && t <= seg.end) return newStarts[seg.index] + (t - seg.start);
  return t;
}
function splitText(text) {
  const parts = text.split(/(?<=[။!?])\s+/).map(s => s.trim()).filter(Boolean);
  const segments = []; let buf = '';
  for (const part of parts) { if ((buf + ' ' + part).length > 800 && buf) { segments.push(buf); buf = part; } else buf = (buf ? buf + ' ' : '') + part; }
  if (buf) segments.push(buf);
  return segments.length ? segments : [text];
}
async function geminiTtsSegment(text, voice, model, key, outMp3) {
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ contents: [{ parts: [{ text }] }], generationConfig: { responseModalities: ['AUDIO'], speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } } } }) });
  const j = await res.json();
  const part = j.candidates?.[0]?.content?.parts?.find(p => p.inlineData);
  if (!part) throw new Error(`Gemini TTS returned no audio: ${JSON.stringify(j).slice(0, 500)}`);
  const raw = Buffer.from(part.inlineData.data, 'base64');
  const rawFile = outMp3 + '.raw';
  fs.writeFileSync(rawFile, raw);
  // Gemini TTS audio is 24kHz mono PCM (audio/L16). Convert with ffmpeg; never invent a file on failure.
  await run('ffmpeg', ['-y', '-f', 's16le', '-ar', '24000', '-ac', '1', '-i', rawFile, outMp3]);
  fs.unlinkSync(rawFile);
}
function escapeXml(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
async function azureTtsSegment(text, voice, speed, pitch, key, region, outMp3) {
  // Paid Microsoft Azure Speech — NOT the free Edge endpoint blocked in Studio99.
  const ratePct = Math.round((speed - 1) * 100);
  const rate = `${ratePct >= 0 ? '+' : ''}${ratePct}%`;
  const pitchAttr = pitch && /^[+-]?\d+(Hz|%)$/.test(pitch.trim()) ? ` pitch="${pitch.trim()}"` : '';
  const ssml = `<speak version="1.0" xml:lang="my-MM"><voice name="${voice}"><prosody rate="${rate}"${pitchAttr}>${escapeXml(text)}</prosody></voice></speak>`;
  const res = await fetch(`https://${region}.tts.speech.microsoft.com/cognitiveservices/v1`, { method: 'POST', headers: { 'Ocp-Apim-Subscription-Key': key, 'Content-Type': 'application/ssml+xml', 'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3', 'User-Agent': 'Sangbik' }, body: ssml });
  if (!res.ok) throw new Error(`Azure Speech HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (!buf.length) throw new Error('Azure Speech returned empty audio');
  fs.writeFileSync(outMp3, buf);
}
async function concatMp3s(files, outMp3, pauseSec, tmpDir) {
  // Pause between segments is implemented as real silence (ffmpeg anullsrc), not by faking timing.
  const list = [];
  for (let i = 0; i < files.length; i++) {
    list.push(files[i]);
    if (pauseSec > 0 && i < files.length - 1) {
      const sil = path.join(tmpDir, `sil${i}.mp3`);
      await run('ffmpeg', ['-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', String(pauseSec), '-c:a', 'libmp3lame', sil]);
      list.push(sil);
    }
  }
  const listFile = path.join(tmpDir, 'list.txt');
  fs.writeFileSync(listFile, list.map(f => `file '${f}'`).join('\n'));
  await run('ffmpeg', ['-y', '-f', 'concat', '-safe', '0', '-i', listFile, '-c', 'copy', outMp3]);
}
function estimate(script, wpm) {
  const words = script.trim().split(/\s+/).filter(Boolean).length;
  const chars = script.replace(/\s/g, '').length;
  return { words, characters: chars, minutesAtWpm: wpm ? +(words / wpm).toFixed(2) : null };
}
function structuredDraft(project, scenes, notes, spoilerMode) {
  const fmt = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  const at = i => scenes[i] ? fmt(scenes[i]) : '—';
  const head = spoilerMode === 'spoiler-free'
    ? 'ဒီပြန်လည်ပြောပြချက်သည် ဇာတ်လမ်းအဓိကအဖြစ်အပျက်များကို မဖော်ပြဘဲ၊ ဇာတ်ကောင်နှင့် အငွေ့အသက်ကိုသာ မိတ်ဆက်ပေးပါသည်။'
    : 'အောက်ပါ မူကြမ်းသည် ဇာတ်ဝင်ခန်းအချိန်များကိုသာ အခြေခံထားပါသည်။ ဇာတ်လမ်းအဖြစ်အပျက်အမှန်များကို သင်ကိုယ်တိုင် ပြင်ဆင်ဖြည့်စွက်ပါ — မသိသောအဖြစ်အပျက်ကို မထည့်ပါနှင့်။';
  return [
    `## အဖွင့် (Hook) — ${at(0)}`,
    head,
    `## အစ — ${at(0)} မှ ${at(1)}`,
    '[ဤနေရာတွင် အစပိုင်းဇာတ်ကောင်နှင့် အခြေအနေကို သင်ကိုယ်တိုင်ရေးပါ]',
    `## အလယ် — ${at(1)} မှ ${at(2)}`,
    '[ဤနေရာတွင် အလယ်ပိုင်းအဖြစ်အပျက်ကို သင်ကိုယ်တိုင်ရေးပါ]',
    `## အထွတ်အထိပ် — ${at(2)} မှ ${at(3)}`,
    '[ဤနေရာတွင် အထွတ်အထိပ်ကို သင်ကိုယ်တိုင်ရေးပါ]',
    `## အဆုံး — ${at(3)}`,
    '[ဤနေရာတွင် အဆုံးသတ်ကို သင်ကိုယ်တိုင်ရေးပါ]',
    notes ? `## မှတ်စု\n${notes}` : '',
  ].filter(Boolean).join('\n\n');
}
async function aiDraft(project, scenes, notes, spoilerMode) {
  const provider = (process.env.AI_PROVIDER || '').toLowerCase();
  const key = process.env.AI_API_KEY || process.env.OPENAI_API_KEY || process.env.GEMINI_API_KEY;
  if (!provider || !key) return null;
  const prompt = `Write an ORIGINAL, commentary-style Burmese movie recap script draft (not a transcript, do not reproduce long dialogue). Scene timestamps (seconds): ${scenes.join(', ')}. Duration: ${project.metadata?.duration || '?'}s. Spoiler mode: ${spoilerMode}. User notes: ${notes || '(none)'}. Structure: hook, beginning, middle, climax, ending. Keep it transformative commentary, no invented plot events beyond the notes/timestamps.`;
  try {
    if (provider === 'gemini') {
      const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${key}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }) });
      const j = await res.json();
      const text = j.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || '';
      if (!text) throw new Error('Empty Gemini response');
      return text;
    }
    // Default: OpenAI-compatible chat completions
    const res = await fetch('https://api.openai.com/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body: JSON.stringify({ model: 'gpt-4o-mini', messages: [{ role: 'user', content: prompt }] }) });
    const j = await res.json();
    const text = j.choices?.[0]?.message?.content || '';
    if (!text) throw new Error('Empty OpenAI response');
    return text;
  } catch (e) { throw new Error(`AI provider failed: ${e.message}`); }
}


// ---- Chunked upload helpers ----
function chunkMetaPath(uploadId) { return path.join(CHUNK_DIR, uploadId + '.json'); }
function loadChunkMeta(uploadId) { try { return JSON.parse(fs.readFileSync(chunkMetaPath(uploadId), 'utf8')); } catch { return null; } }
function saveChunkMeta(m) { fs.mkdirSync(CHUNK_DIR, { recursive: true }); fs.writeFileSync(chunkMetaPath(m.uploadId), JSON.stringify(m)); }
function validUploadId(id) { return typeof id === 'string' && /^[A-Za-z0-9._-]{4,200}$/.test(id); }

// ---- Scene detection (shared by analyze + auto-job) ----
async function detectScenes(filePath) {
  // FAST-v2: detect scenes at 320px wide instead of full 1280px — same cuts, ~4x fewer pixels to decode.
  const { err } = await run('ffmpeg', ['-i', filePath, '-vf', "scale=320:-1,select='gt(scene,0.4)',showinfo", '-f', 'null', '-']);
  const scenes = [];
  for (const m of err.matchAll(/pts_time:([\d.]+)/g)) { const t = parseFloat(m[1]); if (!scenes.length || t - scenes[scenes.length - 1] > 1) scenes.push(+t.toFixed(2)); }
  if (!scenes.length) scenes.push(0);
  return scenes;
}

// ---- Translation of one segment (AI only; never invented) ----
async function translateTextToBurmese(text) {
  const provider = (process.env.AI_PROVIDER || '').toLowerCase();
  const key = process.env.AI_API_KEY || process.env.OPENAI_API_KEY || process.env.GEMINI_API_KEY;
  if (!provider || !key) return null;
  const prompt = `Translate the following spoken text into natural, conversational Burmese (Myanmar script) suitable for subtitles/dubbing. Return ONLY the Burmese translation, no explanation, no extra text:\n\n${text}`;
  if (provider === 'gemini') {
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${key}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }) });
    const j = await res.json();
    const out = j.candidates?.[0]?.content?.parts?.map(x => x.text).join('').trim();
    return out || null;
  }
  const res = await fetch('https://api.openai.com/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body: JSON.stringify({ model: 'gpt-4o-mini', messages: [{ role: 'user', content: prompt }] }) });
  const j = await res.json();
  return (j.choices?.[0]?.message?.content || '').trim() || null;
}

// ---- Auto transcript+translation job (background, never blocks upload) ----
async function runAutoJob(projectId) {
  const p = projects[projectId]; if (!p || !p.file) return;
  if (p.autoJob && p.autoJob.status === 'running') return;
  p.autoJob = { status: 'running', stage: 'scenes', percent: 0, completedSegments: 0, totalSegments: 0, lines: [], message: 'Auto transcript job started: scenes, then audio segments.', startedAt: new Date().toISOString() };
  saveProjects(projects);
  try {
    // FAST-v2: scene detection skipped in auto job — transcript uses 60s time segments, not scenes, so decoding the full video for scenes only wasted minutes on Render Free. Scenes are still available via /analyze if needed.
    p.scenes = p.scenes || [];
    p.autoJob.stage = 'scenes-done'; p.autoJob.percent = 5; saveProjects(projects);
    // No key gate: offline Whisper + Google Translate need no key.
    if (!whisperInstalled()) {
      p.autoJob = { ...p.autoJob, status: 'failed', stage: 'done', percent: 100, completedSegments: 0, totalSegments: 0,
        message: 'Offline Whisper is not installed on this server (server/.venv or whisper_local.py missing) — no transcript was faked. Manual lines still work.',
        finishedAt: new Date().toISOString() };
      saveProjects(projects); return;
    }
    const duration = p.metadata?.duration || (await probe(p.file.path)).duration || 0;
    const totalSegments = Math.max(1, Math.ceil(duration / 60));
    p.autoJob.totalSegments = totalSegments; p.autoJob.stage = 'extracting-audio'; p.autoJob.percent = 10; saveProjects(projects);
    const tmpDir = fs.mkdtempSync(path.join(OUT_DIR, 'auto-'));
    const wavPath = path.join(tmpDir, 'audio.wav');
    await run('ffmpeg', ['-y', '-i', p.file.path, '-vn', '-ac', '1', '-ar', '16000', wavPath]);
    // Segment jobs: transcribe (Whisper) then immediately translate that segment, in parallel workers.
    const jobs = [];
    for (let i = 0; i < totalSegments; i++) jobs.push({ index: i, start: i * 60, end: Math.min((i + 1) * 60, duration || 60) });
    // Offline Whisper, one process for all segment wavs (model loaded once).
    const segWavs = [];
    for (const job of jobs) {
      const segWav = path.join(tmpDir, `seg${job.index}.wav`);
      await run('ffmpeg', ['-y', '-ss', String(job.start), '-t', String(job.end - job.start), '-i', wavPath, segWav]);
      segWavs.push(segWav);
    }
    p.autoJob.stage = 'transcribing-offline'; p.autoJob.percent = 20; saveProjects(projects);
    const whisperResults = await transcribeOfflineFiles(segWavs);
    const allPieces = [];
    whisperResults.forEach((res, idx) => {
      const job = jobs[idx];
      for (const sg of (res.segments || [])) allPieces.push({ start: +(job.start + sg.start).toFixed(2), end: +(job.start + sg.end).toFixed(2), original: sg.text.trim() });
    });
    p.autoJob.completedSegments = totalSegments; p.autoJob.stage = 'translating-google'; p.autoJob.percent = 70; saveProjects(projects);
    const gt = await translateLinesWithGoogleOrFallback(allPieces);
    for (let i = 0; i < allPieces.length; i++) {
      const piece = allPieces[i];
      const burmese = gt.translations ? (gt.translations[i] || '') : '';
      const line = { ...piece, burmese, translationPending: !burmese, translationPendingReason: burmese ? null : (gt.error || 'translation failed') };
      p.autoJob.lines.push(line);
      if (burmese && !p.translationLines.some(l => l.start === line.start && l.original === line.original)) p.translationLines.push(line);
    }
    p.autoJob.translationPath = gt.path; p.autoJob.translationError = gt.error || null;
    p.autoJob.percent = 95; saveProjects(projects);
    p.autoJob.lines.sort((a, b) => a.start - b.start);
    p.translationLines.sort((a, b) => a.start - b.start);
    try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
    p.autoJob = { ...p.autoJob, status: 'done', stage: 'done', percent: 100, message: `Auto transcript + translation finished (Whisper offline tiny, translation via ${p.autoJob.translationPath || 'google'}).${p.autoJob.translationError ? ' Note: ' + p.autoJob.translationError : ''}`, finishedAt: new Date().toISOString() };
    saveProjects(projects);
  } catch (e) {
    p.autoJob = { ...p.autoJob, status: 'failed', stage: 'failed', message: `Auto transcript job failed: ${e.message}`, error: e.message, finishedAt: new Date().toISOString() };
    saveProjects(projects);
  }
}
function enqueueAutoJob(projectId) { // fire-and-forget; upload response must not wait for it
  setImmediate(() => runAutoJob(projectId).catch(() => {}));
}

const app = express();
app.use(cors()); app.use(express.json({ limit: '2mb' }));
const storage = multer.diskStorage({ destination: UPLOAD_DIR, filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_')}`) });
const upload = multer({ storage, limits: { fileSize: MAX_MB * 1024 * 1024 }, fileFilter: (req, file, cb) => { const ext = path.extname(file.originalname).toLowerCase(); cb(ALLOWED_EXT.includes(ext) ? null : new Error(`Unsupported file type: ${ext}. Allowed: ${ALLOWED_EXT.join(', ')}`), ALLOWED_EXT.includes(ext)); } });

app.get('/api/health', (req, res) => res.json({ ok: true, app: 'Sangbik', name: 'Sangbik', ttsVoices: TTS_VOICES, ttsDefaultProvider: defaultTtsProvider(), transcriptionProvider: 'whisper-offline-tiny', whisperModel: WHISPER_MODEL_NAME, whisperOfflineConfigured: whisperInstalled(), translationProvider: 'google-translate', ttsProviderDefault: 'edge-tts', edgeTtsConfigured: edgeTtsInstalled(), localConfigured: localTtsConfigured(), aiConfigured: !!(process.env.AI_PROVIDER && (process.env.AI_API_KEY || process.env.OPENAI_API_KEY || process.env.GEMINI_API_KEY)), transcriptionConfigured: whisperInstalled(), azureConfigured: !!(process.env.AZURE_SPEECH_KEY && process.env.AZURE_SPEECH_REGION), openaiTtsConfigured: !!process.env.OPENAI_API_KEY, geminiConfigured: !!process.env.GEMINI_API_KEY, geminiVoice: process.env.GEMINI_VOICE || 'Kore', geminiTtsModel: process.env.GEMINI_TTS_MODEL || 'gemini-2.5-flash-preview-tts', ttsProviders: ['edge-tts', 'local', 'azure', 'openai', 'gemini'], maxFileMb: MAX_MB }));

app.post('/api/projects', (req, res) => {
  const id = newId();
  projects[id] = { id, title: req.body.title || 'Untitled Recap', mode: ['recap', 'dubbing'].includes(req.body.mode) ? req.body.mode : 'recap', createdAt: new Date().toISOString(), status: 'created', metadata: null, file: null, scenes: [], analysis: null, script: '', scriptEstimate: null, translationLines: [], transcript: null, tts: null, export: null, urlNote: null, music: null, sceneOrder: null };
  saveProjects(projects); res.json(projects[id]);
});
app.get('/api/projects', (req, res) => res.json(Object.values(projects).sort((a, b) => b.createdAt.localeCompare(a.createdAt))));
app.get('/api/projects/:id', (req, res) => { const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' }); res.json(p); });

app.post('/api/upload', upload.single('video'), async (req, res) => {
  try {
    const projectId = req.body.projectId;
    const p = projects[projectId];
    if (!p) { fs.unlinkSync(req.file.path); return res.status(404).json({ error: 'Project not found' }); }
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    const meta = await probe(req.file.path); // throws if not a real media file
    p.file = { path: req.file.path, name: req.file.originalname, storedName: path.basename(req.file.path), size: req.file.size };
    p.metadata = meta; p.status = 'uploaded'; saveProjects(projects); enqueueAutoJob(p.id);
    res.json({ project: p, metadata: meta });
  } catch (e) { res.status(400).json({ error: `File validation failed: ${e.message}` }); }
});

// ---- Chunked + resumable upload ----
const chunkUpload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 6 * 1024 * 1024 } });
app.post('/api/upload-chunk', chunkUpload.single('chunk'), (req, res) => {
  const { uploadId, chunkIndex, totalChunks, fileName } = req.body || {};
  const idx = parseInt(chunkIndex, 10); const total = parseInt(totalChunks, 10);
  if (!validUploadId(uploadId) || Number.isNaN(idx) || Number.isNaN(total) || idx < 0 || idx >= total || !req.file) return res.status(400).json({ error: 'uploadId, chunkIndex, totalChunks and chunk file are required' });
  const meta = loadChunkMeta(uploadId) || { uploadId, fileName: fileName || 'video', totalChunks: total, received: [], totalBytes: 0, createdAt: new Date().toISOString() };
  if (meta.totalChunks !== total) return res.status(400).json({ error: 'totalChunks mismatch for this uploadId' });
  const chunkPath = path.join(CHUNK_DIR, `${uploadId}.part${idx}`);
  fs.writeFileSync(chunkPath, req.file.buffer);
  if (!meta.received.includes(idx)) { meta.received.push(idx); meta.received.sort((a, b) => a - b); }
  meta.totalBytes = meta.received.reduce((sum, i) => { try { return sum + fs.statSync(path.join(CHUNK_DIR, `${uploadId}.part${i}`)).size; } catch { return sum; } }, 0);
  saveChunkMeta(meta);
  res.json({ uploadId, received: meta.received, receivedCount: meta.received.length, totalChunks: total, totalBytes: meta.totalBytes });
});
app.get('/api/upload-status/:uploadId', (req, res) => {
  const meta = loadChunkMeta(req.params.uploadId);
  if (!meta) return res.json({ uploadId: req.params.uploadId, received: [], receivedCount: 0, totalChunks: 0, totalBytes: 0 });
  res.json({ uploadId: meta.uploadId, fileName: meta.fileName, received: meta.received, receivedCount: meta.received.length, totalChunks: meta.totalChunks, totalBytes: meta.totalBytes });
});
app.post('/api/upload-complete', async (req, res) => {
  try {
    const { uploadId, projectId } = req.body || {};
    const p = projects[projectId];
    if (!validUploadId(uploadId)) return res.status(400).json({ error: 'uploadId is required' });
    if (!p) return res.status(404).json({ error: 'Project not found' });
    const meta = loadChunkMeta(uploadId);
    if (!meta) return res.status(404).json({ error: 'Upload session not found — re-upload the missing chunks' });
    const missing = [];
    for (let i = 0; i < meta.totalChunks; i++) if (!meta.received.includes(i)) missing.push(i);
    if (missing.length) return res.status(400).json({ error: 'Missing chunks', missing });
    const ext = path.extname(meta.fileName || '').toLowerCase();
    if (!ALLOWED_EXT.includes(ext)) return res.status(400).json({ error: `Unsupported file type: ${ext}. Allowed: ${ALLOWED_EXT.join(', ')}` });
    const safeName = String(meta.fileName).replace(/[^a-zA-Z0-9._-]/g, '_');
    const dest = path.join(UPLOAD_DIR, `${Date.now()}-${safeName}`);
    const ws = fs.createWriteStream(dest);
    let assembled = 0;
    for (let i = 0; i < meta.totalChunks; i++) { const buf = fs.readFileSync(path.join(CHUNK_DIR, `${uploadId}.part${i}`)); ws.write(buf); assembled += buf.length; }
    await new Promise((resolve, reject) => ws.end(err => err ? reject(err) : resolve()));
    // Verify assembled size against chunk bytes, then ffprobe (throws if not real media).
    if (assembled !== meta.totalBytes) return res.status(400).json({ error: `Assembled size mismatch: ${assembled} vs chunk total ${meta.totalBytes}` });
    const fileMeta = await probe(dest);
    p.file = { path: dest, name: meta.fileName, storedName: path.basename(dest), size: assembled };
    p.metadata = fileMeta; p.status = 'uploaded'; saveProjects(projects);
    // Cleanup chunks (best effort, after successful assembly).
    for (let i = 0; i < meta.totalChunks; i++) { try { fs.unlinkSync(path.join(CHUNK_DIR, `${uploadId}.part${i}`)); } catch {} }
    try { fs.unlinkSync(chunkMetaPath(uploadId)); } catch {}
    enqueueAutoJob(p.id);
    res.json({ project: p, metadata: fileMeta, job: p.autoJob || { status: 'queued' } });
  } catch (e) { res.status(400).json({ error: `Assembly/validation failed: ${e.message}` }); }
});
app.get('/api/projects/:id/job-status', (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' });
  res.json(p.autoJob || { status: 'not-started', stage: 'not-started', percent: 0, completedSegments: 0, totalSegments: 0, lines: [], message: 'No auto job yet — upload a video to start one, or use Re-run auto transcript.' });
});
app.post('/api/projects/:id/auto-job', (req, res) => {
  const p = projects[req.params.id]; if (!p || !p.file) return res.status(400).json({ error: 'Project or video not found' });
  if (p.autoJob && p.autoJob.status === 'running') return res.json({ job: p.autoJob });
  enqueueAutoJob(p.id);
  res.json({ job: { status: 'queued', stage: 'queued', percent: 0, message: 'Auto transcript job re-queued.' } });
});



// ---- Scene order: reorder scene segments; export respects it (see export) ----
app.put('/api/projects/:id/scene-order', (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found — create a project first.' });
  const order = req.body?.order;
  if (!Array.isArray(order) || order.length < 2 || !order.every(n => Number.isInteger(n) && n >= 0)) return res.status(400).json({ error: 'order must be an array of at least 2 scene-segment indices, e.g. [1,0,2].' });
  if (new Set(order).size !== order.length) return res.status(400).json({ error: 'Scene order must not repeat a segment index.' });
  if (p.scenes && p.scenes.length && p.metadata && p.metadata.duration) {
    const nSeg = getSceneSegments(p.scenes, 0, p.metadata.duration).length;
    if (order.length !== nSeg || Math.max(...order) >= nSeg) return res.status(400).json({ error: `Scene order does not match this video: it has ${nSeg} scene segments (indices 0-${nSeg - 1}), but you sent [${order.join(', ')}]. Re-run Analyse, check the segment list, then save again.` });
  }
  p.sceneOrder = order; saveProjects(projects);
  res.json({ project: p, message: `Scene order saved: [${order.join(', ')}]. It is applied at export; if it does not match the current scenes it is ignored with a note in export-status.` });
});
app.get('/api/fonts', (req, res) => res.json({ fonts: FONT_WHITELIST, default: 'Noto Sans Myanmar', note: 'For Burmese text use Noto Sans Myanmar or Noto Serif Myanmar. DejaVu Sans and Liberation Sans do not contain Myanmar glyphs, so Burmese subtitles in those fonts may render as boxes.' }));

// ---- Background music: upload per project, validated with ffprobe ----
const musicUpload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 200 * 1024 * 1024 } });
app.post('/api/projects/:id/music', musicUpload.single('music'), async (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found — create a project first, then upload music to it.' });
  if (!req.file) return res.status(400).json({ error: 'No music file uploaded. Choose an MP3, WAV or M4A audio file.' });
  const ext = path.extname(req.file.originalname).toLowerCase();
  if (!ALLOWED_MUSIC_EXT.includes(ext)) return res.status(400).json({ error: `Unsupported music file type: ${ext}. Allowed music: ${ALLOWED_MUSIC_EXT.join(', ')}` });
  try {
    const dest = path.join(UPLOAD_DIR, `${p.id}-music${ext}`);
    fs.writeFileSync(dest, req.file.buffer);
    const meta = await probe(dest);
    if (!meta.audioCodec) { try { fs.unlinkSync(dest); } catch {} return res.status(400).json({ error: 'That file has no audio stream — upload a real MP3, WAV or M4A audio file, not a video or empty file.' }); }
    p.music = { path: dest, name: req.file.originalname, storedName: path.basename(dest), size: req.file.size, format: meta.format, duration: meta.duration, audioCodec: meta.audioCodec };
    saveProjects(projects);
    res.json({ project: p, music: p.music, message: `Music saved: ${req.file.originalname} (${meta.duration}s, ${meta.audioCodec}). It will be mixed in the next export — set Music volume / fades in the Video Editor.` });
  } catch (e) { res.status(400).json({ error: `Music validation failed (ffprobe could not read this file): ${e.message}` }); }
});

app.use((err, req, res, next) => { if (err) return res.status(400).json({ error: err.message }); next(); });

app.post('/api/projects/:id/url', async (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' });
  const { url, confirmed } = req.body || {};
  if (!url) return res.status(400).json({ error: 'URL is required' });
  if (!confirmed) return res.status(400).json({ error: 'URL download is not performed unless you confirm you own the content or have permission to process it. Set confirmed=true to proceed.' });
  const ext = path.extname(new URL(url).pathname).toLowerCase();
  if (!ALLOWED_EXT.includes(ext)) return res.status(400).json({ error: 'URL download is not performed: only direct media URLs ending in ' + ALLOWED_EXT.join(', ') + ' are attempted. Platform pages, playlists and DRM-protected links are never downloaded.' });
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(30000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const buf = Buffer.from(await r.arrayBuffer());
    if (buf.length > MAX_MB * 1024 * 1024) return res.status(400).json({ error: 'Remote file exceeds MAX_FILE_MB limit' });
    const filePath = path.join(UPLOAD_DIR, `${Date.now()}-remote${ext}`);
    fs.writeFileSync(filePath, buf);
    const meta = await probe(filePath);
    p.file = { path: filePath, name: path.basename(url), storedName: path.basename(filePath), size: buf.length, sourceUrl: url };
    p.metadata = meta; p.status = 'uploaded'; saveProjects(projects); enqueueAutoJob(p.id);
    res.json({ project: p, metadata: meta });
  } catch (e) { res.status(400).json({ error: `URL download failed (not attempted for DRM/platform content): ${e.message}` }); }
});

app.post('/api/projects/:id/analyze', async (req, res) => {
  const p = projects[req.params.id]; if (!p || !p.file) return res.status(400).json({ error: 'Project or uploaded video not found' });
  try {
    const { err } = await run('ffmpeg', ['-i', p.file.path, '-vf', "scale=320:-1,select='gt(scene,0.4)',showinfo", '-f', 'null', '-']);
    const scenes = [];
    for (const m of err.matchAll(/pts_time:([\d.]+)/g)) { const t = parseFloat(m[1]); if (!scenes.length || t - scenes[scenes.length - 1] > 1) scenes.push(+t.toFixed(2)); }
    if (!scenes.length) scenes.push(0);
    p.scenes = scenes;
    const notes = req.body?.notes || ''; const spoilerMode = req.body?.spoilerMode || 'full';
    let script = null, aiUsed = false, aiError = null;
    try { script = await aiDraft(p, scenes, notes, spoilerMode); aiUsed = !!script; } catch (e) { aiError = e.message; }
    if (!script) script = structuredDraft(p, scenes, notes, spoilerMode);
    p.analysis = { scenes, sceneCount: scenes.length, spoilerMode, notes, aiUsed, aiError, configMessage: aiUsed ? null : 'No AI API key configured (set AI_PROVIDER + key in server/.env). Returned a structured draft built only from scene timestamps and your notes — edit it before generating the final script. No AI analysis was performed.' };
    p.script = script; p.scriptEstimate = estimate(script, 140); p.status = 'analyzed'; saveProjects(projects);
    res.json({ project: p, analysis: p.analysis, script });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.put('/api/projects/:id/script', (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' });
  const { script, wpm } = req.body || {};
  if (typeof script !== 'string') return res.status(400).json({ error: 'script is required' });
  p.script = script;
  p.scriptEstimate = { ...estimate(script, wpm || 140), minutesShortTarget: 2, minutesMediumTarget: 5, minutesLongTarget: 10 };
  saveProjects(projects); res.json({ project: p });
});

app.put('/api/projects/:id/translation', (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' });
  const lines = normaliseLines(req.body?.lines);
  p.translationLines = lines; p.status = p.status === 'created' ? 'translated' : p.status; saveProjects(projects);
  res.json({ project: p, lines });
});
app.post('/api/projects/:id/transcribe', async (req, res) => {
  const p = projects[req.params.id]; if (!p || !p.file) return res.status(400).json({ error: 'Project or video not found' });
  try {
    const tmpDir = fs.mkdtempSync(path.join(OUT_DIR, 'trans-'));
    const wavPath = path.join(tmpDir, 'audio.wav');
    await run('ffmpeg', ['-y', '-i', p.file.path, '-vn', '-ac', '1', '-ar', '16000', wavPath]);
    const duration = p.metadata?.duration || (await probe(p.file.path)).duration || 0;
    const totalSegments = Math.max(1, Math.ceil(duration / 60));
    const segWavs = [];
    for (let i = 0; i < totalSegments; i++) {
      const segWav = path.join(tmpDir, `seg${i}.wav`);
      await run('ffmpeg', ['-y', '-ss', String(i * 60), '-t', String(Math.min(60, duration - i * 60 || 60)), '-i', wavPath, segWav]);
      segWavs.push(segWav);
    }
    const results = await transcribeOfflineFiles(segWavs);
    const lines = [];
    results.forEach((res, idx) => { for (const sg of (res.segments || [])) lines.push({ start: +(idx * 60 + sg.start).toFixed(2), end: +(idx * 60 + sg.end).toFixed(2), original: sg.text.trim(), burmese: '' }); });
    const language = results[0]?.language || null;
    try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
    p.transcript = { text: lines.map(l => l.original).join(' '), segments: lines, provider: 'whisper-offline-tiny', language, model: WHISPER_MODEL_NAME };
    if (!p.translationLines.length) p.translationLines = lines;
    saveProjects(projects); res.json({ project: p, transcript: p.transcript });
  } catch (e) { res.status(500).json({ error: `Offline Whisper transcription failed: ${e.message}` }); }
});
// Translate transcript segments to Burmese via Google Translate (batch, no key).
// Fallback is AI translation only if Google fails and an AI key is configured;
// if both fail, no translation is invented and the exact error is returned.
app.post('/api/projects/:id/translate-google', async (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' });
  const segs = (p.transcript?.segments || []).filter(x => x.original);
  if (!segs.length) return res.status(400).json({ error: 'No transcript segments to translate — run Transcribe first.' });
  const gt = await translateLinesWithGoogleOrFallback(segs);
  if (!gt.translations) return res.status(502).json({ error: gt.error });
  const lines = segs.map((sg, i) => ({ start: sg.start, end: sg.end, original: sg.original, burmese: gt.translations[i] || '' }));
  p.translationLines = lines; p.translationPath = gt.path; p.translationGoogleLog = gt.googleLog || null;
  saveProjects(projects);
  res.json({ project: p, lines, path: gt.path, googleLog: gt.googleLog || null, note: gt.error || null });
});
app.post('/api/projects/:id/tts', async (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' });
  const provider = (req.body?.provider || defaultTtsProvider()).toLowerCase();
  const source = req.body?.source || 'script';
  let text = '';
  if (source === 'translation') {
    if (!p.translationLines?.length) return res.status(400).json({ error: 'No Burmese translation lines saved yet. Save lines in the Translation Editor first.' });
    text = p.translationLines.map(l => l.burmese).join(' ');
  } else {
    if (!p.script) return res.status(400).json({ error: 'Project or script not found' });
    text = p.script;
  }
  const segments = splitText(text);
  const tmpDir = fs.mkdtempSync(path.join(OUT_DIR, 'tts-'));
  try {
    const files = [];
    const speed = parseFloat(req.body?.speed || 1);
    const pauseSec = parseFloat(req.body?.pause || 0);
    if (provider === 'azure') {
      // PRIMARY: paid Azure Speech, not the free Edge endpoint blocked in Studio99.
      const key = process.env.AZURE_SPEECH_KEY; const region = process.env.AZURE_SPEECH_REGION;
      if (!key || !region) return res.status(501).json({ error: 'Azure Speech is not configured: set BOTH AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in server/.env (server-side only, never in frontend code) and restart the server. No Azure audio was generated.', downloadScript: `/api/projects/${p.id}/download/script`, voicesConfigured: TTS_VOICES });
      const voice = req.body?.voice || TTS_VOICES[0];
      if (!TTS_VOICES.includes(voice)) return res.status(400).json({ error: `Voice not configured. Configured voices (Nilar/Thiha first): ${TTS_VOICES.join(', ')}` });
      for (let i = 0; i < segments.length; i++) { const segFile = path.join(tmpDir, `seg${i}.mp3`); await azureTtsSegment(segments[i], voice, speed, req.body?.pitch, key, region, segFile); files.push(segFile); }
      const outMp3 = path.join(OUT_DIR, `${p.id}-narration.mp3`);
      await concatMp3s(files, outMp3, pauseSec, tmpDir);
      fs.rmSync(tmpDir, { recursive: true, force: true });
      const meta = await probe(outMp3);
      p.tts = { provider: 'azure', voice, source, segments: segments.length, file: outMp3, duration: meta.duration }; saveProjects(projects);
      return res.json({ project: p, tts: p.tts });
    }
    if (provider === 'openai') {
      // SECONDARY / EXPERIMENTAL for Burmese: OpenAI TTS reads Burmese text but has no native Burmese voice catalogue.
      const key = process.env.OPENAI_API_KEY;
      if (!key) return res.status(501).json({ error: 'OpenAI TTS is not configured: set OPENAI_API_KEY in server/.env and restart the server. Labelled experimental for Burmese — no native Burmese voice quality is promised. No audio was generated.', downloadScript: `/api/projects/${p.id}/download/script` });
      const oVoice = req.body?.openaiVoice || process.env.OPENAI_TTS_VOICE || 'alloy';
      for (let i = 0; i < segments.length; i++) {
        const r = await fetch('https://api.openai.com/v1/audio/speech', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body: JSON.stringify({ model: 'tts-1', voice: oVoice, input: segments[i], response_format: 'mp3' }) });
        if (!r.ok) throw new Error(`OpenAI TTS HTTP ${r.status}: ${(await r.text()).slice(0, 500)}`);
        const segFile = path.join(tmpDir, `seg${i}.mp3`); fs.writeFileSync(segFile, Buffer.from(await r.arrayBuffer())); files.push(segFile);
      }
      const outMp3 = path.join(OUT_DIR, `${p.id}-narration.mp3`);
      await concatMp3s(files, outMp3, pauseSec, tmpDir);
      fs.rmSync(tmpDir, { recursive: true, force: true });
      const meta = await probe(outMp3);
      p.tts = { provider: 'openai', voice: oVoice, source, segments: segments.length, file: outMp3, duration: meta.duration }; saveProjects(projects);
      return res.json({ project: p, tts: p.tts });
    }
    if (provider === 'local') {
      // NO-KEY LOCAL TTS: espeak-ng, Burmese voice "my". No account, no key,
      // works offline. Honest label: robotic voice, quality lower than Azure.
      // Speed slider 0.5-2.0 maps to espeak-ng words-per-minute (default 175).
      if (!localTtsConfigured()) return res.status(501).json({ error: 'Local TTS needs espeak-ng on the server: sudo apt-get install espeak-ng, then restart the server. No local audio was generated.', downloadScript: `/api/projects/${p.id}/download/script` });
      const wpm = Math.min(450, Math.max(80, Math.round(175 * speed)));
      for (let i = 0; i < segments.length; i++) {
        const wavFile = path.join(tmpDir, `seg${i}.wav`);
        const segFile = path.join(tmpDir, `seg${i}.mp3`);
        await run('espeak-ng', ['-v', 'my', '-s', String(wpm), '-w', wavFile, segments[i]]);
        const st = fs.statSync(wavFile);
        if (!st.size) throw new Error('espeak-ng produced an empty WAV for a segment — no audio was generated');
        await run('ffmpeg', ['-y', '-i', wavFile, '-c:a', 'libmp3lame', segFile]);
        files.push(segFile);
      }
      const outMp3 = path.join(OUT_DIR, `${p.id}-narration.mp3`);
      await concatMp3s(files, outMp3, pauseSec, tmpDir);
      fs.rmSync(tmpDir, { recursive: true, force: true });
      const meta = await probe(outMp3);
      p.tts = { provider: 'local', voice: 'local-my (espeak-ng)', source, segments: segments.length, file: outMp3, duration: meta.duration }; saveProjects(projects);
      return res.json({ project: p, tts: p.tts });
    }
    if (provider === 'gemini') {
      const key = process.env.GEMINI_API_KEY;
      if (!key) return res.status(501).json({ error: 'Gemini TTS is not configured: set GEMINI_API_KEY in server/.env (never in frontend code) and restart the server. Optionally set GEMINI_VOICE and GEMINI_TTS_MODEL. No Gemini audio was generated.', downloadScript: `/api/projects/${p.id}/download/script` });
      const voice = req.body?.geminiVoice || process.env.GEMINI_VOICE || 'Kore';
      const model = process.env.GEMINI_TTS_MODEL || 'gemini-2.5-flash-preview-tts';
      for (let i = 0; i < segments.length; i++) {
        const segFile = path.join(tmpDir, `seg${i}.mp3`);
        await geminiTtsSegment(segments[i], voice, model, key, segFile);
        files.push(segFile);
      }
      const listFile = path.join(tmpDir, 'list.txt');
      fs.writeFileSync(listFile, files.map(f => `file '${f}'`).join('\n'));
      const outMp3 = path.join(OUT_DIR, `${p.id}-narration.mp3`);
      await run('ffmpeg', ['-y', '-f', 'concat', '-safe', '0', '-i', listFile, '-c', 'copy', outMp3]);
      fs.rmSync(tmpDir, { recursive: true, force: true });
      const meta = await probe(outMp3);
      p.tts = { provider: 'gemini', voice, model, source, segments: segments.length, file: outMp3, duration: meta.duration }; saveProjects(projects);
      return res.json({ project: p, tts: p.tts });
    }
    // Edge TTS (Sangbik default, 2026-10-26): FIRST try the current
    // msedgeservices.com websocket endpoint via server/edge_ms.py — this is
    // the endpoint that actually works in the sandbox (2026-10-26 test:
    // Nilar "မင်္ဂလာပါ" -> 11,232 bytes, 1.872s). Token handling comes only
    // from the installed open-source edge-tts package itself (its public
    // client token constant + Sec-MS-GEC generator); no token is hard-coded
    // here and none was taken from any other app. If that fails, fall back
    // to the old edge-tts CLI (bing endpoint, known to fail in the sandbox
    // with SSL/proxy errors). A 0-byte result is always an error, never audio.
    const voice = req.body?.voice || TTS_VOICES[0];
    if (!TTS_VOICES.includes(voice)) return res.status(400).json({ error: `Voice not configured. Configured voices (Nilar/Thiha first): ${TTS_VOICES.join(', ')}` });
    const ratePct = Math.round((speed - 1) * 100);
    const rateStr = `${ratePct >= 0 ? '+' : ''}${ratePct}%`;
    const pitch = req.body?.pitch;
    const edgePy = fs.existsSync(WHISPER_PYTHON) ? WHISPER_PYTHON : 'python3';
    for (let i = 0; i < segments.length; i++) {
      const segFile = path.join(tmpDir, `seg${i}.mp3`);
      let edgeErr = null;
      try {
        const { out } = await run(edgePy, [path.join(__dirname, 'edge_ms.py'), segments[i], voice, segFile, rateStr]);
        if (!/AUDIO_BYTES=([1-9][0-9]*)/.test(out) || !fs.existsSync(segFile) || !fs.statSync(segFile).size) throw new Error(`msedgeservices returned no audio (${(out || '').slice(-300)})`);
      } catch (e) { edgeErr = e.message; }
      if (!fs.existsSync(segFile) || !fs.statSync(segFile).size) {
        // Second attempt: legacy edge-tts CLI (bing endpoint).
        try {
          const args = ['-m', 'edge_tts', '--voice', voice, '--rate', rateStr, '--write-media', segFile, '--text', segments[i]];
          if (pitch) args.push('--pitch', pitch);
          if (process.env.HTTPS_PROXY) args.push('--proxy', process.env.HTTPS_PROXY);
          await run('python3', args);
        } catch (e) { edgeErr = `${edgeErr} | legacy edge-tts CLI also failed: ${e.message}`; }
      }
      if (!fs.existsSync(segFile) || !fs.statSync(segFile).size) throw new Error(`Edge TTS failed (msedgeservices + legacy CLI): ${edgeErr} — no audio was generated. Local fallback is used only by the Workflow, with a clear log line, never presented as Edge audio.`);
      files.push(segFile);
    }
    const listFile = path.join(tmpDir, 'list.txt');
    fs.writeFileSync(listFile, files.map(f => `file '${f}'`).join('\n'));
    const outMp3 = path.join(OUT_DIR, `${p.id}-narration.mp3`);
    await run('ffmpeg', ['-y', '-f', 'concat', '-safe', '0', '-i', listFile, '-c', 'copy', outMp3]);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    const meta = await probe(outMp3);
    p.tts = { provider: 'edge-tts', voice, source, segments: segments.length, file: outMp3, duration: meta.duration }; saveProjects(projects);
    res.json({ project: p, tts: p.tts });
  } catch (e) { try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {} res.status(500).json({ error: `TTS failed: ${e.message}` }); }
});

app.post('/api/projects/:id/export', async (req, res) => {
  const p = projects[req.params.id]; if (!p || !p.file) return res.status(400).json({ error: 'Project or video not found' });
  const o = req.body || {};
  const dubbing = !!o.dubbingPreset;
  const start = parseFloat(o.trimStart || 0); const end = parseFloat(o.trimEnd || p.metadata?.duration || 0);
  const aspect = o.aspect || (dubbing ? '9:16' : '16:9'); const quality = o.quality || '720p';
  const sub = o.subtitles || {}; const narrVol = parseFloat(o.narrationVolume ?? 1); const origVol = parseFloat(o.originalVolume ?? (dubbing ? 0 : 0.5));
  const subtitlesOn = o.subtitlesOn !== false;
  const subtitlesFrom = o.subtitlesFrom || (dubbing ? 'translation' : 'script');
  const blurMode = ['none', 'full', 'background', 'custom'].includes(o.blurMode) ? o.blurMode : 'none';
  const br0 = o.blurRegion && typeof o.blurRegion === 'object' ? o.blurRegion : {};
  const clamp01 = (v, d) => { const n = Number(v); return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : d; };
  const blurX = clamp01(o.blurX ?? br0.x, 0), blurY = clamp01(o.blurY ?? br0.y, 0.60), blurW = clamp01(o.blurW ?? br0.w, 1), blurH = clamp01(o.blurH ?? br0.h, 0.40);
  const blurRegion = { x: blurX, y: blurY, w: blurW, h: blurH };
  const blurStrength = ['light', 'medium', 'strong'].includes(o.blurStrength) ? o.blurStrength : 'medium';
  const blurSigma = BLUR_SIGMA[blurStrength];
  const zoomEvery5s = !!o.zoomEvery5s;
  const zoomStrength = ['light', 'medium', 'strong'].includes(o.zoomStrength) ? o.zoomStrength : 'medium';
  const zoomMax = ZOOM_MAX[zoomStrength];
  const transition = ['none', 'fade', 'crossfade'].includes(o.transition) ? o.transition : 'none';
  const transitionDurationRaw = parseFloat(o.transitionDuration || 0.5);
  const transitionDuration = Math.min(2, Math.max(0, Number.isFinite(transitionDurationRaw) ? transitionDurationRaw : 0.5));
  const panMode = ['none', 'left-right', 'right-left'].includes(o.panMode) ? o.panMode : 'none';
  const subFont = (o.subtitles && o.subtitles.font) ? String(o.subtitles.font) : 'Noto Sans Myanmar';
  if (!FONT_WHITELIST.includes(subFont)) return res.status(400).json({ error: `Subtitle font not available on this server: ${subFont}. Available fonts: ${FONT_WHITELIST.join(', ')} (for Burmese text use Noto Sans Myanmar).` });
  const enhanceClarity = !!o.enhanceClarity;
  const clarityStrength = ['light', 'medium', 'strong'].includes(o.clarityStrength) ? o.clarityStrength : 'medium';
  const clarityUnsharp = CLARITY_UNSHARP[clarityStrength];
  const clarityDenoise = CLARITY_DENOISE[clarityStrength];
  // Music: only used if a music file was uploaded via POST /api/projects/:id/music.
  const musicVol = Math.min(1.5, Math.max(0, parseFloat(o.musicVolume ?? 0.3)));
  const musicFadeIn = Math.max(0, parseFloat(o.musicFadeIn || 0));
  const musicFadeOut = Math.max(0, parseFloat(o.musicFadeOut || 0));
  const hasMusic = !!(p.music && p.music.path && fs.existsSync(p.music.path));
  // Scene reorder (validated here; ignored at export with a note if it no longer matches).
  const rawOrder = Array.isArray(p.sceneOrder) ? p.sceneOrder : null;
  const sceneSegs = getSceneSegments(p.scenes || [], start, end);
  const sceneOrderValid = !!(rawOrder && rawOrder.length === sceneSegs.length && [...rawOrder].sort((a, b) => a - b).every((v, i) => v === i));
  const sceneOrderApplied = !!(sceneOrderValid && sceneSegs.length >= 2);
  const sceneOrderNote = rawOrder && !sceneOrderApplied ? 'Scene order ignored: it does not match the current scene segments (re-run Analyse, then save the order again).' : null;
  let segments;
  if (subtitlesFrom === 'translation' && p.translationLines?.length) segments = p.translationLines.map(l => ({ start: l.start, end: l.end, text: l.burmese }));
  else segments = scriptToSegments(p.script || 'No script', Math.max(end - start, 1));
  if (sceneOrderApplied) {
    // Retime subtitles onto the reordered timeline (each line keeps its offset inside its old segment).
    // A line spanning two old segments is mapped by its start and end independently, so its
    // duration on the new timeline may differ — documented here and in README.
    segments = segments.map(sg => ({ ...sg, start: mapTimeForOrder(rawOrder, sceneSegs, sg.start), end: mapTimeForOrder(rawOrder, sceneSegs, sg.end) })).filter(sg => sg.end > sg.start);
  }
  const srtPath = writeSrt(p, segments);
  const vttPath = writeVtt(p, segments, subtitlesFrom === 'translation' ? '-translation' : '');
  p.export = { status: 'processing', progress: 'started', aspect, quality, srt: srtPath, vtt: vttPath, dubbingPreset: dubbing, subtitlesFrom, subtitlesOn, blurMode, blurStrength, blurSigma, blurRegion, zoomEvery5s, zoomStrength, zoomMax, enhanceClarity, clarityStrength, clarityUnsharp, musicVolume: musicVol, musicFadeIn, musicFadeOut, musicFile: hasMusic ? p.music.name : null, sceneOrder: sceneOrderApplied ? rawOrder : null, sceneOrderApplied, sceneOrderNote, transition, transitionDuration, transitionApplied: null, transitionNote: sceneOrderNote, panMode, subtitleFont: subFont };
  saveProjects(projects);
  (async () => {
    try {
      // Zoom every 5 seconds (visual effect only — NOT a copyright-avoidance feature):
      // pass 1 splits the trimmed range into 5s segments, crops each segment by an
      // alternating zoom factor (1.0x, zoomMax, 1.0x, ...) centred, scales back to
      // source size, and concats them. If blur is also enabled, pass 2 is the main
      // export below, which applies the blur filter to this intermediate file.
      let inputPath = p.file.path; let effStart = start; let effEnd = end;
      // Segment concat helper: plain concat, or xfade+acrossfade chain when transition=crossfade.
      async function concatOrXfade(segFiles, segDurs, outPath, hasAudio) {
        if (transition === 'crossfade' && segFiles.length >= 2) {
          const td = Math.min(transitionDuration, Math.min(...segDurs) * 0.9);
          if (td <= 0) throw new Error('Crossfade duration is too small for these segments.');
          const argsX = []; segFiles.forEach(f => argsX.push('-i', f));
          let vFilter = ''; let cur = '0:v'; let offset = segDurs[0] - td;
          for (let i = 1; i < segFiles.length; i++) { const out = `v${i}`; vFilter += (vFilter ? ';' : '') + `[${cur}][${i}:v]xfade=transition=fade:duration=${td}:offset=${offset}[${out}]`; cur = out; offset += segDurs[i] - td; }
          let aFilter = ''; let aCur = '0:a';
          if (hasAudio) { for (let i = 1; i < segFiles.length; i++) { const out = `a${i}`; aFilter += (aFilter ? ';' : '') + `[${aCur}][${i}:a]acrossfade=d=${td}[${out}]`; aCur = out; } }
          const fc = vFilter + (hasAudio && aFilter ? ';' + aFilter : '');
          const a2 = [...argsX, '-filter_complex', fc, '-map', `[${cur}]`];
          if (hasAudio) a2.push('-map', `[${aCur}]`, '-c:a', 'aac'); else a2.push('-an');
          a2.push('-c:v', 'libx264', '-preset', 'veryfast', outPath);
          await run('ffmpeg', a2); return outPath;
        }
        const listFile = outPath + '.list.txt';
        fs.writeFileSync(listFile, segFiles.map(f => `file '${f}'`).join('\n'));
        await run('ffmpeg', ['-y', '-f', 'concat', '-safe', '0', '-i', listFile, '-c', 'copy', outPath]);
        try { fs.unlinkSync(listFile); } catch {}
        return outPath;
      }
      // Scene reorder pass (runs before zoom): cut each scene segment in the saved order.
      if (sceneOrderApplied) {
        const reorderDir = fs.mkdtempSync(path.join(OUT_DIR, 'reorder-'));
        const segFiles = []; const segDurs = [];
        const hasAudioR = !!p.metadata?.audioCodec;
        for (const idx of rawOrder) {
          const seg = sceneSegs[idx]; const segFile = path.join(reorderDir, `seg${idx}.mp4`);
          const argsSeg = ['-y', '-ss', String(seg.start), '-t', String(seg.duration), '-i', p.file.path, '-c:v', 'libx264', '-preset', 'veryfast'];
          if (hasAudioR) argsSeg.push('-c:a', 'aac'); else argsSeg.push('-an');
          argsSeg.push(segFile); await run('ffmpeg', argsSeg); segFiles.push(segFile); segDurs.push(seg.duration);
        }
        const reorderOut = path.join(OUT_DIR, `${p.id}-reordered.mp4`);
        await concatOrXfade(segFiles, segDurs, reorderOut, hasAudioR);
        try { fs.rmSync(reorderDir, { recursive: true, force: true }); } catch {}
        inputPath = reorderOut; effStart = 0; effEnd = segDurs.reduce((a, b) => a + b, 0);
      }
      if (zoomEvery5s) {
        const totalDur = Math.max(end - start, 0.1);
        const segCount = Math.ceil(totalDur / 5);
        const zoomDir = fs.mkdtempSync(path.join(OUT_DIR, 'zoom-'));
        const segFiles = [];
        const hasAudio = !!p.metadata?.audioCodec;
        for (let i = 0; i < segCount; i++) {
          const segStart = start + i * 5; const segDur = Math.min(5, end - segStart);
          if (segDur <= 0) break;
          const z = (i % 2 === 0) ? 1.0 : zoomMax;
          const segFile = path.join(zoomDir, `seg${i}.mp4`);
          const srcW = p.metadata?.width; const srcH = p.metadata?.height;
          const vf = z === 1.0 ? 'null' : `crop=iw/${z}:ih/${z},scale=${srcW || 'iw'}:${srcH || 'ih'}`;
          const argsSeg = ['-y', '-ss', String(segStart), '-t', String(segDur), '-i', p.file.path, '-vf', vf, '-c:v', 'libx264', '-preset', 'veryfast'];
          if (hasAudio) argsSeg.push('-c:a', 'aac'); else argsSeg.push('-an');
          argsSeg.push(segFile);
          await run('ffmpeg', argsSeg);
          segFiles.push(segFile);
        }
        const zoomOut = path.join(OUT_DIR, `${p.id}-zoom.mp4`);
        const zoomSegDurs = [];
        { const totalDurZ = Math.max(end - start, 0.1); for (let i = 0; i < segCount; i++) zoomSegDurs.push(Math.min(5, end - (start + i * 5))); }
        await concatOrXfade(segFiles, zoomSegDurs, zoomOut, hasAudio);
        try { fs.rmSync(zoomDir, { recursive: true, force: true }); } catch {}
        inputPath = zoomOut; effStart = 0; effEnd = totalDur;
      }
      const effDuration = Math.max(effEnd - effStart, 0.1);
      const transitionApplied = transition === 'crossfade' ? ((sceneOrderApplied || zoomEvery5s) ? 'crossfade' : 'fade (fallback: crossfade needs scene reorder or zoom segments)') : transition;
      const transitionNote = transition === 'crossfade' && !(sceneOrderApplied || zoomEvery5s) ? 'Crossfade requested but no segment pass is active (no scene reorder, no zoom) — fade in/out was used instead.' : (sceneOrderNote || null);
      const dims = quality === '1080p' ? { '16:9': [1920, 1080], '9:16': [1080, 1920], '1:1': [1080, 1080] } : { '16:9': [1280, 720], '9:16': [720, 1280], '1:1': [720, 720] };
      const [W, H] = dims[aspect] || dims['16:9'];
      const ar = aspect === '9:16' ? '9/16' : aspect === '1:1' ? '1/1' : '16/9';
      const align = sub.position === 'top' ? 8 : sub.position === 'middle' ? 5 : 2;
      const color = (sub.color || '#FFFFFF').replace('#', '');
      const bgr = color.length === 6 ? color.slice(4, 6) + color.slice(2, 4) + color.slice(0, 2) : 'FFFFFF';
      const style = `FontName=${subFont},FontSize=${sub.size || 24},PrimaryColour=&H${bgr}&,Alignment=${align}`;
      const scaleFlags = (enhanceClarity && quality === '1080p') ? ':flags=lanczos' : '';
      const baseVf = `crop='if(gt(iw/ih,${ar}),ih*${ar},iw)':'if(gt(iw/ih,${ar}),ih,iw/(${ar}))',scale=${W}:${H}${scaleFlags}`;
      // FILTER ORDER (documented): aspect crop/scale -> blur (if any) -> clarity sharpen+denoise -> subtitles burn-in.
      // Clarity is sharpen/denoise only — it cannot restore detail that was never recorded.
      const clarityFilter = enhanceClarity ? `,hqdn3d=${clarityDenoise},unsharp=${clarityUnsharp}` : '';
      const fadeDur = transitionApplied.startsWith('fade') ? Math.min(transitionDuration, effDuration / 2) : 0;
      const fadeFilter = fadeDur > 0 ? `,fade=t=in:st=0:d=${fadeDur},fade=t=out:st=${Math.max(0, effDuration - fadeDur)}:d=${fadeDur}` : '';
      // Pan: scale to 1.2x target, then crop a moving window (left->right or right->left).
      // Crop x/y expressions are evaluated per frame (t = time in seconds), so this is a real slow pan.
      const panExpr = panMode === 'left-right' ? `min(t/${effDuration},1)` : `1-min(t/${effDuration},1)`;
      const panVf = panMode === 'none' ? null : `scale=${Math.round(W * 1.2)}:${Math.round(H * 1.2)}${scaleFlags},crop=${W}:${H}:x='(iw-ow)*${panExpr}':y='(ih-oh)/2'`;
      const subFilter = subtitlesOn ? `,subtitles='${srtPath.replace(/'/g, "\\'")}':force_style='${style}'` : '';
      const outMp4 = path.join(OUT_DIR, `${p.id}-recap.mp4`);
      const args = ['-y', '-ss', String(effStart), '-to', String(effEnd), '-i', inputPath];
      const inputs = [p.file.path];
      let filter;
      if (blurMode === 'full') {
        // Blur the whole frame (gblur sigma: light 5 / medium 15 / strong 30).
        filter = `[0:v]${panVf || baseVf},gblur=sigma=${blurSigma}${clarityFilter}${fadeFilter}${subFilter}[vout]`;
      } else if (blurMode === 'background') {
        // Blurred background fills the frame; sharp original is overlaid centred inside it.
        // Note: blur is a visual effect only — it does NOT make copyrighted content legal.
        filter = `[0:v]scale=${W}:${H}${scaleFlags}:force_original_aspect_ratio=increase,crop=${W}:${H},gblur=sigma=${blurSigma}[bg];[0:v]${panVf || `scale=${W}:${H}${scaleFlags}:force_original_aspect_ratio=decrease`}[fg];[bg][fg]overlay=(W-overlay_w)/2:(H-overlay_h)/2${clarityFilter}${fadeFilter}${subFilter}[vout]`;
      } else if (blurMode === 'custom') {
        filter = `[0:v]${panVf || baseVf},split[a][b];[a]gblur=sigma=${blurSigma},crop=w='iw*${blurW}':h='ih*${blurH}':x='iw*${blurX}':y='ih*${blurY}'[br];[b][br]overlay=x='iw*${blurX}':y='ih*${blurY}'${clarityFilter}${fadeFilter}${subFilter}[vout]`;
      } else {
        filter = `[0:v]${panVf || baseVf}${clarityFilter}${fadeFilter}${subFilter}[vout]`;
      }
      // Audio chain: original (volume) + narration (volume, if generated) + music
      // (volume, fades, looped/trimmed to the video duration, if uploaded).
      // All inputs are mixed with amix; nothing is silently ignored any more.
      let audioInputs = []; // labelled audio streams to mix, in order
      filter += `;[0:a]volume=${origVol}[a0]`; audioInputs.push('[a0]');
      let nextInputIdx = 1;
      if (p.tts?.file && fs.existsSync(p.tts.file)) {
        args.push('-i', p.tts.file); inputs.push(p.tts.file);
        filter += `;[${nextInputIdx}:a]volume=${narrVol}[a1]`; audioInputs.push('[a1]'); nextInputIdx++;
      }
      if (hasMusic) {
        // -stream_loop -1 must precede -i so the music loops until amix cuts it off.
        args.push('-stream_loop', '-1', '-i', p.music.path); inputs.push(p.music.path);
        const mIdx = nextInputIdx; nextInputIdx++;
        let mFilter = `[${mIdx}:a]volume=${musicVol}`;
        if (musicFadeIn > 0) mFilter += `,afade=t=in:st=0:d=${musicFadeIn}`;
        if (musicFadeOut > 0 && musicFadeOut < effDuration) mFilter += `,afade=t=out:st=${Math.max(0, effDuration - musicFadeOut)}:d=${musicFadeOut}`;
        mFilter += `,atrim=duration=${effDuration},asetpts=PTS-STARTPTS[music]`;
        filter += `;${mFilter}`; audioInputs.push('[music]');
      }
      let audioMap;
      if (audioInputs.length === 1) { filter += `;[a0]anull[aout]`; audioMap = '[aout]'; }
      else { filter += `;${audioInputs.join('')}amix=inputs=${audioInputs.length}:duration=first:dropout_transition=0[aout]`; audioMap = '[aout]'; }
      args.push('-filter_complex', filter, '-map', '[vout]', '-map', audioMap, '-c:v', 'libx264', '-preset', 'veryfast', '-c:a', 'aac', outMp4);
      await run('ffmpeg', args);
      const meta = await probe(outMp4);
      p.export = { status: 'ready', progress: 'done', aspect, quality, file: outMp4, duration: meta.duration, width: meta.width, height: meta.height, srt: srtPath, vtt: vttPath, dubbingPreset: dubbing, subtitlesFrom, subtitlesOn, blurMode, blurStrength, blurSigma, blurRegion, zoomEvery5s, zoomStrength, zoomMax, enhanceClarity, clarityStrength, clarityUnsharp, musicVolume: musicVol, musicFadeIn, musicFadeOut, musicFile: hasMusic ? p.music.name : null, sceneOrder: sceneOrderApplied ? rawOrder : null, sceneOrderApplied, sceneOrderNote, transition, transitionDuration, transitionApplied, transitionNote, panMode, subtitleFont: subFont };
      saveProjects(projects);
    } catch (e) { p.export = { status: 'failed', error: e.message }; saveProjects(projects); }
  })();
  res.json({ export: p.export });
});

// ---- Sangbik Workflow: Download -> Model -> Transcribe -> Translate -> Narrate -> Render ----
// Original code for this app only. Stages run in order; a stage that needs a missing
// key is recorded as needs-key and that stage is stopped — no transcript, translation
// or audio is ever invented to fill it. Later stages that do not depend on it (e.g.
// Local no-key narration from a manual script, and Render) may still run, and the
// final workflow status is "partial" when any stage was needs-key. "Download" here
// means the project's video input step (an upload, or a permission-confirmed direct
// media URL via /api/projects/:id/url); platform/DRM downloads are never attempted.
// "Model" actually loads the offline Whisper tiny model (whisper_local.py
// --check) and only then reports "Model ready: tiny (multilingual)".
// "Transcribe" runs that offline model (no key). "Translate" uses Google
// Translate batch (no key), with AI fallback only if Google fails and an AI
// key is configured. "Narrate" defaults to Edge TTS; if Edge fails, the
// Workflow falls back to Local espeak-ng with a clear log line — Local
// audio is never presented as Edge audio.
const WORKFLOW_STAGE_NAMES = ['download', 'model', 'transcribe', 'translate', 'narrate', 'render'];
function wfClock() { return new Date().toLocaleTimeString('en-GB', { timeZone: 'Asia/Kuala_Lumpur', hour12: false }); } // Log clock in the user's timezone (Asia/Kuala_Lumpur), matching the phone clock in the screenshot style; the full ISO timestamp is also stored.
function wfLog(p, stage, level, message) {
  if (!p.workflow) p.workflow = { status: 'not-started', stages: [], log: [] };
  p.workflow.log.push({ time: new Date().toISOString(), clock: wfClock(), stage, level, message });
}
function wfStage(p, name, status, detail) {
  const st = p.workflow.stages.find(x => x.name === name);
  if (st) { st.status = status; st.detail = detail || ''; if (status === 'done' || status === 'needs-key' || status === 'failed' || status === 'needs-input') st.finishedAt = new Date().toISOString(); else st.startedAt = new Date().toISOString(); }
}
async function wfSelf(path, method, body) {
  const r = await fetch(`http://localhost:${PORT}${path}`, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  let j = null; try { j = await r.json(); } catch {}
  return { ok: r.ok, status: r.status, json: j };
}
async function runWorkflow(projectId, opts) {
  const p = projects[projectId]; if (!p) return;
  if (p.workflow && p.workflow.status === 'running') return;
  p.workflow = { status: 'running', stages: WORKFLOW_STAGE_NAMES.map(n => ({ name: n, status: 'pending', detail: '' })), log: [], startedAt: new Date().toISOString(), options: opts || {} };
  saveProjects(projects);
  const base = `http://localhost:${PORT}`;
  try {
    // 1. Download (video input step)
    wfStage(p, 'download', 'running', 'Checking project video input…'); saveProjects(projects);
    if (!p.file || !fs.existsSync(p.file.path)) {
      wfStage(p, 'download', 'needs-input', 'No video uploaded yet — upload a video (or a permission-confirmed direct media URL) first, then run the workflow again.');
      wfLog(p, 'download', 'warn', 'Download: no video input — stage stopped, needs-input.');
      p.workflow.status = 'needs-input'; p.workflow.finishedAt = new Date().toISOString(); saveProjects(projects); return;
    }
    const dur = p.metadata?.duration || 0;
    wfStage(p, 'download', 'done', `${p.file.name} — ${p.file.size} bytes, ${dur}s, ${p.metadata?.resolution || '?'}`);
    wfLog(p, 'download', 'success', `Download: video input ready — ${p.file.name} (${p.file.size} bytes, ${dur}s).`);
    saveProjects(projects);

    // 2. Model — actually load the offline Whisper tiny model (no key).
    wfStage(p, 'model', 'running', 'Loading offline Whisper tiny model…'); saveProjects(projects);
    let modelReady = false;
    try {
      const readyLine = await runWhisperCheck();
      modelReady = true;
      wfStage(p, 'model', 'done', `Model ready: tiny (multilingual) — ${readyLine}`);
      wfLog(p, 'model', 'success', 'model: Model ready: tiny (multilingual)');
    } catch (e) {
      wfStage(p, 'model', 'failed', `Offline Whisper model failed to load: ${e.message}`);
      wfLog(p, 'model', 'error', `model: Model load failed — ${e.message}`);
    }
    saveProjects(projects);

    // 3. Transcribe — offline Whisper tiny (no key).
    wfStage(p, 'transcribe', 'running', 'Transcribing with offline Whisper tiny…'); saveProjects(projects);
    if (!modelReady) {
      wfStage(p, 'transcribe', 'failed', 'Transcription skipped — offline Whisper model failed to load (see Model stage). No transcript was faked.');
      wfLog(p, 'transcribe', 'error', 'asr: Transcription skipped — model load failed.');
    } else {
      const r = await wfSelf(`/api/projects/${p.id}/transcribe`, 'POST', {});
      if (r.ok) {
        const n = r.json?.transcript?.segments?.length || 0;
        const span = r.json?.transcript?.segments?.length ? `${r.json.transcript.segments[0].start}s-${r.json.transcript.segments[r.json.transcript.segments.length - 1].end}s of the audio` : 'unknown span';
        wfStage(p, 'transcribe', 'done', `Transcribed ${n} segments (speech span: ${span}).`);
        wfLog(p, 'transcribe', 'success', `asr: Transcribed ${n} segments`);
        if (r.json?.transcript?.language) wfLog(p, 'transcribe', 'info', `asr: Detected language: ${r.json.transcript.language} (Whisper offline tiny, multilingual)`);
        wfLog(p, 'transcribe', 'info', `asr: Speech span: ${span}`);
      } else {
        wfStage(p, 'transcribe', 'failed', r.json?.error || `Transcription failed (HTTP ${r.status}).`);
        wfLog(p, 'transcribe', 'error', `asr: Transcription failed — ${r.json?.error || r.status}`);
      }
    }
    saveProjects(projects);

    // 4. Translate — Google Translate batch (no key), AI fallback only if Google fails.
    wfStage(p, 'translate', 'running', 'Translating with Google Translate…'); saveProjects(projects);
    let lines = p.translationLines || [];
    let burmeseLines = lines.filter(l => l.burmese);
    if (burmeseLines.length && !(p.transcript?.segments?.length && burmeseLines.length < (p.transcript.segments.length))) {
      wfStage(p, 'translate', 'done', `Translated ${burmeseLines.length}/${lines.length} lines (Burmese lines already present — Google Translate not re-run).`);
      wfLog(p, 'translate', 'success', `mt: Translated ${burmeseLines.length}/${lines.length} lines`);
    } else if (p.transcript?.segments?.length) {
      wfLog(p, 'translate', 'info', `mt: Google Translate: ${(p.transcript.segments.length)} lines…`);
      const gr = await wfSelf(`/api/projects/${p.id}/translate-google`, 'POST', {});
      if (gr.ok) {
        lines = gr.json.lines || []; burmeseLines = lines.filter(l => l.burmese);
        wfStage(p, 'translate', 'done', `Translated ${burmeseLines.length}/${lines.length} lines via ${gr.json.path || 'google'}.`);
        if (gr.json.googleLog) wfLog(p, 'translate', 'success', `mt: ${gr.json.googleLog}`);
        wfLog(p, 'translate', 'success', `mt: Translated ${burmeseLines.length}/${lines.length} lines`);
        if (gr.json.note) wfLog(p, 'translate', 'warn', `mt: Note — ${gr.json.note}`);
      } else {
        wfStage(p, 'translate', 'failed', gr.json?.error || `Translation failed (HTTP ${gr.status}).`);
        wfLog(p, 'translate', 'error', `mt: Translation failed — ${gr.json?.error || gr.status}`);
      }
    } else if (lines.length) {
      wfStage(p, 'translate', 'needs-input', `${lines.length} line(s) have no Burmese text and no transcript is available — run Transcribe first, or add Burmese lines manually.`);
      wfLog(p, 'translate', 'warn', 'mt: Translation skipped — no transcript and no Burmese lines.');
    } else {
      wfStage(p, 'translate', 'needs-input', 'No translation lines yet — run Transcribe first, or add lines manually in the Translation Editor.');
      wfLog(p, 'translate', 'warn', 'mt: Translation skipped — no lines.');
    }
    saveProjects(projects);

    // 5. Narrate — Edge TTS default; Local fallback ONLY with a clear log line.
    const provider = (opts.provider || defaultTtsProvider()).toLowerCase();
    const source = opts.source || (burmeseLines.length ? 'translation' : 'script');
    wfStage(p, 'narrate', 'running', `Starting TTS (provider: ${provider}, source: ${source})…`); saveProjects(projects);
    wfLog(p, 'narrate', 'info', `tts: Starting TTS (provider=${provider}, source=${source}, voice=${opts.voice || 'my-MM-NilarNeural'})…`);
    let tr = await wfSelf(`/api/projects/${p.id}/tts`, 'POST', { provider, source, speed: opts.speed || 1, pause: opts.pause ?? 0.5, voice: opts.voice || 'my-MM-NilarNeural' });
    if (tr.ok && tr.json?.tts) {
      wfStage(p, 'narrate', 'done', `Narration ready: voice=${tr.json.tts.voice}, ${tr.json.tts.segments} segment(s), ${tr.json.tts.duration}s.`);
      wfLog(p, 'narrate', 'success', `tts: Narration done (voice=${tr.json.tts.voice}, ${tr.json.tts.segments} segment(s), ${tr.json.tts.duration}s).`);
    } else if (provider === 'edge-tts' || provider === 'edge') {
      const edgeErr = tr.json?.error || `Edge TTS failed (HTTP ${tr.status})`;
      wfLog(p, 'narrate', 'error', `tts: Edge TTS failed — ${edgeErr}`);
      wfLog(p, 'narrate', 'warn', 'tts: Falling back to Local espeak-ng — this audio is Local, NOT Edge audio.');
      const fr = await wfSelf(`/api/projects/${p.id}/tts`, 'POST', { provider: 'local', source, speed: opts.speed || 1, pause: opts.pause ?? 0.5 });
      if (fr.ok && fr.json?.tts) {
        wfStage(p, 'narrate', 'done', `Narration ready via Local fallback (Edge failed): voice=${fr.json.tts.voice}, ${fr.json.tts.segments} segment(s), ${fr.json.tts.duration}s. This audio is Local, NOT Edge.`);
        wfLog(p, 'narrate', 'success', `tts: Narration done via Local fallback (voice=${fr.json.tts.voice}, ${fr.json.tts.segments} segment(s), ${fr.json.tts.duration}s) — NOT Edge audio.`);
      } else {
        wfStage(p, 'narrate', 'failed', `Edge TTS failed and Local fallback also failed: ${fr.json?.error || fr.status}. Edge error: ${edgeErr}`);
        wfLog(p, 'narrate', 'error', `tts: Local fallback failed — ${fr.json?.error || fr.status}`);
      }
    } else {
      const needsKey = tr.status === 501;
      wfStage(p, 'narrate', needsKey ? 'needs-key' : 'failed', tr.json?.error || `Narration failed (HTTP ${tr.status}).`);
      wfLog(p, 'narrate', needsKey ? 'warn' : 'error', `tts: Narration stopped — ${tr.json?.error || tr.status}`);
    }
    saveProjects(projects);

    // 6. Render (reuses the real export endpoint, then polls until ready/failed)
    wfStage(p, 'render', 'running', 'Rendering export with FFmpeg…'); saveProjects(projects);
    wfLog(p, 'render', 'info', `render: Starting Render (FFmpeg export) — blur=${opts.blurMode || 'none'}/${opts.blurStrength || 'medium'}, subtitle color=${opts.subtitles?.color || opts.subtitleColor || '#FFFFFF'}…`);
    const er = await wfSelf(`/api/projects/${p.id}/export`, 'POST', { aspect: opts.aspect || '16:9', quality: opts.quality || '720p', subtitlesFrom: source, subtitlesOn: true, narrationVolume: 1, originalVolume: source === 'translation' ? 0 : 0.5, dubbingPreset: source === 'translation', blurMode: opts.blurMode || 'none', blurStrength: opts.blurStrength || 'medium', subtitles: { color: opts.subtitles?.color || opts.subtitleColor || '#FFFFFF' } });
    if (!er.ok) {
      wfStage(p, 'render', 'failed', er.json?.error || `Export failed to start (HTTP ${er.status}).`);
      wfLog(p, 'render', 'error', `render: Export failed to start — ${er.json?.error || er.status}`);
    } else {
      let exp = null;
      for (let i = 0; i < 90; i++) {
        await new Promise(r => setTimeout(r, 2000));
        const sr = await wfSelf(`/api/projects/${p.id}/export-status`, 'GET');
        exp = sr.json;
        if (exp && exp.status !== 'processing') break;
      }
      if (exp && exp.status === 'ready') {
        let size = 0; try { size = fs.statSync(exp.file).size; } catch {}
        wfStage(p, 'render', 'done', `Render ready: ${exp.width}x${exp.height}, ${exp.duration}s, ${size} bytes.`);
        wfLog(p, 'render', 'success', `render: Render ready — ${exp.width}x${exp.height}, ${exp.duration}s, ${size} bytes.`);
      } else {
        wfStage(p, 'render', 'failed', exp?.error || 'Export did not finish (timed out polling export-status).');
        wfLog(p, 'render', 'error', `render: Render failed — ${exp?.error || 'polling timeout'}`);
      }
    }
    saveProjects(projects);
    const stages = p.workflow.stages;
    if (stages.some(s => s.status === 'failed')) p.workflow.status = 'failed';
    else if (stages.some(s => s.status === 'needs-key' || s.status === 'needs-input')) p.workflow.status = 'partial';
    else p.workflow.status = 'done';
    p.workflow.finishedAt = new Date().toISOString();
    wfLog(p, 'workflow', 'info', `Workflow finished: ${p.workflow.status} — ${stages.map(s => `${s.name}:${s.status}`).join(', ')}.`);
    saveProjects(projects);
  } catch (e) {
    wfLog(p, 'workflow', 'error', `Workflow error: ${e.message}`);
    p.workflow.status = 'failed'; p.workflow.finishedAt = new Date().toISOString(); saveProjects(projects);
  }
}
app.post('/api/projects/:id/workflow/run', (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' });
  if (p.workflow && p.workflow.status === 'running') return res.json({ workflow: p.workflow, message: 'Workflow already running.' });
  setImmediate(() => runWorkflow(p.id, req.body || {}).catch(() => {}));
  res.json({ workflow: { status: 'queued', stages: WORKFLOW_STAGE_NAMES.map(n => ({ name: n, status: 'pending', detail: '' })), log: [] }, message: 'Workflow queued — poll GET /api/projects/:id/workflow.' });
});
app.get('/api/projects/:id/workflow', (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' });
  res.json(p.workflow || { status: 'not-started', stages: WORKFLOW_STAGE_NAMES.map(n => ({ name: n, status: 'pending', detail: '' })), log: [], message: 'No workflow run yet — POST /api/projects/:id/workflow/run to start one.' });
});

app.get('/api/projects/:id/export-status', (req, res) => { const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' }); res.json(p.export || { status: 'not-started' }); });
app.get('/api/projects/:id/download/:kind', (req, res) => {
  const p = projects[req.params.id]; if (!p) return res.status(404).json({ error: 'Project not found' });
  const kind = req.params.kind;
  if (kind === 'script') { res.setHeader('Content-Type', 'text/plain; charset=utf-8'); res.setHeader('Content-Disposition', 'attachment; filename="script.txt"'); return res.send(p.script || ''); }
  if (kind === 'subtitles') { const f = p.export?.srt; if (f && fs.existsSync(f)) return res.download(f, 'subtitles.srt'); return res.status(404).json({ error: 'Subtitles not generated yet' }); }
  if (kind === 'translation-srt') { if (!p.translationLines?.length) return res.status(404).json({ error: 'No translation lines saved yet' }); const f = writeSrt(p, p.translationLines.map(l => ({ start: l.start, end: l.end, text: l.burmese }))); return res.download(f, 'translation.srt'); }
  if (kind === 'translation-vtt') { if (!p.translationLines?.length) return res.status(404).json({ error: 'No translation lines saved yet' }); const f = writeVtt(p, p.translationLines.map(l => ({ start: l.start, end: l.end, text: l.burmese })), '-translation'); return res.download(f, 'translation.vtt'); }
  if (kind === 'audio') { const f = p.tts?.file; if (f && fs.existsSync(f)) return res.download(f, 'narration.mp3'); return res.status(404).json({ error: 'Narration audio not generated yet' }); }
  if (kind === 'video') { const f = p.export?.file; if (f && fs.existsSync(f)) return res.download(f, 'recap.mp4'); return res.status(404).json({ error: 'Export video not ready yet' }); }
  res.status(400).json({ error: 'Unknown download kind' });
});

app.get('/api/projects/:id/preview', (req, res) => { const p = projects[req.params.id]; if (!p || !p.file || !fs.existsSync(p.file.path)) return res.status(404).json({ error: 'Video not found' }); res.sendFile(p.file.path); });
app.listen(PORT, '0.0.0.0', () => console.log(`Sangbik server on http://0.0.0.0:${PORT} (max ${MAX_MB} MB, voices: ${TTS_VOICES.join(', ')})`));
