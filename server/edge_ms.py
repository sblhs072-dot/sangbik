"""Direct Edge TTS via the current msedgeservices endpoint.
Token handling is taken ONLY from the installed open-source edge-tts
package itself (its public TRUSTED_CLIENT_TOKEN constant and its
Sec-MS-GEC generator) — no token is hard-coded here and none was taken
from any other app. Proxy, if set, comes from the environment.
"""
import asyncio, os, ssl, sys
import aiohttp, certifi
from edge_tts.constants import TRUSTED_CLIENT_TOKEN, SEC_MS_GEC_VERSION, WSS_HEADERS
from edge_tts.communicate import connect_id, date_to_string, mkssml, ssml_headers_plus_data
from edge_tts.data_classes import TTSConfig
from edge_tts.drm import DRM

async def main(text, voice, out, rate="+0%"):
    url = (
        "wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1/"
        f"?Ocp-Apim-Subscription-Key={TRUSTED_CLIENT_TOKEN}"
        f"&ConnectionId={connect_id()}"
        f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
        f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}"
    )
    cfg = TTSConfig(voice=voice, rate=rate, volume="+0%", pitch="+0Hz", boundary="WordBoundary")
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    audio = bytearray()
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.ws_connect(url, proxy=proxy, headers=WSS_HEADERS, ssl=ssl_ctx, compress=15) as ws:
            req = connect_id()
            # speech.config
            cfg_msg = (
                f"X-Timestamp:{date_to_string()}\r\n"
                "Content-Type:application/json; charset=utf-8\r\n"
                "Path:speech.config\r\n\r\n"
                '{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":false,"wordBoundaryEnabled":true},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}'
            )
            await ws.send_str(cfg_msg)
            await ws.send_str(ssml_headers_plus_data(req, date_to_string(), mkssml(cfg, text)))
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = msg.data
                    # binary frames: 2-byte header length prefix
                    if len(data) > 2:
                        hlen = int.from_bytes(data[:2], "big")
                        if len(data) > hlen + 2:
                            audio.extend(data[hlen + 2:])
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    if "Path:turn.end" in msg.data:
                        break
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
    print(f"AUDIO_BYTES={len(audio)}")
    if audio:
        open(out, "wb").write(bytes(audio))
        print(f"WROTE={out}")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "+0%"))
