from typing import Optional, List, Tuple
from openai import OpenAI, APIConnectionError, APITimeoutError, BadRequestError, RateLimitError
from dotenv import load_dotenv
from pydub import AudioSegment
import io, os, tempfile, time, math

# 허용 확장자
ALLOWED = {'flac','m4a','mp3','mp4','mpeg','mpga','oga','ogg','wav','webm'}

# 분할 기준(둘 중 하나라도 넘으면 분할)
CHUNK_SECONDS = 600            # 10분
MAX_BYTES = 20 * 1024 * 1024   # 20MB

# 기본 선호 모델(가용성에 따라 자동 폴백)
PREFERRED_MODELS = ["gpt-4o-mini-transcribe", "whisper-1"]

# ---------------------- 클라이언트 ----------------------
def get_client(api_key: Optional[str] = None) -> OpenAI:
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key, timeout=180.0, max_retries=0)  # 재시도는 아래 수동 제어

# ---------------------- 유틸 ----------------------
def get_audio_duration_seconds(raw_bytes: bytes, ext: str) -> float:
    # BytesIO 읽기 시에는 format을 명시해야 안전
    audio = AudioSegment.from_file(io.BytesIO(raw_bytes), format=(ext or "").lower())
    return len(audio) / 1000.0

def _export_wav_16k_mono(seg: AudioSegment) -> str:
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    seg.export(tmp.name, format="wav")
    tmp.close()
    return tmp.name

def _safe_remove(path: Optional[str]) -> None:
    if not path: return
    try: os.remove(path)
    except: pass

# ---------------------- API 호출 ----------------------
def _transcribe_with_model(client: OpenAI, path: str, language_hint: Optional[str], model: str) -> str:
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            language=language_hint or None,
        )
    return getattr(resp, "text", str(resp)) or ""

def _request_with_retries(client: OpenAI, path: str, language_hint: Optional[str]) -> str:
    """
    선호 모델 목록을 순회하며 백오프 재시도.
    - 네트워크/타임아웃/429: 지수 백오프 재시도
    - 400/415 등 요청 오류: 즉시 다음 모델로 폴백
    """
    last_err: Optional[Exception] = None
    for model in PREFERRED_MODELS:
        for attempt in range(5):
            try:
                return _transcribe_with_model(client, path, language_hint, model)
            except (APIConnectionError, APITimeoutError, TimeoutError, RateLimitError) as e:
                last_err = e
                sleep = min(1.5 * (2 ** attempt), 10)
                time.sleep(sleep)
                continue
            except BadRequestError as e:
