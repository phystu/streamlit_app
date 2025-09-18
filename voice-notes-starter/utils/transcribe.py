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
                # 포맷/파라미터/모델 제약 등 → 다음 모델로 폴백
                last_err = e
                break
            except Exception as e:
                last_err = e
                break
        # 다음 모델로 폴백 시도
    if last_err:
        raise last_err
    return ""

# ---------------------- 메인 ----------------------
def transcribe_audio(file_bytes: bytes, filename: str,
                     api_key: Optional[str] = None,
                     language_hint: Optional[str] = None) -> str:
    """
    1) 임시파일 저장
    2) 크기/길이에 따라 자동 분할(10분/20MB)
    3) 각 조각을 16kHz mono WAV로 변환 → 순차 전송 → 결합
    4) 품질 가드: 결과가 너무 짧으면 예외
    """
    if not file_bytes:
        raise ValueError("업로드된 파일이 비어있습니다.")

    client = get_client(api_key)
    ext = (filename.split(".")[-1] or "").lower()

    # 원본 임시 저장
    src_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext if ext in ALLOWED else 'bin'}", delete=False) as tmp:
            tmp.write(file_bytes)
            src_path = tmp.name

        # 오디오 로드(확장자 기반 포맷 명시)
        audio = AudioSegment.from_file(src_path, format=ext if ext in ALLOWED else None)
        total_ms = len(audio)
        file_size = os.path.getsize(src_path)

        # 분할 여부 판단
        need_chunk = (file_size > MAX_BYTES) or (total_ms > CHUNK_SECONDS * 1000)

        # ---- 단일 처리 경로 ----
        if not need_chunk:
            # 1차: 원본 그대로 시도 (허용 확장자일 때)
            if ext in ALLOWED:
                try:
                    text = _request_with_retries(client, src_path, language_hint)
                    if text.strip():
                        if len(text.strip()) < 30:
                            # 품질 가드
                            raise RuntimeError("전사 결과가 비정상적으로 짧습니다.")
                        return text
                except Exception:
                    pass  # 아래 WAV 폴백

            # 2차: 16k WAV 폴백
            wav_path = _export_wav_16k_mono(audio)
            try:
                text = _request_with_retries(client, wav_path, language_hint)
                if not text.strip():
                    raise RuntimeError("전사 결과가 비어 있습니다.")
                if len(text.strip()) < 30:
                    raise RuntimeError("전사 결과가 비정상적으로 짧습니다.")
                return text
            finally:
                _safe_remove(wav_path)

        # ---- 분할 처리 경로 ----
        parts: List[str] = []
        start = 0
        seg_idx = 0
        while start < total_ms:
            end = min(start + CHUNK_SECONDS * 1000, total_ms)
            seg = audio[start:end]
            wav_path = _export_wav_16k_mono(seg)
            seg_idx += 1
            try:
                text = _request_with_retries(client, wav_path, language_hint).strip()
                # 실패/공백이면 구간 표시만 하고 비워둘 수도 있지만, 명확히 알리기 위해 최소 텍스트 요구 X
                header = f"\n\n--- [Segment {seg_idx}] ({int(start/1000)}s ~ {int(end/1000)}s) ---\n\n"
                parts.append(header + text)
            finally:
                _safe_remove(wav_path)
            start = end

        full_text = "\n".join(parts).strip()
        if not full_text:
            raise RuntimeError("전사 결과가 비어 있습니다.")
        if len(full_text) < 30:
            raise RuntimeError("전사 결과가 비정상적으로 짧습니다.")
        return full_text

    finally:
        _safe_remove(src_path)
