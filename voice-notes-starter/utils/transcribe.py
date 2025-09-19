# utils/transcribe.py  (안정 버전)
from typing import Optional, List
from openai import OpenAI, APIConnectionError, APITimeoutError
from dotenv import load_dotenv
from pydub import AudioSegment
import io, os, tempfile, time

# 허용 확장자
ALLOWED = {'flac','m4a','mp3','mp4','mpeg','mpga','oga','ogg','wav','webm'}

# 분할 기준(둘 중 하나라도 넘으면 분할)
CHUNK_SECONDS = 600        # 10분
MAX_BYTES = 20 * 1024 * 1024  # 20MB

def get_client(api_key: Optional[str] = None) -> OpenAI:
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    # 타임아웃/재시도 기본값
    return OpenAI(api_key=key, timeout=180.0, max_retries=3)

def get_audio_duration_seconds(raw_bytes: bytes, ext: str) -> float:
    audio = AudioSegment.from_file(io.BytesIO(raw_bytes), format=ext)
    return len(audio) / 1000.0

def _transcribe_file_path(client: OpenAI, path: str, language_hint: Optional[str]) -> str:
    # 파라미터 최소화(일부 조합에서 400 유발 방지) + whisper-1 기본
    with open(path, "rb") as f:
        r = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            language=language_hint or None,
        )
    return getattr(r, "text", str(r))

def _request_with_retries(client: OpenAI, path: str, language_hint: Optional[str]) -> str:
    last = None
    for attempt in range(4):
        try:
            return _transcribe_file_path(client, path, language_hint)
        except (APIConnectionError, APITimeoutError, TimeoutError) as e:
            last = e
            time.sleep(min(1.5 * (2 ** attempt), 8))
            continue
        except Exception as e:
            last = e
            break
    if last:
        raise last

def _export_wav_16k_mono(seg: AudioSegment) -> str:
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    seg.export(tmp.name, format="wav")
    tmp.close()
    return tmp.name

def transcribe_audio(file_bytes: bytes, filename: str, api_key: Optional[str] = None, language_hint: Optional[str] = None) -> str:
    """
    1) 업로드 파일을 임시 파일로 저장
    2) 크기/길이에 따라 자동 분할(10분/20MB 기준)
    3) 각 조각을 16kHz mono WAV로 변환 → 순차 전송 → 결합
    """
    if not file_bytes:
        raise ValueError("업로드된 파일이 비어있습니다.")

    client = get_client(api_key)
    ext = (filename.split(".")[-1] or "").lower()

    # 원본 임시 저장
    with tempfile.NamedTemporaryFile(suffix=f".{ext if ext in ALLOWED else 'bin'}", delete=False) as tmp:
        tmp.write(file_bytes)
        src_path = tmp.name

    try:
        # 오디오 로드
        audio = AudioSegment.from_file(src_path, format=ext if ext in ALLOWED else None)
        total_ms = len(audio)
        file_size = os.path.getsize(src_path)

        # 분할 여부 판단
        need_chunk = (file_size > MAX_BYTES) or (total_ms > CHUNK_SECONDS * 1000)

        if not need_chunk:
            # 작은 파일: 원본 확장자면 그대로 시도, 실패시 WAV 폴백
            try:
                if ext in ALLOWED:
                    return _request_with_retries(client, src_path, language_hint)
            except Exception:
                pass  # 아래 WAV 폴백

            wav_path = _export_wav_16k_mono(audio)
            try:
                return _request_with_retries(client, wav_path, language_hint)
            finally:
                try: os.remove(wav_path)
                except: pass

        # === 분할 처리 ===
        parts: List[str] = []
        start = 0
        while start < total_ms:
            end = min(start + CHUNK_SECONDS * 1000, total_ms)
            seg = audio[start:end]
            wav_path = _export_wav_16k_mono(seg)
            try:
                text = _request_with_retries(client, wav_path, language_hint)
                # 파트 구분자 포함해 이어붙임(원하면 제거 가능)
                header = f"\n\n--- [Segment {len(parts)+1}] ({start/1000:.0f}s ~ {end/1000:.0f}s) ---\n\n"
                parts.append(header + (text or "").strip())
            finally:
                try: os.remove(wav_path)
                except: pass
            start = end

        return "\n".join(parts).strip()

    finally:
        try: os.remove(src_path)
        except: pass
