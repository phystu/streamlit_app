# utils/transcribe.py
from __future__ import annotations

from typing import Optional, List
from openai import OpenAI, APIConnectionError, APITimeoutError
from dotenv import load_dotenv
from pydub import AudioSegment
from pathlib import Path
import io, os, re, tempfile, time, shutil

# 허용 확장자
ALLOWED = {'flac','m4a','mp3','mp4','mpeg','mpga','oga','ogg','wav','webm'}

# 분할 기준(둘 중 하나라도 넘으면 분할)
CHUNK_SECONDS = 600            # 10분
MAX_BYTES = 20 * 1024 * 1024   # 20MB

# ---------- 유틸: 멀티파트 헤더용 안전 ASCII 파일명 ----------
_ASCII_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]+$")

def _ascii_filename(name: str, default: str = "audio.wav") -> str:
    p = Path(name or default)
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", p.stem) or "audio"
    suffix = p.suffix if _ASCII_SAFE_SUFFIX.match(p.suffix or "") else Path(default).suffix
    return f"{stem}{suffix}"

# ---------- OpenAI 클라이언트 ----------
def get_client(api_key: Optional[str] = None) -> OpenAI:
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    # timeout/max_retries는 최신 SDK에서 지원
    return OpenAI(api_key=key, timeout=180.0, max_retries=3)

# ---------- 길이 계산 ----------
def get_audio_duration_seconds(raw_bytes: bytes, ext: str) -> float:
    audio = AudioSegment.from_file(io.BytesIO(raw_bytes), format=ext)
    return len(audio) / 1000.0

# ---------- OpenAI 전송(파일 경로 입력) ----------
def _transcribe_file_path(client: OpenAI, path: str | os.PathLike, language_hint: Optional[str]) -> str:
    """
    멀티파트의 filename 헤더를 ASCII로 보장하기 위해
    항상 임시 디렉터리에 ASCII 안전 이름으로 복사한 뒤 그 파일을 엽니다.
    """
    src = Path(path)
    safe_name = _ascii_filename(src.name, default="audio.wav")
    with tempfile.TemporaryDirectory(prefix="upl_") as td:
        tmp = Path(td) / safe_name
        shutil.copyfile(src, tmp)  # 바이너리 그대로 복사

        with open(tmp, "rb") as f:
            r = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",   # Whisper 대체용 최신 전사 모델
                file=f,
                language=(language_hint or None),
            )
    return getattr(r, "text", str(r))

def _request_with_retries(client: OpenAI, path: str | os.PathLike, language_hint: Optional[str]) -> str:
    last: Optional[Exception] = None
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

# ---------- WAV 내보내기 ----------
def _export_wav_16k_mono(seg: AudioSegment) -> str:
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    seg.export(tmp.name, format="wav")
    tmp.close()
    return tmp.name

# ---------- 메인: 바이트 입력 전사 ----------
def transcribe_audio(
    file_bytes: bytes,
    filename: str,
    api_key: Optional[str] = None,
    language_hint: Optional[str] = None
) -> str:
    """
    1) 업로드 파일을 임시 파일로 저장(확장자 유지)
    2) 크기/길이에 따라 자동 분할(10분/20MB 기준)
    3) 각 조각을 16kHz mono WAV로 변환 → 순차 전송 → 결합
    ※ OpenAI 전송 시에는 항상 ASCII 안전 파일명으로 복사하여 업로드(Unicode 헤더 이슈 회피)
    """
    if not file_bytes:
        raise ValueError("업로드된 파일이 비어있습니다.")

    client = get_client(api_key)
    ext = (filename.split(".")[-1] or "").lower()

    # 원본 임시 저장 (이 경로의 파일명은 업로드에 직접 쓰지 않으므로 Unicode 포함 가능)
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
            # 작은 파일: 원본 확장자면 그대로 시도(업로드 시 내부에서 ASCII 임시복사)
            try:
                if ext in ALLOWED:
                    return _request_with_retries(client, src_path, language_hint)
            except Exception:
                # 아래 WAV 폴백
                pass

            wav_path = _export_wav_16k_mono(audio)
            try:
                return _request_with_retries(client, wav_path, language_hint)
            finally:
                try: os.remove(wav_path)
                except: pass

        # === 분할 처리 ===
        parts: List[str] = []
        start = 0
        seg_idx = 0
        while start < total_ms:
            end = min(start + CHUNK_SECONDS * 1000, total_ms)
            seg = audio[start:end]
            wav_path = _export_wav_16k_mono(seg)
            seg_idx += 1
            try:
                text = _request_with_retries(client, wav_path, language_hint)
                header = f"\n\n--- [Segment {seg_idx}] ({start/1000:.0f}s ~ {end/1000:.0f}s) ---\n\n"
                parts.append(header + (text or "").strip())
            finally:
                try: os.remove(wav_path)
                except: pass
            start = end

        return "\n".join(parts).strip()

    finally:
        try: os.remove(src_path)
        except: pass
