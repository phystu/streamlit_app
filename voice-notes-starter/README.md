# 🩺 음성기반 자동 회의록/연구노트 (Streamlit)

이 프로젝트는 **대학병원 연구/일반 회의**에서 쓰기 좋은 **음성 → 전사 → 요약 → 서식 적용 → Markdown/PDF 저장** 파이프라인을 제공합니다.

## 기능
- Whisper 기반 **전사(음성→문자)**
- GPT 요약으로 **핵심 정리/결정/액션** 자동 생성
- 자동 분류: **일반 회의록** vs **연구노트**
- Jinja 템플릿으로 **Markdown 생성**
- `wkhtmltopdf`가 있을 때 **고품질 PDF**, 없으면 **간단 PDF**로 저장

## 설치 (Windows PowerShell 기준)
```powershell
# 1) 폴더로 이동
cd voice-notes-starter

# 2) 가상환경 만들기/활성화
python -m venv .venv
.venv\Scripts\Activate

# 실행정책 문제시(관리자 PowerShell):
# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3) 필수 라이브러리 설치
pip install -r requirements.txt

# 4) .env 준비
copy .env.example .env
# .env 파일 열어 OPENAI_API_KEY 를 본인 키로 설정
```

### (선택) 고품질 PDF를 원한다면
- [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html) 설치 후, `pdfkit`가 자동으로 감지합니다.
- PATH 인식이 안되면, `os.add_dll_directory` 또는 `pdfkit.configuration(wkhtmltopdf="C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")` 식으로 지정 가능.

## 실행
```powershell
streamlit run app.py
```
브라우저에서 UI가 열립니다.

## 사용 흐름
1. 오디오 파일 업로드 (`mp3/m4a/wav/ogg/webm` 등)
2. 제목/일시/장소/참석자/진행·서기 입력
3. **전사 → 요약 → 서식 적용 실행** 버튼 클릭
4. 자동으로 문서 유형(일반/연구) 판별 및 템플릿 적용
5. **Markdown / PDF 다운로드**

## 커스터마이징 포인트
- 템플릿: `templates/meeting.md.j2`, `templates/research.md.j2`
- 분류 로직: `utils/classify.py`
- 요약 프롬프트: `utils/summarize.py`

## 자주 발생하는 오류
- `OPENAI_API_KEY` 없음 → `.env` 설정 필요
- 오디오 길이 확인 오류 → ffmpeg 필요 시 [ffmpeg.org](https://ffmpeg.org/download.html) 설치 (pydub가 사용)
- PDF 변환 오류 → wkhtmltopdf 미설치 가능, fallback로 단순 PDF 생성

## 보안 팁
- 환자 정보(PHI)는 업로드 전에 익명화 권장
- 내부망 프록시/방화벽 환경에서는 사내 정책 준수
