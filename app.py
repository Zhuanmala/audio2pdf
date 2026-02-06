import hashlib
import hmac
import io
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from google import genai
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

load_dotenv()

MAX_FILE_SIZE_MB = 100
TARGET_LANGUAGE = "Chinese (Simplified)"
DEFAULT_GOOGLE_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
)
SUPPORTED_AUDIO_TYPES = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg"]
MIME_BY_SUFFIX = {
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
}


def resolve_config_value(key: str) -> str:
    env_value = (os.getenv(key) or "").strip()
    if env_value:
        return env_value

    try:
        secret_value = str(st.secrets.get(key, "")).strip()
        if secret_value:
            return secret_value
    except Exception:
        pass

    return ""


def resolve_google_api_key() -> str:
    return resolve_config_value("GOOGLE_API_KEY")


def resolve_google_model() -> str:
    configured = resolve_config_value("GOOGLE_MODEL")
    if configured:
        return configured
    return DEFAULT_GOOGLE_MODELS[0]


def resolve_login_credentials() -> tuple[str, str]:
    username = resolve_config_value("APP_USERNAME")
    password = resolve_config_value("APP_PASSWORD")
    return username, password


def get_google_client() -> genai.Client:
    api_key = resolve_google_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Set it in .env or Streamlit Secrets."
        )
    return genai.Client(api_key=api_key)


def infer_mime_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return MIME_BY_SUFFIX.get(suffix, "audio/mpeg")


def extract_gemini_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text.strip())
    return "\n".join(chunks).strip()


def is_model_not_found_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "not_found" in message or "is not found for api version" in message


def build_model_candidates() -> list[str]:
    candidates: list[str] = []

    working = st.session_state.get("google_working_model", "")
    configured = resolve_google_model()

    if isinstance(working, str) and working.strip():
        candidates.append(working.strip())
    if configured:
        candidates.append(configured)

    candidates.extend(DEFAULT_GOOGLE_MODELS)

    deduped: list[str] = []
    for model_name in candidates:
        normalized = model_name.strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def generate_content_with_fallback(client: genai.Client, contents: list[Any]) -> Any:
    attempted: list[str] = []
    last_error: Exception | None = None

    for model_name in build_model_candidates():
        attempted.append(model_name)
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            st.session_state.google_working_model = model_name
            return response
        except Exception as exc:
            last_error = exc
            if not is_model_not_found_error(exc):
                raise
            continue

    raise RuntimeError(
        "No available Google model found. "
        f"Attempted: {', '.join(attempted)}. "
        "Set GOOGLE_MODEL in Secrets to a valid model for your region/project. "
        f"Last error: {last_error}"
    )


def get_file_state_name(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return "ACTIVE"
    state_name = getattr(state, "name", None)
    if isinstance(state_name, str) and state_name:
        return state_name
    state_str = str(state)
    if "." in state_str:
        return state_str.split(".")[-1]
    return state_str


def wait_for_file_active(
    client: genai.Client, file_name: str, max_wait_seconds: int = 180
) -> None:
    start_time = time.time()
    while True:
        status_file = client.files.get(name=file_name)
        state_name = get_file_state_name(status_file)

        if state_name in {"ACTIVE", "STATE_UNSPECIFIED", ""}:
            return
        if state_name == "FAILED":
            raise RuntimeError("Google file processing failed.")
        if time.time() - start_time > max_wait_seconds:
            raise RuntimeError("Google file processing timed out.")
        time.sleep(1.5)


def transcribe_audio_with_google(audio_bytes: bytes, filename: str) -> str:
    client = get_google_client()
    uploaded = None
    suffix = Path(filename).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        uploaded = client.files.upload(file=temp_path)
        wait_for_file_active(client=client, file_name=uploaded.name)

        prompt = (
            "Transcribe the audio into plain text. "
            "Keep original meaning and line breaks. "
            "Return transcript text only, no explanation."
        )
        response = generate_content_with_fallback(
            client=client,
            contents=[prompt, uploaded],
        )
        transcript = extract_gemini_text(response)
        if not transcript:
            raise RuntimeError("No transcript text returned from Google.")
        return transcript
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def translate_text_with_google(text: str, target_language: str) -> str:
    client = get_google_client()
    prompt = (
        "Translate the following text to "
        f"{target_language}. "
        "Keep line breaks and original structure. "
        "Return translated text only.\n\n"
        f"{text}"
    )
    response = generate_content_with_fallback(
        client=client,
        contents=[prompt],
    )
    translated = extract_gemini_text(response)
    if not translated:
        raise RuntimeError("No translation text returned from Google.")
    return translated


def get_pdf_font_name() -> str:
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        return "Helvetica"


def draw_text_block(
    pdf: canvas.Canvas,
    title: str,
    content: str,
    x_margin: float,
    y_position: float,
    page_width: float,
    page_height: float,
    font_name: str,
) -> float:
    title_font_size = 13
    body_font_size = 11
    line_height = 15
    block_width = page_width - (2 * x_margin)
    current_y = y_position

    if current_y < 80:
        pdf.showPage()
        current_y = page_height - 50

    pdf.setFont(font_name, title_font_size)
    pdf.drawString(x_margin, current_y, title)
    current_y -= 20

    pdf.setFont(font_name, body_font_size)
    lines = content.splitlines() if content.strip() else [""]
    for paragraph in lines:
        wrapped = simpleSplit(paragraph, font_name, body_font_size, block_width)
        if not wrapped:
            wrapped = [""]
        for line in wrapped:
            if current_y < 60:
                pdf.showPage()
                current_y = page_height - 50
                pdf.setFont(font_name, body_font_size)
            pdf.drawString(x_margin, current_y, line)
            current_y -= line_height
        current_y -= 5

    return current_y - 8


def build_pdf_bytes(
    original_text: str,
    translated_text: str,
    target_language: str,
    source_filename: str,
) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    x_margin = 50
    y = page_height - 50
    font_name = get_pdf_font_name()

    pdf.setTitle("Audio Translation Report")
    pdf.setFont(font_name, 16)
    pdf.drawString(x_margin, y, "Audio Translation Report")
    y -= 24

    pdf.setFont(font_name, 10)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.drawString(x_margin, y, f"Generated: {timestamp}")
    y -= 14
    pdf.drawString(x_margin, y, f"Source file: {source_filename}")
    y -= 14
    pdf.drawString(x_margin, y, f"Target language: {target_language}")
    y -= 24

    y = draw_text_block(
        pdf=pdf,
        title="Original Transcript",
        content=original_text,
        x_margin=x_margin,
        y_position=y,
        page_width=page_width,
        page_height=page_height,
        font_name=font_name,
    )

    draw_text_block(
        pdf=pdf,
        title="Translation",
        content=translated_text,
        x_margin=x_margin,
        y_position=y,
        page_width=page_width,
        page_height=page_height,
        font_name=font_name,
    )

    pdf.save()
    return buffer.getvalue()


def validate_audio_size(size_bytes: int) -> str:
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            f"Audio file is {size_mb:.1f} MB, which exceeds {MAX_FILE_SIZE_MB} MB. "
            "Please upload a smaller file."
        )
    return ""


def build_download_pdf_name(original_filename: str) -> str:
    stem = Path(original_filename).stem or "audio"
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in stem
    )
    return f"{safe_stem}_translation.pdf"


def initialize_state() -> None:
    defaults = {
        "auth_ok": False,
        "processed_hash": None,
        "last_error": "",
        "pdf_bytes": b"",
        "pdf_filename": "translation.pdf",
        "google_working_model": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_login_gate() -> bool:
    expected_username, expected_password = resolve_login_credentials()
    if not expected_username or not expected_password:
        st.error("登录未配置：缺少 APP_USERNAME 或 APP_PASSWORD。")
        st.info("请在 .env 或 Streamlit Secrets 中添加 APP_USERNAME 和 APP_PASSWORD。")
        return False

    if st.session_state.auth_ok:
        with st.sidebar:
            st.success(f"已登录：{expected_username}")
            if st.button("退出登录", use_container_width=True):
                st.session_state.auth_ok = False
                st.rerun()
        return True

    st.markdown("### 欢迎使用^(*￣(oo)￣)^宝的智能小工具")
    with st.form("login_form", clear_on_submit=False):
        input_username = st.text_input("账号")
        input_password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", use_container_width=True)

    if submitted:
        username_ok = hmac.compare_digest(input_username.strip(), expected_username)
        password_ok = hmac.compare_digest(input_password, expected_password)
        if username_ok and password_ok:
            st.session_state.auth_ok = True
            st.rerun()
        st.error("账号或密码错误。")

    return False


def process_uploaded_audio(uploaded_audio: Any) -> None:
    audio_bytes = uploaded_audio.getvalue()
    if not audio_bytes:
        st.session_state.last_error = "Uploaded audio is empty."
        return

    file_hash = hashlib.sha256(audio_bytes).hexdigest()
    if st.session_state.processed_hash == file_hash and st.session_state.pdf_bytes:
        return

    st.session_state.last_error = ""
    st.session_state.pdf_bytes = b""
    st.session_state.pdf_filename = build_download_pdf_name(uploaded_audio.name)

    try:
        with st.spinner("Transcribing and translating with Google..."):
            transcript = transcribe_audio_with_google(
                audio_bytes=audio_bytes,
                filename=uploaded_audio.name,
            )
            translated = translate_text_with_google(
                text=transcript,
                target_language=TARGET_LANGUAGE,
            )
            pdf_bytes = build_pdf_bytes(
                original_text=transcript,
                translated_text=translated,
                target_language=TARGET_LANGUAGE,
                source_filename=uploaded_audio.name,
            )
    except Exception as exc:
        st.session_state.last_error = f"Processing failed: {exc}"
        return

    st.session_state.processed_hash = file_hash
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.last_error = ""


def main() -> None:
    st.set_page_config(page_title="Audio Translation to PDF", layout="centered")
    st.title("Audio Translation to PDF")
    st.caption("Upload audio and get an auto-translated PDF file.")

    initialize_state()
    if not render_login_gate():
        return

    if not resolve_google_api_key():
        st.error("GOOGLE_API_KEY is missing.")
        st.info("Please set GOOGLE_API_KEY in .env or Streamlit Cloud Secrets.")
        st.stop()

    uploaded_audio = st.file_uploader(
        "Upload audio",
        type=SUPPORTED_AUDIO_TYPES,
        accept_multiple_files=False,
    )

    if uploaded_audio is None:
        st.info("Please upload an audio file to start.")
        return

    size_error = validate_audio_size(uploaded_audio.size)
    if size_error:
        st.error(size_error)
        return

    st.audio(uploaded_audio)
    process_uploaded_audio(uploaded_audio)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)
        return

    if st.session_state.pdf_bytes:
        st.success("Done. PDF is ready to download.")
        st.download_button(
            "Download PDF",
            data=st.session_state.pdf_bytes,
            file_name=st.session_state.pdf_filename,
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
