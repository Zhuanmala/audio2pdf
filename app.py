import hashlib
import hmac
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

load_dotenv()

MAX_FILE_SIZE_MB = 100
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
TRANSLATE_MODEL = "gpt-4o-mini"
TARGET_LANGUAGE = "Chinese (Simplified)"
SUPPORTED_AUDIO_TYPES = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]


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


def resolve_api_key() -> str:
    return resolve_config_value("OPENAI_API_KEY")


def resolve_login_credentials() -> tuple[str, str]:
    username = resolve_config_value("APP_USERNAME")
    password = resolve_config_value("APP_PASSWORD")
    return username, password


def get_openai_client() -> OpenAI:
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in .env or Streamlit Secrets."
        )
    return OpenAI(api_key=api_key)


def normalize_language_hint(source_language: Optional[str]) -> Optional[str]:
    if not source_language:
        return None
    normalized = source_language.strip().lower()
    if normalized in {"auto", "automatic", "detect"}:
        return None
    return normalized


def transcribe_audio(
    audio_bytes: bytes,
    filename: Optional[str],
    model: str,
    source_language: Optional[str] = None,
    prompt: str = "",
) -> str:
    suffix = Path(filename).suffix if filename else ".wav"
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        with open(temp_path, "rb") as audio_file:
            params = {"model": model, "file": audio_file}
            language_hint = normalize_language_hint(source_language)
            if language_hint:
                params["language"] = language_hint
            if prompt.strip():
                params["prompt"] = prompt.strip()

            result = get_openai_client().audio.transcriptions.create(**params)
            text = getattr(result, "text", None)
            if text is None:
                text = str(result)
            return text.strip()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def extract_response_text(result: object) -> str:
    output_text = getattr(result, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in getattr(result, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())

    if chunks:
        return "\n".join(chunks)
    return str(result).strip()


def translate_text(text: str, target_language: str, model: str) -> str:
    instructions = (
        "You are a translation engine. "
        "Translate user text into the requested target language. "
        "Keep meaning and line breaks. Return only translated text."
    )
    user_input = f"Target language: {target_language}\n\nText:\n{text}"

    result = get_openai_client().responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
    )
    return extract_response_text(result)


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


def validate_audio_size(size_bytes: int) -> Optional[str]:
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            f"Audio file is {size_mb:.1f} MB, which exceeds {MAX_FILE_SIZE_MB} MB. "
            "Please upload a smaller file or split it."
        )
    return None


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
        "transcript_text": "",
        "translated_text": "",
        "pdf_bytes": b"",
        "pdf_filename": "translation.pdf",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_login_gate() -> bool:
    expected_username, expected_password = resolve_login_credentials()
    if not expected_username or not expected_password:
        st.error("登录未配置：缺少 APP_USERNAME 或 APP_PASSWORD。")
        st.info(
            "请在 .env 或 Streamlit Secrets 中添加 APP_USERNAME 和 APP_PASSWORD。"
        )
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


def process_uploaded_audio(uploaded_audio) -> None:
    audio_bytes = uploaded_audio.getvalue()
    if not audio_bytes:
        st.session_state.last_error = "Uploaded audio is empty."
        return

    file_hash = hashlib.sha256(audio_bytes).hexdigest()
    if st.session_state.processed_hash == file_hash and st.session_state.pdf_bytes:
        return

    st.session_state.last_error = ""
    st.session_state.transcript_text = ""
    st.session_state.translated_text = ""
    st.session_state.pdf_bytes = b""
    st.session_state.pdf_filename = build_download_pdf_name(uploaded_audio.name)

    try:
        with st.spinner("Transcribing and translating..."):
            transcript = transcribe_audio(
                audio_bytes=audio_bytes,
                filename=uploaded_audio.name,
                model=TRANSCRIBE_MODEL,
                source_language=None,
                prompt="",
            )
            translated = translate_text(
                text=transcript,
                target_language=TARGET_LANGUAGE,
                model=TRANSLATE_MODEL,
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
    st.session_state.transcript_text = transcript
    st.session_state.translated_text = translated
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.last_error = ""


def main() -> None:
    st.set_page_config(page_title="Audio Translator PDF", layout="centered")
    st.title("Audio Translation to PDF")
    st.caption("Upload audio and get an auto-translated PDF file.")

    initialize_state()
    if not render_login_gate():
        return

    api_key = resolve_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is missing.")
        st.info(
            "Local: create .env with OPENAI_API_KEY=... | "
            "Streamlit Cloud: set OPENAI_API_KEY in app Secrets."
        )
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
