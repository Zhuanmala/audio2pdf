# Audio Translation PDF (MVP)

This project is a minimal tool for:

1. Uploading audio
2. Auto speech-to-text transcription
3. Auto translation
4. Exporting a downloadable PDF report

## Stack

- Streamlit (web UI)
- Google Gemini API
  - Audio transcription
  - Text translation
- ReportLab for PDF generation

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
APP_USERNAME=your_login_username
APP_PASSWORD=your_login_password
```

4. Run the app:

```bash
streamlit run app.py
```

## Usage

- Login first with configured account/password.
- Upload one audio file (`mp3/mp4/m4a/wav/webm/...`)
- App automatically runs transcription and translation
- Download the generated PDF

## Public Deployment (Mobile Accessible)

This app is ready for **Streamlit Community Cloud** public deployment.

1. Push this project to a GitHub repository.
2. Go to `https://share.streamlit.io`.
3. Click **Create app** and select your repository/branch.
4. Main file path: `app.py`.
5. Open **Advanced settings**:
   - Python version: choose a compatible version (recommended: `3.12`)
   - In **Secrets**, set:

```toml
GOOGLE_API_KEY = "your_google_api_key"
APP_USERNAME = "your_login_username"
APP_PASSWORD = "your_login_password"
```

6. Deploy and wait for build to finish.
7. Open the generated `https://<your-app>.streamlit.app` URL on phone browser.

### Deployment Files Included

- `.streamlit/config.toml`: upload limit and usage stats settings.
- `.streamlit/secrets.toml.example`: secrets template.

## Notes

- App upload size limit is set to 100 MB.
- Current target language is configured in `app.py` with `TARGET_LANGUAGE`.
- Do not commit `.env` or real secrets to GitHub.
- If you see quota errors, check your Google AI API key quota/billing in Google Cloud.
