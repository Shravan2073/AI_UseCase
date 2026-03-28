# AI Booking Assistant

Streamlit-based AI Booking Assistant that supports:

- PDF upload + RAG retrieval
- Booking intent detection
- Multi-turn slot filling
- Explicit confirmation before save
- SQLite persistence (`customers`, `bookings`)
- Email confirmation via SMTP
- Admin dashboard with search/filter
- Short-term memory (last 25 messages)

## Project Structure

```text
AI_UseCase/
  app.py
  requirements.txt
  README.md
  config/
    config.py
  models/
    embeddings.py
    llm.py
  utils/
    admin_dashboard.py
    booking_flow.py
    chat_logic.py
    tools.py
  db/
    database.py
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables (local shell) or Streamlit secrets (cloud):

```bash
GROQ_API_KEY=<your_groq_api_key>
GROQ_MODEL=llama-3.1-70b-versatile

SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASSWORD=<your_sendgrid_api_key>
SMTP_SENDER=<your_verified_sender_email>
```

PowerShell example for local run:

```powershell
$env:GROQ_API_KEY="<your_groq_api_key>"
$env:GROQ_MODEL="llama-3.1-70b-versatile"
$env:SMTP_HOST="smtp.sendgrid.net"
$env:SMTP_PORT="587"
$env:SMTP_USER="apikey"
$env:SMTP_PASSWORD="<your_sendgrid_api_key>"
$env:SMTP_SENDER="<your_verified_sender_email>"
streamlit run app.py
```

Notes:

- If `GROQ_API_KEY` is missing, app still works with basic fallback responses.
- If SMTP values are missing, booking is saved and app displays a graceful email failure message.

## Run Locally

```bash
streamlit run app.py
```

## How to Test Assignment Features

1. Upload one or more PDFs and click **Index Uploaded PDFs**.
2. Ask a content question from uploaded PDFs to verify RAG.
3. Start booking with a prompt like:

```text
I want to book a consultation.
```

4. Provide required fields:

- Name
- Email
- Phone
- Booking type
- Date (`YYYY-MM-DD`)
- Time (`HH:MM`)

5. Confirm with `confirm`.
6. Validate:

- Booking ID returned in chat
- Record appears in **Admin Dashboard**
- Email sent (if SMTP configured)

## Streamlit Cloud Deployment

1. Push this folder to a GitHub repository.
2. Deploy in Streamlit Cloud using `app.py` as entrypoint.
3. Add secrets in Streamlit Cloud settings:

```toml
GROQ_API_KEY="..."
GROQ_MODEL="llama-3.1-70b-versatile"
SMTP_HOST="smtp.gmail.com"
SMTP_PORT="587"
SMTP_USER="..."
SMTP_PASSWORD="..."
SMTP_SENDER="..."
```

Security note: never commit real keys in source files (README, app code, or Git history).

The app is designed to run even when SQLite resets between restarts.
