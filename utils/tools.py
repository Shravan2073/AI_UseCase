import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Dict, List, Tuple

from config.config import load_config
from db.database import create_booking, upsert_customer
from models.embeddings import format_context, retrieve_chunks
from models.llm import generate_response


def rag_tool(
    query: str,
    vector_store: Dict,
    chat_model,
    short_memory: List[Dict],
) -> str:
    chunks = retrieve_chunks(vector_store, query, top_k=4)
    context = format_context(chunks)

    if not context:
        system_prompt = (
            "You are a helpful booking assistant. "
            "If the user's query is not answerable from uploaded PDFs, respond naturally and "
            "ask if they want to ask about booking details or upload a more relevant PDF."
        )
        return generate_response(
            model=chat_model,
            system_prompt=system_prompt,
            memory_messages=short_memory,
            user_prompt=query,
            rag_context=None,
        )

    system_prompt = (
        "You are a helpful booking assistant. Answer using the retrieved context first. "
        "If context does not fully answer, say what is missing briefly."
    )
    return generate_response(
        model=chat_model,
        system_prompt=system_prompt,
        memory_messages=short_memory,
        user_prompt=query,
        rag_context=context,
    )


def booking_persistence_tool(payload: Dict) -> Tuple[bool, int, str]:
    try:
        customer_id = upsert_customer(
            name=payload["name"],
            email=payload["email"],
            phone=payload["phone"],
        )
        booking_id = create_booking(
            customer_id=customer_id,
            booking_type=payload["booking_type"],
            date=payload["date"],
            time=payload["time"],
            status="confirmed",
        )
        return True, booking_id, "Booking stored successfully."
    except Exception as exc:
        return False, -1, f"Database error: {exc}"


def email_tool(to_email: str, subject: str, body: str) -> Tuple[bool, str]:
    cfg = load_config()
    if not (cfg.smtp_user and cfg.smtp_password and cfg.smtp_sender):
        return False, "SMTP is not configured. Booking is saved, but email was not sent."

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("AI Booking Assistant", cfg.smtp_sender))
    msg["To"] = to_email

    try:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=20) as server:
            server.starttls()
            server.login(cfg.smtp_user, cfg.smtp_password)
            server.sendmail(cfg.smtp_sender, [to_email], msg.as_string())
        return True, "Confirmation email sent."
    except Exception as exc:
        return False, f"Email could not be sent, but booking was saved. Details: {exc}"
