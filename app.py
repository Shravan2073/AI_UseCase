import streamlit as st
import os
import re
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import load_config
from db.database import init_db
from models.embeddings import build_vector_store
from models.llm import get_chatgroq_model
from utils.admin_dashboard import render_admin_dashboard
from utils.booking_flow import BookingFlow
from utils.chat_logic import detect_intent, validate_field
from utils.tools import booking_persistence_tool, email_tool, rag_tool

def instructions_page():
    st.title("AI Booking Assistant")
    st.markdown("Setup guide for running the assignment project locally.")
    
    st.markdown("""
    ## 🔧 Installation
                
    
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## Environment Variables

    Create a `.env` or set Streamlit Cloud secrets with:

    - `GROQ_API_KEY` (optional but recommended)
    - `GROQ_MODEL` (default: `llama-3.1-70b-versatile`)
    - `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_SENDER` (for email)

    If SMTP values are missing, booking is still saved and app shows a friendly email failure message.

    ## Assignment Features Included

    - PDF upload + RAG retrieval
    - Booking intent detection and slot collection
    - Explicit confirmation before DB insert
    - SQLite persistence (`customers` and `bookings`)
    - Email confirmation tool (SMTP)
    - Admin dashboard with filters
    - Short-term memory in session state (last 25 messages)
    
    ---
    Navigate to **Chat** to start and **Admin Dashboard** to inspect bookings.
    """)

def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("memory", [])
    st.session_state.setdefault("vector_store", {"chunks": [], "idf": {}, "chunk_vectors": []})
    st.session_state.setdefault("booking", {"active": False, "slots": {}, "awaiting_confirmation": False})


def append_message(role: str, content: str, history_limit: int) -> None:
    st.session_state.messages.append({"role": role, "content": content})
    st.session_state.memory.append({"role": role, "content": content})
    st.session_state.memory = st.session_state.memory[-history_limit:]


def _wants_auto_extract(user_text: str) -> bool:
    text = user_text.lower()
    extract_intent = any(
        phrase in text
        for phrase in [
            "extract",
            "auto fill",
            "autofill",
            "take details",
            "read details",
            "use details",
            "from pdf",
            "form the pdf",
        ]
    )
    source_hint = any(token in text for token in ["pdf", "ticket", "details", "browser", "document"])
    booking_hint = any(token in text for token in ["book", "booking", "slot", "appointment"])
    return (extract_intent and source_hint) or (source_hint and booking_hint)


def _extract_booking_slots_from_pdf_chunks(vector_store: dict) -> dict:
    chunks = vector_store.get("chunks", []) if vector_store else []
    blob = "\n".join(chunks[:60])
    if not blob.strip():
        return {}

    slots = {}

    def normalize_date(value: str):
        text = value.strip().replace(",", " ")
        text = re.sub(r"\s+", " ", text)
        formats = [
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d.%m.%Y",
            "%d %b %Y",
            "%d %B %Y",
            "%b %d %Y",
            "%B %d %Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
        return None

    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", blob)
    if email:
        slots["email"] = email.group(0)

    phone_candidates = re.findall(r"(?:\+?\d[\d\-\s()]{8,}\d)", blob)
    for cand in phone_candidates:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cand.strip()):
            continue
        digits = re.sub(r"\D", "", cand)
        if 10 <= len(digits) <= 15:
            slots["phone"] = cand.strip()
            break

    # Prefer departure/travel/event labeled date if present.
    labeled_date = re.search(
        r"(?im)^\s*(departure\s*date|travel\s*date|journey\s*date|date)\s*[:\-]\s*([^\n]+)$",
        blob,
    )
    if labeled_date:
        candidate = labeled_date.group(2).strip()
        m = re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}[\-/\.]\d{2}[\-/\.]\d{4}\b|\b(?:\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2}\s+\d{4})\b", candidate)
        if m:
            parsed = normalize_date(m.group(0))
            if parsed:
                slots["date"] = parsed

    if "date" not in slots:
        date_candidates = re.findall(
            r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}[\-/\.]\d{2}[\-/\.]\d{4}\b|\b(?:\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2}\s+\d{4})\b",
            blob,
        )
        for cand in date_candidates:
            parsed = normalize_date(cand)
            if parsed:
                slots["date"] = parsed
                break

    # Prefer departure/travel/event labeled time if present.
    labeled_time = re.search(
        r"(?im)^\s*(departure\s*time|travel\s*time|journey\s*time|time)\s*[:\-]\s*([^\n]+)$",
        blob,
    )
    if labeled_time:
        candidate = labeled_time.group(2).strip().lower()
        t24 = re.search(r"\b([01]\d|2[0-3]):[0-5]\d\b", candidate)
        t12 = re.search(r"\b(1[0-2]|0?[1-9]):([0-5]\d)\s?(am|pm)\b", candidate)
        if t24:
            slots["time"] = t24.group(0)
        elif t12:
            hour = int(t12.group(1))
            minute = t12.group(2)
            suffix = t12.group(3)
            if suffix == "pm" and hour != 12:
                hour += 12
            if suffix == "am" and hour == 12:
                hour = 0
            slots["time"] = f"{hour:02d}:{minute}"

    if "time" not in slots:
        time = re.search(r"\b([01]\d|2[0-3]):[0-5]\d\b", blob)
        if time:
            slots["time"] = time.group(0)

    # Labeled fields are safest for name and booking type extraction.
    name = re.search(r"(?im)^\s*(name|passenger name|customer name)\s*[:\-]\s*([A-Za-z][A-Za-z\s'\.-]{2,60})\s*$", blob)
    if name:
        slots["name"] = name.group(2).strip().title()

    btype = re.search(r"(?im)^\s*(booking type|service|ticket type)\s*[:\-]\s*([A-Za-z][A-Za-z\s\-/]{2,60})\s*$", blob)
    if btype:
        slots["booking_type"] = btype.group(2).strip().title()

    return slots


def _refers_to_pdf_values(user_text: str) -> bool:
    text = user_text.lower()
    return any(
        phrase in text
        for phrase in [
            "same as",
            "same day",
            "same time",
            "in the pdf",
            "in pdf",
            "on the ticket",
            "departure date",
            "departure time",
        ]
    )


def handle_user_prompt(prompt: str, chat_model, cfg) -> str:
    flow = BookingFlow(st.session_state.booking)
    vector_store = st.session_state.vector_store

    # Allow "extract from PDF and book" to work on the very first message.
    if not st.session_state.booking.get("active") and _wants_auto_extract(prompt):
        st.session_state.booking["active"] = True
        lower = prompt.lower()
        if "browser" in lower and "pdf" not in lower and "ticket" not in lower:
            return (
                "I cannot access your browser data directly for privacy reasons. "
                "Please upload your ticket PDF, and I can extract booking details from it."
            )

        if vector_store.get("chunks"):
            extracted = _extract_booking_slots_from_pdf_chunks(vector_store)
            updated_fields = []
            for key, value in extracted.items():
                if st.session_state.booking["slots"].get(key):
                    continue
                if validate_field(key, value):
                    st.session_state.booking["slots"][key] = value
                    updated_fields.append(key)

            if updated_fields:
                missing = flow.get_missing_fields()
                if missing:
                    return (
                        f"I extracted: {', '.join(updated_fields)}. "
                        f"I still need: {', '.join(missing)}. {flow.ask_next_missing_field()}"
                    )
                flow.state["awaiting_confirmation"] = True
                return "I extracted all required details from the PDF.\n\n" + flow.summary_for_confirmation()

        return "Sure, I can help with a booking. " + flow.ask_next_missing_field()

    if st.session_state.booking.get("active"):
        lower = prompt.lower()
        if _wants_auto_extract(prompt):
            if "browser" in lower and "pdf" not in lower and "ticket" not in lower:
                return (
                    "I cannot access your browser data directly for privacy reasons. "
                    "Please upload your ticket PDF, and I can try to extract booking details from it."
                )

            vector_store = st.session_state.vector_store
            if not vector_store.get("chunks"):
                return "Please upload and index your ticket PDF first, then ask me to extract details."

            extracted = _extract_booking_slots_from_pdf_chunks(vector_store)
            updated_fields = []
            for key, value in extracted.items():
                if key in flow.state["slots"] and flow.state["slots"].get(key):
                    continue
                if validate_field(key, value):
                    flow.state["slots"][key] = value
                    updated_fields.append(key)

            if not updated_fields:
                return (
                    "I could not reliably extract details from the uploaded PDF. "
                    "Please provide them one by one. " + flow.ask_next_missing_field()
                )

            missing = flow.get_missing_fields()
            if missing:
                fields_text = ", ".join(updated_fields)
                return f"I extracted: {fields_text}. I still need: {', '.join(missing)}. {flow.ask_next_missing_field()}"

            flow.state["awaiting_confirmation"] = True
            return "I extracted all required details from the PDF.\n\n" + flow.summary_for_confirmation()

        if _refers_to_pdf_values(prompt):
            vector_store = st.session_state.vector_store
            if vector_store.get("chunks"):
                extracted = _extract_booking_slots_from_pdf_chunks(vector_store)
                missing_before = flow.get_missing_fields()
                updated_fields = []
                for field in missing_before:
                    value = extracted.get(field)
                    if value and validate_field(field, value):
                        flow.state["slots"][field] = value
                        updated_fields.append(field)

                if updated_fields:
                    missing_now = flow.get_missing_fields()
                    if missing_now:
                        return (
                            f"I used PDF values for: {', '.join(updated_fields)}. "
                            f"I still need: {', '.join(missing_now)}. {flow.ask_next_missing_field()}"
                        )
                    flow.state["awaiting_confirmation"] = True
                    return "I filled all missing details from the PDF.\n\n" + flow.summary_for_confirmation()

        confirmed, flow_response = flow.update_from_user(prompt)
        if confirmed and flow_response == "confirmed":
            payload = st.session_state.booking["slots"]
            ok, booking_id, db_message = booking_persistence_tool(payload)
            if not ok:
                flow.reset()
                return db_message

            email_subject = f"Booking Confirmation #{booking_id}"
            email_body = (
                f"Hello {payload['name']},\n\n"
                f"Your booking is confirmed.\n"
                f"Booking ID: {booking_id}\n"
                f"Type: {payload['booking_type']}\n"
                f"Date: {payload['date']}\n"
                f"Time: {payload['time']}\n\n"
                "Thank you."
            )
            email_ok, email_msg = email_tool(payload["email"], email_subject, email_body)
            flow.reset()
            if email_ok:
                return f"Booking saved successfully with ID {booking_id}. {email_msg}"
            return f"Booking saved successfully with ID {booking_id}. {email_msg}"

        return flow_response

    intent = detect_intent(prompt)
    if intent == "booking":
        # If user references ticket/PDF-derived values on the first booking message,
        # attempt to prefill from indexed PDFs before asking manual slot questions.
        if _refers_to_pdf_values(prompt) or _wants_auto_extract(prompt):
            st.session_state.booking["active"] = True
            if not vector_store.get("chunks"):
                return (
                    "I can use your PDF details, but I need indexed PDF text first. "
                    "Please upload the file and click 'Index Uploaded PDFs', then ask again."
                )

            extracted = _extract_booking_slots_from_pdf_chunks(vector_store)
            updated_fields = []
            for key, value in extracted.items():
                if validate_field(key, value):
                    st.session_state.booking["slots"][key] = value
                    updated_fields.append(key)

            if updated_fields:
                missing = flow.get_missing_fields()
                if missing:
                    return (
                        f"I extracted: {', '.join(updated_fields)}. "
                        f"I still need: {', '.join(missing)}. {flow.ask_next_missing_field()}"
                    )
                flow.state["awaiting_confirmation"] = True
                return "I extracted all required details from the PDF.\n\n" + flow.summary_for_confirmation()

            return (
                "I could not detect required booking fields from your indexed PDF yet. "
                "Please share missing details manually. " + flow.ask_next_missing_field()
            )

        return flow.start()

    if vector_store.get("chunks"):
        return rag_tool(prompt, vector_store, chat_model, st.session_state.memory)
    return (
        "Please upload at least one PDF to enable RAG answers, or ask to create a booking. "
        "For example: 'I want to book a consultation on 2026-04-03 at 14:30'."
    )


def chat_page(chat_model, cfg):
    st.title("EzBooking: AI Booking Assistant")

    with st.expander("Upload PDFs for RAG", expanded=True):
        uploads = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Index Uploaded PDFs", use_container_width=True):
            if not uploads:
                st.warning("Please upload at least one valid PDF.")
            else:
                try:
                    st.session_state.vector_store = build_vector_store(uploads)
                    chunk_count = len(st.session_state.vector_store.get("chunks", []))
                    if chunk_count == 0:
                        st.error("No readable text found in PDFs.")
                    else:
                        st.success(f"Indexed {chunk_count} text chunks for retrieval.")
                except Exception as exc:
                    st.error(f"Failed to process PDFs: {exc}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask questions from PDFs or start booking...")
    if not prompt:
        return

    append_message("user", prompt, cfg.history_limit)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = handle_user_prompt(prompt, chat_model, cfg)
            st.markdown(response)
    append_message("assistant", response, cfg.history_limit)

def main():
    st.set_page_config(
        page_title="EzBooking",
        page_icon="📅",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    cfg = load_config()
    init_db()
    init_session_state()
    chat_model = get_chatgroq_model()
    
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Admin Dashboard", "Instructions"],
            index=0
        )
        
        # Add clear chat button in sidebar for chat page
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.memory = []
                st.session_state.booking = {
                    "active": False,
                    "slots": {},
                    "awaiting_confirmation": False,
                    "awaiting_edit_field_selection": False,
                    "editing_field": None,
                }
                st.rerun()
    
    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page(chat_model, cfg)
    if page == "Admin Dashboard":
        render_admin_dashboard()

if __name__ == "__main__":
    main()