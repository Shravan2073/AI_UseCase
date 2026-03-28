import streamlit as st
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import load_config
from db.database import init_db
from models.embeddings import build_vector_store
from models.llm import get_chatgroq_model
from utils.admin_dashboard import render_admin_dashboard
from utils.booking_flow import BookingFlow
from utils.chat_logic import detect_intent
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


def handle_user_prompt(prompt: str, chat_model, cfg) -> str:
    flow = BookingFlow(st.session_state.booking)

    if st.session_state.booking.get("active"):
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
        return flow.start()

    vector_store = st.session_state.vector_store
    if vector_store.get("chunks"):
        return rag_tool(prompt, vector_store, chat_model, st.session_state.memory)
    return (
        "Please upload at least one PDF to enable RAG answers, or ask to create a booking. "
        "For example: 'I want to book a consultation on 2026-04-03 at 14:30'."
    )


def chat_page(chat_model, cfg):
    st.title("AI Booking Assistant")

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
        page_title="AI Booking Assistant",
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
                st.session_state.booking = {"active": False, "slots": {}, "awaiting_confirmation": False}
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