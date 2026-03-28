import os
from dataclasses import dataclass


def _get_setting(name: str, default: str = "") -> str:
	# Priority: OS env var, then Streamlit secrets, then default.
	value = os.getenv(name)
	if value is not None and str(value).strip() != "":
		return str(value).strip()

	try:
		import streamlit as st

		if name in st.secrets:
			secret_value = st.secrets[name]
			if secret_value is not None and str(secret_value).strip() != "":
				return str(secret_value).strip()
	except Exception:
		pass

	return default


@dataclass
class AppConfig:
	groq_api_key: str
	groq_model: str
	smtp_host: str
	smtp_port: int
	smtp_user: str
	smtp_password: str
	smtp_sender: str
	booking_domain_label: str
	history_limit: int


def load_config() -> AppConfig:
	return AppConfig(
		groq_api_key=_get_setting("GROQ_API_KEY", ""),
		groq_model=_get_setting("GROQ_MODEL", "llama-3.3-70b-versatile"),
		smtp_host=_get_setting("SMTP_HOST", "smtp.gmail.com"),
		smtp_port=int(_get_setting("SMTP_PORT", "587")),
		smtp_user=_get_setting("SMTP_USER", ""),
		smtp_password=_get_setting("SMTP_PASSWORD", ""),
		smtp_sender=_get_setting("SMTP_SENDER", ""),
		booking_domain_label=_get_setting("BOOKING_DOMAIN_LABEL", "service booking"),
		history_limit=int(_get_setting("HISTORY_LIMIT", "25")),
	)


