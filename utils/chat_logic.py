import re
from datetime import datetime
from typing import Dict

BOOKING_KEYWORDS = (
    "book",
    "booking",
    "appointment",
    "reserve",
    "reservation",
    "schedule",
)


def detect_intent(user_text: str) -> str:
    text = user_text.lower()
    if any(keyword in text for keyword in BOOKING_KEYWORDS):
        return "booking"
    return "general"


def extract_fields_from_text(user_text: str) -> Dict[str, str]:
    text = user_text.strip()
    lower = text.lower()
    fields: Dict[str, str] = {}

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if email_match:
        fields["email"] = email_match.group(0)

    phone_candidates = re.findall(r"(?:\+?\d[\d\-\s()]{8,}\d)", text)
    for cand in phone_candidates:
        stripped = cand.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
            continue
        digits = re.sub(r"\D", "", stripped)
        if 10 <= len(digits) <= 15:
            fields["phone"] = re.sub(r"\s+", "", stripped)
            break

    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if date_match:
        fields["date"] = date_match.group(0)

    time_24_match = re.search(r"\b([01]\d|2[0-3]):[0-5]\d\b", text)
    time_12_match = re.search(r"\b(1[0-2]|0?[1-9]):([0-5]\d)\s?(am|pm)\b", lower)
    if time_24_match:
        fields["time"] = time_24_match.group(0)
    elif time_12_match:
        hour = int(time_12_match.group(1))
        minute = time_12_match.group(2)
        suffix = time_12_match.group(3)
        if suffix == "pm" and hour != 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        fields["time"] = f"{hour:02d}:{minute}"

    name_match = re.search(r"(?:my name is|i am|this is)\s+([A-Za-z][A-Za-z\s'-]{1,40})", lower)
    if name_match:
        fields["name"] = name_match.group(1).strip().title()

    bt_match = re.search(r"(?:for|book|booking|appointment for)\s+([A-Za-z][A-Za-z\s-]{2,40})", lower)
    if bt_match:
        value = bt_match.group(1).strip()
        value = re.split(r"\b(on|at|for|with|tomorrow|today)\b", value)[0].strip()
        fields["booking_type"] = value.title()

    return fields


def validate_field(field: str, value: str) -> bool:
    if not value:
        return False
    if field == "email":
        return re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value) is not None
    if field == "date":
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    if field == "phone":
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
            return False
        digits = re.sub(r"\D", "", value)
        return 10 <= len(digits) <= 15
    if field == "time":
        try:
            datetime.strptime(value, "%H:%M")
            return True
        except ValueError:
            return False
    return True
