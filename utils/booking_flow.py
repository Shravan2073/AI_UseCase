from typing import Dict, List, Tuple

from utils.chat_logic import extract_fields_from_text, validate_field

REQUIRED_FIELDS = ["name", "email", "phone", "booking_type", "date", "time"]

FIELD_PROMPTS = {
    "name": "What is your full name?",
    "email": "Please share your email address.",
    "phone": "Please share your phone number.",
    "booking_type": "What service would you like to book?",
    "date": "Please enter the date in YYYY-MM-DD format.",
    "time": "Please enter the time in 24-hour HH:MM format.",
}


class BookingFlow:
    def __init__(self, state: Dict):
        self.state = state
        self.state.setdefault("slots", {})
        self.state.setdefault("awaiting_confirmation", False)
        self.state.setdefault("active", False)
        self.state.setdefault("awaiting_edit_field_selection", False)
        self.state.setdefault("editing_field", None)

    def start(self) -> str:
        self.state["active"] = True
        return "Sure, I can help with a booking. " + self.ask_next_missing_field()

    def update_from_user(self, user_text: str) -> Tuple[bool, str]:
        # Step 1: user selected a field to edit and now must provide new value.
        if self.state.get("editing_field"):
            editing_field = self.state["editing_field"]
            extracted = extract_fields_from_text(user_text)
            candidate = extracted.get(editing_field, user_text.strip())
            if validate_field(editing_field, candidate):
                if editing_field in {"name", "booking_type"}:
                    candidate = candidate.title()
                self.state["slots"][editing_field] = candidate
                self.state["editing_field"] = None
                self.state["awaiting_confirmation"] = True
                return False, self.summary_for_confirmation()
            return False, self._invalid_field_message(editing_field)

        # Step 2: user said "no" and now must specify which field to edit.
        if self.state.get("awaiting_edit_field_selection"):
            field = self._resolve_field_name(user_text)
            if field is None:
                valid = ", ".join(REQUIRED_FIELDS)
                return False, f"Please choose a valid field to update: {valid}."
            self.state["awaiting_edit_field_selection"] = False
            self.state["editing_field"] = field
            return False, FIELD_PROMPTS[field]

        extracted = extract_fields_from_text(user_text)
        missing_before = self.get_missing_fields()
        for key, value in extracted.items():
            if key in REQUIRED_FIELDS and validate_field(key, value):
                self.state["slots"][key] = value

        # If extraction did not detect anything, treat the reply as a direct
        # answer to the next missing field (for example, plain full name text).
        if missing_before and not any(k in extracted for k in REQUIRED_FIELDS):
            next_field = missing_before[0]
            candidate = user_text.strip()
            if validate_field(next_field, candidate):
                if next_field in {"name", "booking_type"}:
                    candidate = candidate.title()
                self.state["slots"][next_field] = candidate

        if self.state.get("awaiting_confirmation"):
            lower = user_text.strip().lower()
            if lower in {"yes", "y", "confirm", "confirmed"}:
                return True, "confirmed"
            if lower in {"no", "n", "change", "edit"}:
                self.state["awaiting_confirmation"] = False
                self.state["awaiting_edit_field_selection"] = True
                return False, "No problem. Tell me the field you want to update."
            return False, "Please type 'confirm' to proceed or 'no' to modify details."

        missing = self.get_missing_fields()
        if missing:
            # User provided input but current slot is still invalid/missing.
            if missing_before and missing[0] == missing_before[0] and user_text.strip():
                candidate = extracted.get(missing[0], user_text.strip())
                if not validate_field(missing[0], candidate):
                    return False, self._invalid_field_message(missing[0])
            return False, self.ask_next_missing_field()

        self.state["awaiting_confirmation"] = True
        return False, self.summary_for_confirmation()

    def get_missing_fields(self) -> List[str]:
        return [field for field in REQUIRED_FIELDS if not self.state["slots"].get(field)]

    def ask_next_missing_field(self) -> str:
        missing = self.get_missing_fields()
        if not missing:
            return "All details collected."
        return FIELD_PROMPTS[missing[0]]

    def summary_for_confirmation(self) -> str:
        s = self.state["slots"]
        return (
            "Please confirm your booking details:\n"
            f"- Name: {s.get('name')}\n"
            f"- Email: {s.get('email')}\n"
            f"- Phone: {s.get('phone')}\n"
            f"- Booking type: {s.get('booking_type')}\n"
            f"- Date: {s.get('date')}\n"
            f"- Time: {s.get('time')}\n\n"
            "Type 'confirm' to save this booking or 'no' to change details."
        )

    def reset(self) -> None:
        self.state["active"] = False
        self.state["awaiting_confirmation"] = False
        self.state["awaiting_edit_field_selection"] = False
        self.state["editing_field"] = None
        self.state["slots"] = {}

    def _resolve_field_name(self, user_text: str):
        text = user_text.strip().lower()
        mapping = {
            "name": "name",
            "full name": "name",
            "email": "email",
            "mail": "email",
            "phone": "phone",
            "mobile": "phone",
            "booking": "booking_type",
            "booking type": "booking_type",
            "service": "booking_type",
            "type": "booking_type",
            "date": "date",
            "time": "time",
        }
        if text in mapping:
            return mapping[text]
        for key, value in mapping.items():
            if key in text:
                return value
        return None

    def _invalid_field_message(self, field: str) -> str:
        if field == "phone":
            return "Please enter a valid phone number with 10-15 digits (for example: +919876543210)."
        if field == "email":
            return "Please enter a valid email address (for example: name@example.com)."
        if field == "date":
            return "Please enter the date in YYYY-MM-DD format (for example: 2026-04-03)."
        if field == "time":
            return "Please enter the time in 24-hour HH:MM format (for example: 14:30)."
        return FIELD_PROMPTS.get(field, "Please provide a valid value.")
