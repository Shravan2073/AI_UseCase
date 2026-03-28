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

    def start(self) -> str:
        self.state["active"] = True
        return "Sure, I can help with a booking. " + self.ask_next_missing_field()

    def update_from_user(self, user_text: str) -> Tuple[bool, str]:
        extracted = extract_fields_from_text(user_text)
        for key, value in extracted.items():
            if key in REQUIRED_FIELDS and validate_field(key, value):
                self.state["slots"][key] = value

        if self.state.get("awaiting_confirmation"):
            lower = user_text.strip().lower()
            if lower in {"yes", "y", "confirm", "confirmed"}:
                return True, "confirmed"
            if lower in {"no", "n", "change", "edit"}:
                self.state["awaiting_confirmation"] = False
                return False, "No problem. Tell me the field you want to update."
            return False, "Please type 'confirm' to proceed or 'no' to modify details."

        missing = self.get_missing_fields()
        if missing:
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
        self.state["slots"] = {}
