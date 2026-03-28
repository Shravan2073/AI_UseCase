import streamlit as st

from db.database import list_bookings


def render_admin_dashboard() -> None:
    st.subheader("Admin Dashboard")
    st.caption("View and search all stored bookings")

    c1, c2, c3 = st.columns(3)
    name_query = c1.text_input("Search by name")
    email_query = c2.text_input("Search by email")
    date_filter = c3.text_input("Filter by date (YYYY-MM-DD)")

    rows = list_bookings(
        name_query=name_query or None,
        email_query=email_query or None,
        date_query=date_filter or None,
    )

    if not rows:
        st.info("No bookings found for current filters.")
        return

    st.dataframe(rows, use_container_width=True)
