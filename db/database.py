import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "bookings.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                booking_type TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_customer(name: str, email: str, phone: str) -> int:
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT customer_id FROM customers WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE customers SET name = ?, phone = ? WHERE customer_id = ?",
                (name, phone, existing["customer_id"]),
            )
            conn.commit()
            return int(existing["customer_id"])

        cursor = conn.execute(
            "INSERT INTO customers (name, email, phone) VALUES (?, ?, ?)",
            (name, email, phone),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def create_booking(
    customer_id: int,
    booking_type: str,
    date: str,
    time: str,
    status: str = "confirmed",
) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO bookings (customer_id, booking_type, date, time, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (customer_id, booking_type, date, time, status),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def list_bookings(
    name_query: Optional[str] = None,
    email_query: Optional[str] = None,
    date_query: Optional[str] = None,
) -> List[Dict]:
    conn = get_connection()
    try:
        sql = """
            SELECT
                b.id,
                c.name,
                c.email,
                c.phone,
                b.booking_type,
                b.date,
                b.time,
                b.status,
                b.created_at
            FROM bookings b
            JOIN customers c ON c.customer_id = b.customer_id
            WHERE 1 = 1
        """
        params = []

        if name_query:
            sql += " AND lower(c.name) LIKE ?"
            params.append(f"%{name_query.lower()}%")
        if email_query:
            sql += " AND lower(c.email) LIKE ?"
            params.append(f"%{email_query.lower()}%")
        if date_query:
            sql += " AND b.date = ?"
            params.append(date_query)

        sql += " ORDER BY b.created_at DESC"

        rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
