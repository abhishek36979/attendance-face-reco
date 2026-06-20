import sqlite3
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "attendance.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'present',
                UNIQUE(student_id, date),
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            );
            """
        )


def add_student(student_id: str, name: str, email: str = "") -> bool:
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO students (student_id, name, email, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, name, email, datetime.now().isoformat()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_student(student_id: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_students():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM students ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]


def mark_attendance(student_id: str, status: str = "present") -> tuple[bool, str]:
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM attendance WHERE student_id = ? AND date = ?",
            (student_id, today),
        ).fetchone()
        if existing:
            return False, "Already marked for today"

        student = conn.execute(
            "SELECT name FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        if not student:
            return False, "Student not found"

        conn.execute(
            """
            INSERT INTO attendance (student_id, date, time, status)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, today, now, status),
        )
        return True, student["name"]


def get_attendance_by_date(attendance_date: str | None = None):
    attendance_date = attendance_date or date.today().isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT a.date, a.time, a.status, s.student_id, s.name, s.email
            FROM attendance a
            JOIN students s ON s.student_id = a.student_id
            WHERE a.date = ?
            ORDER BY a.time
            """,
            (attendance_date,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_marked_student_ids(attendance_date: str | None = None) -> set[str]:
    attendance_date = attendance_date or date.today().isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT student_id FROM attendance WHERE date = ?",
            (attendance_date,),
        ).fetchall()
        return {row["student_id"] for row in rows}


def get_student_attendance_history(student_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT date, time, status FROM attendance
            WHERE student_id = ?
            ORDER BY date DESC, time DESC
            """,
            (student_id,),
        ).fetchall()
        return [dict(r) for r in rows]
