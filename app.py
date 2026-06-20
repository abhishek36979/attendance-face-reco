import csv
import sys
from datetime import date
from pathlib import Path

import cv2

import database as db
import face_manager as fm

EXPORT_DIR = Path(__file__).parent / "data" / "exports"


def clear_screen():
    print("\n" * 2)


def print_header(title: str):
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)


def register_student():
    print_header("Register New Student")
    student_id = input("Student ID / Roll No: ").strip()
    name = input("Full Name: ").strip()
    email = input("Email (optional): ").strip()

    if not student_id or not name:
        print("Student ID and name are required.")
        return

    if db.get_student(student_id):
        print(f"Student {student_id} already exists.")
        return

    print("\nOpening camera to capture face...")
    print("Keep face centered, use good lighting, and slowly turn head left/right.\n")

    success, message = fm.capture_face_samples(student_id)
    if not success:
        print(f"Face capture failed: {message}")
        return

    if not db.add_student(student_id, name, email):
        print("Failed to save student info.")
        return

    print(message)
    print("Training face model...")
    ok, train_msg = fm.train_model()
    print(train_msg if ok else f"Training failed: {train_msg}")

    if ok:
        print(f"\nStudent '{name}' registered successfully!")
        print(f"Face ID in system: {student_id}")
        print("Use the SAME Student ID when marking attendance.")


def resolve_display_name(student_id: str | None, name_map: dict) -> str:
    if not student_id:
        return "Unknown"
    return name_map.get(student_id, student_id.replace("_", " ").title())


def mark_attendance():
    print_header("Mark Attendance")
    recognizer, label_map = fm.load_model()
    if recognizer is None:
        print("No trained model found. Register students first.")
        return

    name_map = {s["student_id"]: s["name"] for s in db.get_all_students()}
    face_ids = fm.get_registered_face_ids()
    missing_faces = [sid for sid in name_map if sid not in face_ids]
    if missing_faces:
        print("WARNING: These students have no face photos - re-register them:")
        for sid in missing_faces:
            print(f"  - {sid} ({name_map[sid]})")
        print()

    marked_today = db.get_marked_student_ids()
    if marked_today:
        names = ", ".join(resolve_display_name(sid, name_map) for sid in marked_today)
        print(f"Already marked today: {names}\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    tracker = fm.RecognitionTracker(require_blink=True)
    last_status = "Look at camera, then BLINK once to mark attendance"

    print("Step 1: Face is recognized and your name appears on the box")
    print("Step 2: BLINK once to verify you are live")
    print("Step 3: Attendance is marked automatically")
    print("Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = fm.detect_faces(gray)

        if faces:
            bbox = faces[0]
            x, y, w, h = bbox

            result = tracker.update(gray, frame, bbox, recognizer, label_map)
            student_id = result["student_id"]
            display_id = result.get("display_id") or result.get("best_id")
            live = result["live"]
            live_msg = result["live_msg"]
            is_spoof = result.get("is_spoof", False)
            blink_done = result.get("blink_done", False)
            confidence = result.get("confidence", 999)
            name = resolve_display_name(display_id, name_map)

            if is_spoof:
                label = f"BLOCKED: {live_msg}"
                if display_id:
                    label = f"BLOCKED: {name} - {live_msg}"
                color = (0, 0, 220)
                last_status = live_msg
            elif display_id and display_id in marked_today:
                label = f"{name} - Already marked"
                color = (0, 165, 255)
                last_status = f"{name} already marked today"
            elif student_id and live and blink_done:
                ok, mark_result = db.mark_attendance(student_id)
                if ok:
                    marked_today.add(student_id)
                    label = f"{name} - Present"
                    color = (0, 200, 0)
                    last_status = f"Marked: {name}"
                    print(f"[OK] {mark_result} marked present")
                else:
                    marked_today.add(student_id)
                    label = f"{name} - Already marked"
                    color = (0, 165, 255)
                    last_status = f"{name} already marked today"
            elif display_id and not blink_done:
                label = f"{name} - Blink once"
                color = (0, 255, 255)
                last_status = "Blink once to mark attendance"
            elif display_id:
                label = f"{name} - {live_msg}"
                color = (0, 255, 255)
                last_status = live_msg
            else:
                label = "Unknown"
                color = (0, 0, 255)
                last_status = "Face not registered - use option 1 to register"

            fm.draw_face_box(display, x, y, w, h, label, color)

            conf_text = f"Match: {confidence:.0f}" if display_id else "No match"
            cv2.putText(
                display,
                conf_text,
                (x, y + h + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
        else:
            tracker.reset()
            last_status = "No face detected - move closer to camera"

        cv2.putText(
            display,
            last_status,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            f"{date.today().isoformat()} | Marked: {len(marked_today)} | BLINK required | Q=Quit",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Mark Attendance", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession complete. {len(marked_today)} attendance record(s) added.")


def view_students():
    print_header("Registered Students")
    students = db.get_all_students()
    if not students:
        print("No students registered yet.")
        return

    for s in students:
        print(f"  {s['student_id']:12} | {s['name']:25} | {s['email'] or '-'}")


def view_today_attendance():
    print_header(f"Attendance - {date.today().isoformat()}")
    records = db.get_attendance_by_date()
    if not records:
        print("No attendance marked today.")
        return

    for r in records:
        print(f"  {r['time']} | {r['student_id']:12} | {r['name']:25} | {r['status']}")


def view_student_history():
    print_header("Student Attendance History")
    student_id = input("Enter Student ID: ").strip()
    student = db.get_student(student_id)
    if not student:
        print("Student not found.")
        return

    print(f"\nHistory for {student['name']} ({student_id}):\n")
    history = db.get_student_attendance_history(student_id)
    if not history:
        print("No attendance records found.")
        return

    for r in history:
        print(f"  {r['date']} {r['time']} - {r['status']}")


def export_attendance():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_date = input(f"Date to export (YYYY-MM-DD, blank=today): ").strip()
    if not export_date:
        export_date = date.today().isoformat()

    records = db.get_attendance_by_date(export_date)
    filename = EXPORT_DIR / f"attendance_{export_date}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "time", "student_id", "name", "email", "status"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Exported {len(records)} record(s) to {filename}")


def retrain_model():
    print_header("Retrain Face Model")
    ok, message = fm.train_model()
    print(message if ok else f"Failed: {message}")


def main_menu():
    db.init_db()

    while True:
        clear_screen()
        print_header("Face Attendance System")
        print("  1. Register new student (capture face)")
        print("  2. Mark attendance")
        print("  3. View registered students")
        print("  4. View today's attendance")
        print("  5. View student attendance history")
        print("  6. Export attendance to CSV")
        print("  7. Retrain face model")
        print("  0. Exit")
        print()

        choice = input("Choose option: ").strip()

        if choice == "1":
            register_student()
        elif choice == "2":
            mark_attendance()
        elif choice == "3":
            view_students()
        elif choice == "4":
            view_today_attendance()
        elif choice == "5":
            view_student_history()
        elif choice == "6":
            export_attendance()
        elif choice == "7":
            print("\nRetrain after updates or if recognition is weak.")
            retrain_model()
        elif choice == "0":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main_menu()
