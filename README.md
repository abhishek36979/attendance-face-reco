# Face Attendance System

An OpenCV-based attendance application that registers students using face capture, recognizes them from a webcam, marks daily attendance, and stores all data for later use.

---

## What This Project Does

| Feature | Description |
|--------|-------------|
| **Register students** | Capture face samples from webcam and save student details |
| **Mark attendance** | Recognize faces, show name on box, block photo/phone spoofing |
| **Anti-spoofing** | Detects printed photos and phone screens; requires blink or natural movement |
| **View records** | See registered students and today's attendance |
| **History** | View attendance history for any student |
| **Export** | Download attendance as CSV files |
| **Retrain model** | Rebuild the face recognition model after adding students |

---

## Requirements

### Hardware
- A computer with a **working webcam**
- Good lighting (helps face detection and recognition)

### Software
- **Python 3.9 or newer** (3.10+ recommended)
- **pip** (Python package installer — included with Python)

Check your Python version:

```powershell
python --version
```

If `python` does not work, try:

```powershell
py --version
```

---

## Dependencies

These Python packages are required:

| Package | Purpose |
|---------|---------|
| **opencv-contrib-python** | Webcam access, face detection, and face recognition (LBPH) |
| **numpy** | Image and array processing (used by OpenCV) |

Built-in Python modules (no install needed):
- `sqlite3` — database storage
- `csv` — export attendance
- `pathlib`, `datetime`, `pickle` — file and data handling

---

## Installation

### Step 1: Open the project folder

```powershell
cd "c:\Users\VORA\OneDrive\Desktop\attendance"
```

### Step 2: (Recommended) Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error on Windows, run this once in PowerShell (as Administrator if needed):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs:
- `opencv-contrib-python>=4.8.0`
- `numpy>=1.24.0`

### Step 4: Verify installation

```powershell
python -c "import cv2; print('OpenCV version:', cv2.__version__); print('Face module OK:', hasattr(cv2, 'face'))"
```

You should see the OpenCV version and `Face module OK: True`.

> **Important:** Use `opencv-contrib-python`, not plain `opencv-python`. The face recognizer (`cv2.face`) is only available in the contrib package.

---

## How to Run

From the project folder:

```powershell
python app.py
```

You will see the main menu:

```
==================================================
  Face Attendance System
==================================================
  1. Register new student (capture face)
  2. Mark attendance
  3. View registered students
  4. View today's attendance
  5. View student attendance history
  6. Export attendance to CSV
  7. Retrain face model
  0. Exit
```

---

## How to Use (Step by Step)

### 1. Register a new student

1. Choose option **1**
2. Enter:
   - **Student ID / Roll No** (e.g. `101`, `CS2024001`)
   - **Full Name**
   - **Email** (optional — press Enter to skip)
3. The webcam window opens
4. Look at the camera and keep your face inside the green box
5. The app captures about **30 face samples** automatically
6. When done, the face model is trained and the student is saved

**Tips for registration:**
- Sit in front of the camera with good lighting
- Avoid hats, masks, or heavy shadows on your face
- Register each person only once (use a unique Student ID)

Press **Q** in the camera window to cancel registration.

---

### 2. Mark attendance

1. Choose option **2**
2. The webcam opens
3. Each registered student shows their **live face** to the camera
4. A box appears around the face with a label:
   - **Green** — `[Name] - Present` (attendance marked)
   - **Orange** — `[Name] - Already marked`
   - **Yellow** — `Verifying [Name]...` or waiting for liveness check
   - **Red** — `Unknown` (face not registered)
   - **Dark red** — `BLOCKED: Phone screen detected` / `Flat print detected` (spoof blocked)
5. **Blink once** or **move your head slightly** so the system knows you are live (not a photo)
6. Press **Q** to close the camera and finish

**Anti-spoofing (phone/photo protection):**
- Holding a phone with someone's photo will **NOT** mark attendance
- The system checks for blinks, natural movement, flat prints, and screen glow
- You must use your real face in front of the camera

**Rules:**
- Each student can be marked **only once per day**
- Unknown faces are never marked
- Recognition uses multiple frames for better accuracy (not a single snapshot)

---

### 3. View registered students

Choose option **3** to see all students with ID, name, and email.

---

### 4. View today's attendance

Choose option **4** to see who marked attendance today with time and status.

---

### 5. View student attendance history

1. Choose option **5**
2. Enter the **Student ID**
3. See all past attendance dates and times for that student

---

### 6. Export attendance to CSV

1. Choose option **6**
2. Enter a date (`YYYY-MM-DD`) or press Enter for **today**
3. A CSV file is saved to:

```
data/exports/attendance_YYYY-MM-DD.csv
```

CSV columns: `date`, `time`, `student_id`, `name`, `email`, `status`

Open the file in Excel or Google Sheets for reports.

---

### 7. Retrain face model

Choose option **7** if you added new students or face recognition is not working well. This rebuilds the model from all saved face images.

---

## Where Data Is Stored

All data is saved automatically inside the `data/` folder:

```
data/
├── attendance.db       # SQLite database (students + attendance records)
├── faces/              # Face images for each student
│   └── [student_id]/   # e.g. data/faces/101/000.jpg, 001.jpg, ...
├── face_model.yml      # Trained face recognition model
├── labels.pkl          # Maps model labels to student IDs
└── exports/            # Exported CSV files
```

**Do not delete** `data/` unless you want to reset the entire system.

---

## Project Files

| File | Purpose |
|------|---------|
| `app.py` | Main menu and user interface |
| `database.py` | SQLite database operations |
| `face_manager.py` | Face capture, training, recognition, and on-box labels |
| `liveness.py` | Anti-spoofing: blink, motion, and phone/photo detection |
| `requirements.txt` | Python package dependencies |

---

## Troubleshooting

### Camera does not open
- Close other apps using the webcam (Zoom, Teams, etc.)
- Check Windows camera permissions: **Settings → Privacy → Camera**
- Try restarting the app

### `Could not open camera`
- No webcam connected, or camera is in use by another program

### `No trained model found`
- Register at least one student first (option 1)

### `Face not recognized` / poor accuracy
- Improve lighting
- Re-register the student (option 1) — slowly turn head left/right during capture
- Run **Retrain face model** (option 7) after re-registering

### Attendance blocked / "Photo detected" / "Phone screen detected"
- This is intentional — do not use a phone photo or printed picture
- Look directly at the webcam and **blink once**
- Move your head slightly left or right
- Avoid very dark rooms or strong blue screen light on your face

### `ModuleNotFoundError: No module named 'cv2'`
- Run: `python -m pip install -r requirements.txt`

### `AttributeError: module 'cv2' has no attribute 'face'`
- You installed `opencv-python` instead of `opencv-contrib-python`
- Fix:

```powershell
python -m pip uninstall opencv-python opencv-contrib-python -y
python -m pip install opencv-contrib-python
```

### `Student ID already exists`
- Use a different ID, or delete the old data from `data/` to start fresh

---

## Quick Start Checklist

- [ ] Python 3.9+ installed
- [ ] Project folder opened in terminal
- [ ] `pip install -r requirements.txt` completed
- [ ] Webcam working
- [ ] Run `python app.py`
- [ ] Register students (option 1)
- [ ] Mark attendance (option 2)
- [ ] Export CSV when needed (option 6)

---

## Example Workflow (First Time)

1. Install dependencies
2. Run `python app.py`
3. Register Student `101` — Name: `John Doe`
4. Register Student `102` — Name: `Jane Smith`
5. Use option **2** — both students show face to mark present
6. Use option **4** — verify today's list
7. Use option **6** — export CSV for records

---

## License

Free to use for learning and school/college projects.
#   a t t e n d a n c e - f a c e - r e c o  
 