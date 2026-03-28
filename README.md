# SmartAttend - Facial Recognition Attendance System

A full-stack web application designed for a college mini-project, featuring a redesigned dark-themed UI and Subject-based tracking.

## 🎨 Theme & UI
- **Design:** Modern, minimalist dark theme (Purple/Blue).
- **Fonts:** Poppins (Google Fonts).
- **Navigation:** Sidebar-based app layout.

## ✨ Core Features
- **Teacher Login:** Default admin (`admin` / `admin`).
- **Student Login:** Split landing page for role selection.
- **Subject Tracking:** Categorize attendance by specific classes (Engineering Graphics, etc.).
- **Webcam Interface:** Real-time facial scanning via OpenCV MJPEG streams.
- **Excel Export:** Subject-wise daily/monthly reports.

## 📂 Project Structure
```
facerecognition/
├── app.py                # Main Flask application with routing and endpoints
├── models.py             # SQLAlchemy database models (User, FaceEncoding, Attendance)
├── camera_utils.py       # OpenCV and face_recognition helper logic
├── requirements.txt      # Python dependencies list
├── README.md             # This setup guide
├── instance/             # Auto-generated SQLite database (attendance.db)
├── static/
│   └── css/
│       └── style.css     # Bespoke theme styling
└── templates/
    ├── base.html              # HTML Layout Shell
    ├── login.html             # Login Interface
    ├── teacher_dashboard.html # Admin features and table
    ├── student_dashboard.html # Student view
    ├── register_student.html  # Face capturing UI
    └── take_attendance.html   # Live recognition UI
```

## ⚙️ Setup Instructions

### 1. Pre-requisites
- **Python 3.8 to 3.11** installed. Note: Python 3.12+ might require compiling `dlib` manually on Windows. Python 3.10 is recommended for ease of `dlib` installation.
- **CMake**: `face_recognition` library requires CMake to be installed and added to PATH for building `dlib`. (Download from cmake.org if compiling fails).

### 2. Install Dependencies
Open your terminal/command prompt, navigate to the `facerecognition` folder and run:
```bash
pip install -r requirements.txt
```
*(If face-recognition fails to install on Windows, ensure CMake is installed, or `pip install cmake` before running the requirements installation).*

### 3. Run the Application
Start the Flask development server:
```bash
python app.py
```

### 4. Accessing the System
- Open your browserhttp://localhost:5000 and go to: ``
- The system will auto-initialize the database `instance/attendance.db` on first run.

## 🔑 Default Credentials
- **Role**: Teacher
- **Username**: `admin`
- **Password**: `admin`

- **Role**: Student
- **Username**: `student`
- **Password**: `student`

*(Students can also use their own Student ID and the password assigned during registration).*

## 🧠 System Logic Details
1. **Face Encoding:** When a student is registered, frames are captured via the `VideoCamera` class (`camera_utils.py`). `face_recognition.face_encodings` generates 128-dimensional encodings.
2. **Matching:** For attendance, the live camera frames compare live encodings against all stored ones in the database using `face_recognition.compare_faces` with a strict tolerance threshold (default `0.5`).
3. **Database:** SQLite handles relations. A unique constraint on `(user_id, date)` blocks duplicate attendance logs.

## 🎓 College Project Tips
- The source code in `app.py`, `models.py`, and `camera_utils.py` contains inline comments to help explain the internal workflow during project presentations.
- You can demonstrate the duplicate prevention clearly by standing in front of the camera twice on the same day; the dashboard will only log you once.
- To demonstrate Excel export, open the downloaded `.xlsx` file using Microsoft Excel to show data parsing correctness using Pandas.
