from flask import Flask, render_template, redirect, url_for, flash, request, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import io
from datetime import datetime
import pandas as pd
import numpy as np
import random

from models import db, User, FaceEncoding, Attendance, Subject, SchoolConfig, Timetable

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_for_this_mini_project'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_defaults():
    if not User.query.filter_by(role='teacher').first():
        hashed_pw = bcrypt.generate_password_hash('admin').decode('utf-8')
        db.session.add(User(username='admin', name='Admin Teacher', password_hash=hashed_pw, role='teacher'))

    if not User.query.filter_by(username='student').first():
        hashed_pw = bcrypt.generate_password_hash('student').decode('utf-8')
        db.session.add(User(username='student', name='Test Student', password_hash=hashed_pw, role='student'))

    db.session.commit()

with app.app_context():
    db.create_all()
    init_defaults()

@app.route('/', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('teacher_dashboard' if current_user.role=='teacher' else 'student_dashboard'))

    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and bcrypt.check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user)
            return redirect(url_for('teacher_dashboard' if user.role=='teacher' else 'student_dashboard'))
        flash('Login failed', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard/teacher')
@login_required
def teacher_dashboard():
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))

    students = User.query.filter_by(role='student').all()
    subjects = Subject.query.all()

    return render_template('teacher_dashboard.html',
                           students=students,
                           subjects=subjects)

@app.route('/dashboard/student')
@login_required
def student_dashboard():
    if current_user.role != 'student':
        return redirect(url_for('teacher_dashboard'))

    attendances = Attendance.query.filter_by(user_id=current_user.id).all()
    return render_template('student_dashboard.html',
                           total_attended=len(attendances))

@app.route('/register_student', methods=['GET', 'POST'])
@login_required
def register_student():
    if current_user.role != 'teacher':
        return redirect(url_for('student_dashboard'))

    if request.method == 'POST':
        hashed_pw = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        new_student = User(
            username=request.form.get('username'),
            name=request.form.get('name'),
            password_hash=hashed_pw,
            role='student'
        )
        db.session.add(new_student)
        db.session.commit()
        flash('Student registered', 'success')
        return redirect(url_for('teacher_dashboard'))

    return render_template('register_student.html')

@app.route('/export_attendance')
@login_required
def export_attendance():
    attendances = Attendance.query.all()
    data = []

    for att in attendances:
        data.append({
            'Date': att.date,
            'Student': att.user.name,
            'Subject': att.subject.name
        })

    df = pd.DataFrame(data)
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(output, download_name="attendance.xlsx", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
