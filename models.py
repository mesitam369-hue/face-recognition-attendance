from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False) # e.g., student ID or teacher's username
    name = db.Column(db.String(150), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), nullable=False) # 'teacher' or 'student'
    
    # Relationships
    encodings = db.relationship('FaceEncoding', backref='user', lazy=True)
    attendances = db.relationship('Attendance', backref='user', lazy=True)

class FaceEncoding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    encoding_data = db.Column(db.Text, nullable=False) # JSON serialized numpy array

    def get_encoding(self):
        return json.loads(self.encoding_data)
        
    def set_encoding(self, encoding_list):
        self.encoding_data = json.dumps(encoding_list)

class Subject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    icon = db.Column(db.String(50), nullable=True) # E.g., font-awesome class or custom identifier
    total_hours = db.Column(db.Integer, nullable=False, default=20)
    
    attendances = db.relationship('Attendance', backref='subject', lazy=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=lambda: datetime.utcnow().date())
    time = db.Column(db.Time, nullable=False, default=lambda: datetime.utcnow().time())
    status = db.Column(db.String(50), nullable=False, default='Present')
    
    # Ensure a user is only marked present once per day PER SUBJECT
    __table_args__ = (db.UniqueConstraint('user_id', 'subject_id', 'date', name='_user_subject_date_uc'),)

class SchoolConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.String(255), nullable=False)

class Timetable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    day_of_week = db.Column(db.String(20), nullable=False) # e.g., 'Monday'
    period_number = db.Column(db.Integer, nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=True) # None means free period
    
    subject = db.relationship('Subject', backref='timetable_periods', lazy=True)
    
    __table_args__ = (db.UniqueConstraint('day_of_week', 'period_number', name='_day_period_uc'),)
