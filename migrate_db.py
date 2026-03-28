import sqlite3
import os

db_path = os.path.join('instance', 'attendance.db')

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if 'total_hours' column exists in 'subject' table
    cursor.execute("PRAGMA table_info(subject)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'total_hours' not in columns:
        print("Adding 'total_hours' column to 'subject' table...")
        cursor.execute("ALTER TABLE subject ADD COLUMN total_hours INTEGER NOT NULL DEFAULT 20")
        conn.commit()
        print("Migration successful.")
    else:
        print("'total_hours' column already exists.")
    
    conn.close()
else:
    print(f"Database not found at {db_path}. Skipping migration.")
