import sqlite3
import os

db_path = os.path.join('instance', 'attendance.db')

def reset_database():
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Nothing to reset.")
        return

    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Delete face encodings
        print("Deleting all face encodings...")
        cursor.execute("DELETE FROM face_encoding")
        
        # Delete attendance records
        print("Deleting all attendance records...")
        cursor.execute("DELETE FROM attendance")
        
        # Delete student users (keep teachers)
        print("Deleting student users...")
        cursor.execute("DELETE FROM user WHERE role = 'student'")
        
        # Commit changes
        conn.commit()
        print("Database cleanup successful.")
        
        # Vacuum to reclaim space
        print("Optimizing database size...")
        cursor.execute("VACUUM")
        conn.commit()
        print("Database optimized.")

    except Exception as e:
        print(f"An error occurred during reset: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    reset_database()
