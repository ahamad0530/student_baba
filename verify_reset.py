import sqlite3
import os

BASE_DIR = os.getcwd()
db_path = os.path.join(BASE_DIR, 'predictions.db')

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
try:
    # 1. Insert a test record
    print("Inserting test record...")
    cursor.execute('''
        INSERT INTO predictions (
            roll_number, academic_year, study_hours, attendance, 
            assignments_completed, previous_grade, participation, 
            sleep_hours, internet_usage, final_score, grade
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        'TEST000', '2025-26', 10.0, 95.0, 10.0, 90.0, 9.0, 8.0, 1.0, 95.0, 'A+'
    ))
    conn.commit()
    
    # 2. Check the ID of the inserted record
    cursor.execute("SELECT id, roll_number FROM predictions WHERE roll_number='TEST000'")
    row = cursor.fetchone()
    if row:
        print(f"Inserted record ID: {row[0]} (Roll: {row[1]})")
        if row[0] == 0:
            print("SUCCESS: ID count started at 0!")
        else:
            print(f"FAILURE: ID started at {row[0]}")
    else:
        print("Error: Could not find the test record.")
        
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
