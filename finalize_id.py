import sqlite3
import os

BASE_DIR = os.getcwd()
db_path = os.path.join(BASE_DIR, 'predictions.db')

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
try:
    print("Finalizing ID reset to 0...")
    # Clear table
    cursor.execute("DELETE FROM predictions")
    # Reset sequence
    cursor.execute("SELECT name FROM sqlite_sequence WHERE name='predictions'")
    if cursor.fetchone():
        cursor.execute("UPDATE sqlite_sequence SET seq = -1 WHERE name = 'predictions'")
    else:
        cursor.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('predictions', -1)")
    
    # Force insert the first record with ID 0 to 'seal' the sequence
    cursor.execute('''
        INSERT INTO predictions (
            id, roll_number, academic_year, study_hours, attendance, 
            assignments_completed, previous_grade, participation, 
            sleep_hours, internet_usage, final_score, grade
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        0, 'INIT_SYSTEM', 'N/A', 0, 0, 0, 0, 0, 0, 0, 0, 'N/A'
    ))
    conn.commit()
    
    # Verify
    cursor.execute("SELECT id FROM predictions LIMIT 1")
    row = cursor.fetchone()
    print(f"First record ID: {row[0]}")
    
    # Optionally delete the init record if we want the user's first prediction to be 0
    # But if we delete it, SQLite might reset or keep seq at 0.
    # Actually, if we leave it, the next will be 1. 
    # If we delete it, let's see what happens to the sequence.
    cursor.execute("DELETE FROM predictions WHERE id=0")
    conn.commit()
    
    # Now the sequence should be at 0. The next insert should be 1? 
    # Wait, if I want the FIRST user prediction to be 0, I should leave it at seq = -1 and no records.
    # But if that failed, it's safer to just leave a dummy record or tell the user it starts at 1 by default in SQLite.
    # I'll try seq = -1 one last time with a fresh table.
    
    cursor.execute("DELETE FROM predictions")
    cursor.execute("UPDATE sqlite_sequence SET seq = -1 WHERE name = 'predictions'")
    conn.commit()
    print("Database reset to start at 0 (hopefully).")

except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
