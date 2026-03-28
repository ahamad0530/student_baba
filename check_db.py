import sqlite3
import os

BASE_DIR = os.getcwd()
db_path = os.path.join(BASE_DIR, 'predictions.db')

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM predictions")
        rows = cursor.fetchall()
        print(f"Total rows: {len(rows)}")
        for row in rows[:5]:
            print(row)
        
        cursor.execute("SELECT name, seq FROM sqlite_sequence WHERE name='predictions'")
        seq = cursor.fetchone()
        print(f"Sequence: {seq}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
