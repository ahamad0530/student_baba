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
        # 1. Clear existing data
        print("Clearing 'predictions' table...")
        cursor.execute("DELETE FROM predictions")
        
        # 2. Reset the sequence
        # SQLite's AUTOINCREMENT starts at seq + 1. 
        # To start at 0, we set seq to -1.
        print("Resetting auto-increment sequence to -1 (so next is 0)...")
        
        # Check if predictions is in sqlite_sequence
        cursor.execute("SELECT name FROM sqlite_sequence WHERE name='predictions'")
        if cursor.fetchone():
            cursor.execute("UPDATE sqlite_sequence SET seq = -1 WHERE name = 'predictions'")
        else:
            cursor.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('predictions', -1)")
            
        conn.commit()
        print("Database ID count reset successfully to start at 0.")
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()
