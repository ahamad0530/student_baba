import sqlite3
import os

db_path = 'predictions.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name, seq FROM sqlite_sequence WHERE name='predictions'")
print(f"Current sequence: {cursor.fetchone()}")

cursor.execute("DELETE FROM predictions")
cursor.execute("UPDATE sqlite_sequence SET seq = -1 WHERE name = 'predictions'")
conn.commit()

cursor.execute("INSERT INTO predictions (roll_number) VALUES ('DEBUG')")
conn.commit()

cursor.execute("SELECT id FROM predictions WHERE roll_number='DEBUG'")
print(f"Inserted ID: {cursor.fetchone()[0]}")

conn.close()
