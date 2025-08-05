# setup_database.py
import sqlite3
import os

DB_PATH = "data/violations.db"
VIOLATIONS_DIR = "data/violations"

# Ensure the directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# Connect to the database (this will create the file)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Drop the old table if it exists to ensure a clean start
cursor.execute("DROP TABLE IF EXISTS violations")

# Create the table with the final, correct schema
cursor.execute('''
    CREATE TABLE violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        frame_index INTEGER NOT NULL,
        image_path TEXT,
        data_path TEXT
    )
''')

conn.commit()
conn.close()

print(f"✅ Database '{DB_PATH}' has been successfully created with the correct 'violations' table.")