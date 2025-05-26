import sqlite3
from sqlite3 import Connection
from typing import List, Tuple

DB_NAME = 'emotion_logs.db'

def get_connection() -> Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotion_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        emotion TEXT NOT NULL,
        content_emotion TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def insert_emotion_log(user_id: str, emotion: str, content_emotion: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO emotion_logs (user_id, emotion, content_emotion) 
        VALUES (?, ?, ?)
    ''', (user_id, emotion, content_emotion))
    conn.commit()
    conn.close()

def fetch_emotions_by_timeslot(user_id: str, start_hour: int, end_hour: int) -> List[Tuple]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT emotion, content_emotion, timestamp 
        FROM emotion_logs 
        WHERE user_id = ?
        AND strftime('%H', timestamp) >= ? 
        AND strftime('%H', timestamp) < ?
    ''', (user_id, f"{start_hour:02d}", f"{end_hour:02d}"))
    rows = cursor.fetchall()
    conn.close()
    return [(row['emotion'], row['content_emotion']) for row in rows]
