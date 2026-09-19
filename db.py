import sqlite3
import hashlib

DB_NAME = "medical_ai.db"

def connect():
    return sqlite3.connect(DB_NAME)

def create_tables():
    conn = connect()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        glucose REAL,
        bmi REAL,
        age REAL,
        risk REAL
    )
    """)

    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    conn = connect()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    result = c.fetchone()
    conn.close()
    return result is not None

def add_history(username, glucose, bmi, age, risk):
    conn = connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO history (username, glucose, bmi, age, risk)
        VALUES (?, ?, ?, ?, ?)
    """, (username, glucose, bmi, age, risk))
    conn.commit()
    conn.close()

def get_history(username):
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT glucose, bmi, age, risk FROM history WHERE username=?",
              (username,))
    data = c.fetchall()
    conn.close()
    return data
