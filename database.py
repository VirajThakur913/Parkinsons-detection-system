import sqlite3
import hashlib
import os
from datetime import datetime

DB_FILE = 'neurosense.db'

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_db():
    print("Initializing database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users table (Admin / Patient accounts)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'patient')),
            patient_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 3. Tests table (Screening results)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tests (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            date TEXT NOT NULL,
            voice_score REAL NOT NULL,
            motor_score REAL NOT NULL,
            final_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            original_drawing_url TEXT,
            thinned_drawing_url TEXT,
            pdf_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    ''')
    
    # 4. Audit Logs table (System Administration tracking)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    
    # Seed default Users
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        print("Seeding default accounts...")
        # Admin account (admin/admin123)
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ('admin', hash_password('admin123'), 'admin')
        )
        # Patient account (patient01/patient123)
        cursor.execute(
            "INSERT INTO users (username, password, role, patient_id) VALUES (?, ?, ?, ?)",
            ('patient01', hash_password('patient123'), 'patient', 'H001')
        )
        conn.commit()
        
    # Seed default Patients
    cursor.execute("SELECT COUNT(*) FROM patients")
    if cursor.fetchone()[0] == 0:
        print("Seeding clinical patient directory...")
        patients_seed = [
            ('H001', 'Sarah Jenkins', 54, 'Female'),
            ('H002', 'Arthur Pendelton', 61, 'Male'),
            ('H003', 'Claire Redfield', 48, 'Female'),
            ('P001', 'Thomas Miller', 67, 'Male'),
            ('P002', 'Eleanor Vance', 72, 'Female'),
            ('P003', 'Harvey Dent', 63, 'Male')
        ]
        cursor.executemany(
            "INSERT INTO patients (id, name, age, gender) VALUES (?, ?, ?, ?)",
            patients_seed
        )
        conn.commit()
        
    conn.close()
    print("Database initialization complete.")

# User auth handlers
def verify_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    h_pass = hash_password(password)
    cursor.execute(
        "SELECT id, username, role, patient_id FROM users WHERE username = ? AND password = ?",
        (username, h_pass)
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return dict(user)
    return None

def register_user(username, password, role, patient_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    h_pass = hash_password(password)
    try:
        cursor.execute(
            "INSERT INTO users (username, password, role, patient_id) VALUES (?, ?, ?, ?)",
            (username, h_pass, role, patient_id)
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

# Patient handlers
def add_patient(patient_id, name, age, gender):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO patients (id, name, age, gender) VALUES (?, ?, ?, ?)",
            (patient_id, name, age, gender)
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def get_all_patients():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age, gender FROM patients ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_patient_by_id(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age, gender FROM patients WHERE id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

# Test screening log handlers
def log_screening_test(test_id, patient_id, voice_score, motor_score, final_score, risk_level, original_url=None, thinned_url=None, pdf_url=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO tests (id, patient_id, date, voice_score, motor_score, final_score, risk_level, original_drawing_url, thinned_drawing_url, pdf_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (test_id, patient_id, date_str, voice_score, motor_score, final_score, risk_level, original_url, thinned_url, pdf_url))
    conn.commit()
    conn.close()

def get_patient_test_history(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, date, voice_score, motor_score, final_score, risk_level, original_drawing_url, thinned_drawing_url, pdf_url
        FROM tests WHERE patient_id = ? ORDER BY date DESC
    ''', (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_recent_tests(limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT t.id, t.patient_id, p.name as patient_name, t.date, t.final_score, t.risk_level, t.pdf_url
        FROM tests t
        JOIN patients p ON t.patient_id = p.id
        ORDER BY t.date DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Audit Logs
def log_audit_action(username, action):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO audit_logs (username, action) VALUES (?, ?)",
        (username, action)
    )
    conn.commit()
    conn.close()

def get_audit_logs(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, action, timestamp FROM audit_logs ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == '__main__':
    init_db()
