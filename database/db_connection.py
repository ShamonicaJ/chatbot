
import os
import sqlite3
from colorama import Fore, Style  # Add this import

def create_connection(db_name):
    """Create a database connection to SQLite database"""
    db_path = f"database/{db_name}.sqlite"  # Enforce .sqlite extension
    os.makedirs('database', exist_ok=True)
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except sqlite3.Error as e:
        print(f"❗ Database connection error: {e}")
        return None