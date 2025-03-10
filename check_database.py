import sqlite3
from database.db_connection import create_connection  # Adjust path as needed

def verify_database_structure():
    """Diagnostic tool to check database schema"""
    conn = create_connection('new_books_db_sqlite')
    if not conn:
        print("❌ Database connection failed")
        return

    cursor = conn.cursor()
    
    # Check existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("\n=== Tables in Database ===")
    print([table[0] for table in tables])

    # Verify books table structure
    try:
        cursor.execute("PRAGMA table_info(books)")
        columns = cursor.fetchall()
        print("\n=== Books Table Columns ===")
        for col in columns:
            print(f"Column {col[1]} (Type: {col[2]}, Nullable: {not col[3]})")
    except sqlite3.OperationalError as e:
        print(f"\n❌ Error: {str(e)}")

    conn.close()

if __name__ == "__main__":
    verify_database_structure()