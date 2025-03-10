# add_responses.py
import sqlite3
from database.db_connection import create_connection

def add_response(category, user_patterns, bot_responses):
    """Insert a new book-themed response into the database."""
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO book_responses (category, user_patterns, bot_responses)
                VALUES (?, ?, ?)
                """, 
                (category, user_patterns, bot_responses)
            )
            connection.commit()
            print(f"Added response for category: {category}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            connection.close()

if __name__ == "__main__":
    category = input("Category (e.g., 'genre'): ")
    user_patterns = input("User patterns (comma-separated): ")
    bot_responses = input("Bot responses (comma-separated): ")
    add_response(category, user_patterns, bot_responses)