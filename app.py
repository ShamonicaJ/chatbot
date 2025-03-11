# from flask import Flask, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# @app.route('/chat', methods=['POST'])
# def chat():
#     return jsonify({"response": "📚 Recommendation: The Silent Patient by Alex Michaelides"})

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from database.db_connection import create_connection  # Ensure this import path is correct

app = Flask(__name__)
CORS(app)

# ==============================
#     DATABASE QUERY FUNCTION
# ==============================
def get_random_book_recommendation():
    """Retrieve a random book recommendation from the database."""
    conn = create_connection('new_books_db')
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        query = """
        SELECT title, authors, rating, publish_year
        FROM books
        WHERE rating IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 1
        """
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            title, authors, rating, year = result
            return (f"📚 Recommendation: {title} by {authors}\n"
                    f"⭐ Rating: {rating}/5 | 📅 Published: {year}")
        else:
            return "⚠️ No recommendations available at the moment."

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return "⚠️ Error fetching recommendations."

    finally:
        conn.close()

# ==============================
#         CHAT ROUTE
# ==============================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '').lower()

    # Book recommendation logic
    if any(keyword in user_input for keyword in ["recommend", "suggest", "book", "read"]):
        return jsonify({"response": get_random_book_recommendation()})

    # Default response for unknown queries
    return jsonify({"response": "🤖 I'm not sure I understand. Try asking for a book recommendation!"})

if __name__ == '__main__':
    app.run(debug=True)
