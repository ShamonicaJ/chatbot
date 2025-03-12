
"""
Book Recommendation Chatbot
Author: Sha'Monica
Date: 2025-03-09
Version: 2.0
"""
# =============================================
#           SAMPLE INTERACTIONS
# =============================================

# Input: "Recommend a book"
# Bot: 📚 Recommendation: The Silent Patient by Alex Michaelides 
#  ⭐ Rating: 4.7/5 📅 Published: 2019


# ==============================
#           IMPORTS
# ==============================
import numpy as np
import pandas as pd
import json
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import sqlite3
from database.db_connection import create_connection
from database.db_operations import store_user_query
from sklearn.model_selection import train_test_split


# TensorFlow configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.models import Sequential # type: ignore

# Colorama for colorful interactions
from colorama import Fore, Style, init
init(autoreset=True)  # Automatically reset colors after each print

# Initialize NLTK
nltk.download('punkt')
stemmer = LancasterStemmer()

# ==============================
#      DATA PREPROCESSING
# ==============================
# def load_intents_data():
#     """Load and process intents.json file with error handling"""
#     try:
#         with open("intents.json", "r", encoding="utf-8") as file:
#             return json.load(file)
#     except Exception as e:
#         print(f"{Fore.RED}Error loading intents.json: {e}{Style.RESET_ALL}")
#         exit(1)
def load_intents_data():
    """Load and process intents.json file with error handling"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Add this
        intent_path = os.path.join(current_dir, "intents.json")   # Add this
        # print(f"🕵️ Looking for intents.json at: {intent_path}")  # Debug line
        
        with open(intent_path, "r", encoding="utf-8") as file:  # Modified
            # print("✅ Successfully loaded intents.json")  # Debug line
            return json.load(file)
    except Exception as e:
        print(f"{Fore.RED}🚨 Critical Error loading intents.json: {e}{Style.RESET_ALL}")
        exit(1)

intents_data = load_intents_data()


def preprocess_intents(intents):
    """
    Process intents data to create training dataset
    Returns: (words, labels, docs_x, docs_y)
    """
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in intents["intents"]:
        current_tag = intent["tag"]
        if current_tag not in labels:
            labels.append(current_tag)
            
        for pattern in intent["patterns"]:
            tokenized_words = nltk.word_tokenize(pattern)
            stemmed_words = [stemmer.stem(w.lower()) for w in tokenized_words]
            words.extend(stemmed_words)
            docs_x.append(stemmed_words)
            docs_y.append(current_tag)
    
    punctuation = ["?", "!", ".", ","]
    words = sorted(list(set([w for w in words if w not in punctuation])))
    labels = sorted(labels)
    
    return words, labels, docs_x, docs_y

words, labels, docs_x, docs_y = preprocess_intents(intents_data)

# ==============================
#    TRAINING DATA PREPARATION
# ==============================
def create_training_data(words, labels, docs_x, docs_y):
    """
    Convert processed data into numerical training format
    Returns: (training, output) numpy arrays
    """
    training = []
    output = []
    out_empty = [0] * len(labels)

    for x, doc in enumerate(docs_x):
        bag = [1 if w in doc else 0 for w in words]
        output_row = out_empty.copy()
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)
        
    return np.array(training), np.array(output)

# Create and split dataset
training, output = create_training_data(words, labels, docs_x, docs_y)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    training, output, 
    test_size=0.2, 
    stratify=output,
    random_state=42
)

# ==============================
#    MODEL DEFINITION & TRAINING
# ==============================

def create_model(input_shape, output_shape):
    """Create and compile the neural network model with attention-inspired feature weighting"""
    model = Sequential([
        Input(shape=(input_shape,), name='input_layer'),
                # Add attention-inspired weighting
        Dense(input_shape, activation='sigmoid', name='attention_gate'),  # New
        keras.layers.Multiply(),  # Attention-inspired feature weighting via gating mechanism
        Dense(64, activation='relu', name='feature_learning'),  # New layer
        keras.layers.Dropout(0.2, name='regularization'),       # New dropout
        Dense(8, activation='relu', name='hidden_layer1'),
        Dense(8, activation='relu', name='hidden_layer2'),
        Dense(8, activation='relu', name='hidden_layer3'),
        Dense(output_shape, activation='softmax', name='output_layer')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_save_model():
    """Handle model training and saving"""
    print(f"\n{Fore.CYAN}Training new model...{Style.RESET_ALL}")
    model = create_model(len(training[0]), len(output[0]))
    model.summary()
     # Add these 2 lines here 👇
    print(f"\n{Fore.MAGENTA}Model Architecture (Attention-inspired Design):{Style.RESET_ALL}")
    model.summary()  # This shows layer structure
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 50 == 0:
                print(f"{Fore.YELLOW}Epoch {epoch+1} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}{Style.RESET_ALL}")
    
    model.fit(
        X_train, y_train,  # Use split data
        validation_data=(X_val, y_val),  # Add validation
        epochs=500,
        batch_size=256,
        verbose=0,
        callbacks=[ProgressCallback()]
    )
    
    model.save('model.h5')
    print(f"\n{Fore.GREEN}✅ Model trained and saved successfully!{Style.RESET_ALL}")
    return model

try:
    model = keras.models.load_model('model.h5')
    print(f"{Fore.GREEN}✅ Loaded pre-trained model{Style.RESET_ALL}")
except (IOError, OSError):
    model = train_and_save_model()

# ==============================
#      DATABASE OPERATIONS
# ==============================

def process_user_input(user_input):
    """Main processing pipeline for user input"""
        # print(f"\n🔍 Processing input: '{user_input}'")
    # print(f"📝 Raw input debug: '{user_input}' (length: {len(user_input)}, hex: {user_input.encode('utf-8').hex()})")  # NEW DEBUG LINE
    print(f"\n🔍 Processing input: '{user_input}'")
    
    try:
        store_user_query(
            db_name='new_books_db',
            user_id='anonymous',  # Replace with actual user ID system if available
            query_text=user_input
        )
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Failed to store query: {e}{Style.RESET_ALL}")

    # # ✅ Step 1: Check for greeting responses in intents.json
    # print("\n🔎 Checking greeting patterns:")  # NEW DEBUG LINE
    # for intent in intents_data["intents"]:
    #     if intent["tag"] == "greeting" and any(pattern.lower() in user_input.lower() for pattern in intent["patterns"]):
    #         print(f"  🔑 Greeting patterns: {intent['patterns']}")  # NEW DEBUG LINE
    #         return random.choice(intent['responses'])
        # ✅ Step 1: Check for greeting responses in intents.json
   
    user_input_clean = user_input.lower().strip()
    for intent in intents_data["intents"]:
        if intent["tag"] == "greeting":
           
            for pattern in intent["patterns"]:
                pattern_clean = pattern.lower().strip()
                print(f"    🧩 Checking pattern: '{pattern_clean}' vs input: '{user_input_clean}'")  # NEW DEBUG LINE
                if pattern_clean in user_input_clean:
               
                    return random.choice(intent['responses'])

    # ✅ Step 2: Handle recommendation intent via database
    if any(keyword in user_input.lower() for keyword in ["recommend", "suggest", "read", "what should i"]):
        return handle_recommendation_flow()

    # ✅ Step 3: Search for additional custom responses in database
    response = get_book_recommendation(
        'new_books_db',
        "SELECT bot_responses FROM book_responses WHERE user_patterns LIKE ?",
        (f"%{user_input}%",)
    )
    if response:
        return random.choice(response[0].split(','))

    # ✅ Step 4: NLP Model prediction for fallback intents (e.g., mood, sass)
    bow = bag_of_words(user_input, words)
    prediction = model.predict(bow, verbose=0)[0]
    tag = labels[np.argmax(prediction)]

    if prediction.max() > 0.5:
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                return f"{Fore.BLUE}Bot:{Style.RESET_ALL} {random.choice(intent['responses'])}   (Category: {tag})"

    # ✅ Step 5: Fallback response
    return f"🤖 I'm not sure I understand. Try asking for a book recommendation!"


# ==============================
#     RECOMMENDATION HANDLING 
# ==============================

def get_book_recommendation(db_name, query, params, retries=3):
    """
    Retrieve book recommendations from the database with retry logic.
    Args:
        db_name: Database filename.
        query: SQL query to execute.
        params: Query parameters.
        retries: Number of retry attempts remaining.
    Returns:
        Query result or None.
    """
    conn = create_connection(db_name)
    if not conn:
        return None

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchone()  # Fetching one result

        if result:
            # print(f"DEBUG - Database query result: {result}")  # Debugging the result
            # # Unpacking the result into variables
            title, author, rating, publish_year = result
            return f"📚 Recommendation: {title} by {author} ⭐ Rating: {rating}/5 | 📅 Published: {publish_year}"
        
        return None

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None

    finally:
        try:
            conn.close()
        except Exception as e:
            print(f"Warning: Error closing connection - {e}")

def get_new_books_recommendation(search_term, prefer_new=True):
    """Get recommendations without rating filter"""
    query = """
        SELECT title, authors, rating, publish_year 
        FROM books
        WHERE (title LIKE ? OR authors LIKE ?)
        ORDER BY 
            CASE WHEN ? THEN publish_year ELSE rating END DESC,
            RANDOM()
        LIMIT 1
    """
    params = (
        f"%{search_term}%", 
        f"%{search_term}%",
        prefer_new
    )
    
    result = get_book_recommendation('new_books_db', query, params)
    
    
    if result:
        title, authors, rating, year = result
        return (f"📚 Recommendation: {title} by {authors}\n"
                f"⭐ Rating: {rating}/5 | 📅 Published: {year}")
    return None

def handle_recommendation_flow():
    """Simplified recommendation flow for random books"""
    print(f"\n{Fore.MAGENTA}Let me suggest a great read! 📖{Style.RESET_ALL}")
    
    # Get a random recommendation
    recommendation = get_random_recommendation()
    
    return recommendation or f"{Fore.RED}No recommendations available at the moment.{Style.RESET_ALL}"

# def get_random_recommendation():
#     """Get recommendation with debug logging"""
#     query = """
#     SELECT title, authors, rating, publish_year 
#     FROM books
#     WHERE rating > 0
#     ORDER BY RANDOM()
#     LIMIT 1
# """
#     try:
#         result = get_book_recommendation('new_books_db', query, ())  # 👈 Add try here
#     except Exception as e:
#         print(f"{Fore.RED}Recommendation error: {e}{Style.RESET_ALL}")
#         return None
    
#     if result:
#         return format_recommendation(result)
                
#     return None
def get_random_recommendation():
    """Get recommendation with debug logging"""
    query = """
    SELECT title, authors, rating, publish_year 
    FROM books
    WHERE rating > 0
    ORDER BY RANDOM()
    LIMIT 1
    """
    try:
        result = get_book_recommendation('new_books_db', query, ())
        if result:
            return result  # Returns full book details (title, author, rating, year)
    except Exception as e:
        print(f"{Fore.RED}Recommendation error: {e}{Style.RESET_ALL}")
        return None

    return None

# ==============================
#     FORMATTING UTILITIES
# ==============================
def clean_book_title(title, max_length=60):
    """Shorten long book titles for clean display"""
    if len(title) <= max_length:
        return title
    return title[:max_length - 3] + '...'  # Truncate with ellipsis

def format_recommendation(result):
    title, authors, rating, _ = result  # _ ignores unused year
    clean_title = clean_book_title(title)
    formatted_rating = f"{float(rating):.1f}"  # Force 1 decimal place
    
    return (
        f"🌟 Recommended Read:\n"
        f'   "{clean_title}"\n'
        f"   ✍️ {authors} | 💫 Rating: {formatted_rating}"
    )

def format_rating(rating):
    """Convert 4.07 to ⭐⭐⭐⭐"""
    full_stars = int(rating)
    half_star = 1 if rating - full_stars >= 0.5 else 0
    return '⭐' * full_stars + '½' * half_star


# ==============================
#      CHATBOT FUNCTIONALITY
# ==============================
def bag_of_words(sentence, vocabulary):
    """    Convert a sentence into a bag-of-words (BoW) representation.
    Each word in the vocabulary is checked against words in the sentence,
    producing a binary vector indicating word presence."""
    sentence_words = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentence)]
    return np.array([[1 if w in sentence_words else 0 for w in vocabulary]])

# ==============================
#      DATA LOADING FUNCTION
# ==============================
# Add this NEW SECTION after your database initialization code
def load_books_data():
    """Load processed books into database"""
    from database.db_connection import create_connection  # Import here
    
    conn = create_connection('new_books_db')
    if not conn: return
    
    try:
        df = pd.read_parquet('processed_books.parquet')
        df.to_sql('books', conn, if_exists='replace', index=False)
        conn.commit()
        print(f"{Fore.GREEN}✅ Loaded {len(df)} books into database{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Data import failed: {e}{Style.RESET_ALL}")
    finally:
        conn.close()
        
def load_book_responses():
    """Load chatbot response patterns into database"""
    from database.db_connection import create_connection
    import pandas as pd
    
    conn = create_connection('new_books_db')
    if not conn: return
    
    try:
        data = {
            'user_patterns': [
                'recommend a book',
                'suggest something',
                'what should I read?'
            ],
            'bot_responses': [
                '📚 Try "{title}" by {author} ({rating}⭐)',
                '🔥 Hot pick: "{title}" published in {year}',
                '🌟 Fan favorite: "{title}" with {rating}/5 rating'
            ]
        }
        
        pd.DataFrame(data).to_sql('book_responses', conn, if_exists='replace', index=False)
        conn.commit()
        print(f"{Fore.GREEN}✅ Loaded {len(data['user_patterns'])} response patterns{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Response import failed: {e}{Style.RESET_ALL}")
    finally:
        conn.close()

# Add to __main__ block after load_books_data()
load_book_responses()

# ==============================
#        MAIN INTERFACE
# ==============================
def chat_interface():
    """Main chatbot interaction loop"""
    print(f"\n{Fore.MAGENTA}Hello, Chatbot!{Style.RESET_ALL}")  # Extra colorful line
    print(f"{Fore.GREEN}\n📚 Welcome to BookBuddy - Your Personal Reading Assistant! 🎉{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Type '/quit' to exit at any time\n{Style.RESET_ALL}")
    
    while True:
        try:
            user_input = input(f"{Fore.CYAN}You: {Style.RESET_ALL}").strip()
            if not user_input:
                continue
                
            if user_input.lower() == "/quit":
                print(f"\n{Fore.YELLOW}📖 Happy reading! Goodbye! 👋{Style.RESET_ALL}")
                break
                
            response = process_user_input(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}📖 Session interrupted. Goodbye! 👋{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}⚠️  Error: {str(e)}{Style.RESET_ALL}")
            print("Please try again or rephrase your question.")

# ==============================
#          INITIALIZATION
# ==============================
def test_storage():
    """Temporary storage test function"""
    from database.db_operations import store_user_query
    store_user_query('new_books_db', 'test_user', 'TEST QUERY 4321')

if __name__ == "__main__":
    # Initialize ALL databases
    dbs = [
        ('new_books_db', '''
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                rating REAL CHECK (rating BETWEEN 0 AND 5),
                publish_year INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS book_responses (
                id INTEGER PRIMARY KEY,
                user_patterns TEXT NOT NULL,
                bot_responses TEXT NOT NULL
            );
        ''')
    ]

    # Create tables
    for db_name, schema in dbs:
        conn = create_connection(db_name)
        if conn:
            try:
                conn.executescript(schema)
                conn.commit()
                print(f"{Fore.GREEN}✅ Initialized {db_name} database{Style.RESET_ALL}")
            except sqlite3.Error as e:
                print(f"{Fore.RED}❌ Error initializing {db_name}: {e}{Style.RESET_ALL}")
            finally:
                conn.close()

    # Verify tables
    test_conn = create_connection('new_books_db')
    if test_conn:
        try:
            cursor = test_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"{Fore.CYAN}\nDatabase Tables:{Style.RESET_ALL}")
            for table in tables:
                print(f"- {table[0]}")
        except sqlite3.Error as e:
            print(f"{Fore.RED}❌ Verification failed: {e}{Style.RESET_ALL}")
        finally:
            test_conn.close()
    
    # Load book data (ensure this function exists)
    load_books_data()
#     test_conn = create_connection('new_books_db')
# if test_conn:
#     try:
#         test_conn.row_factory = sqlite3.Row
#         cursor = test_conn.cursor()
        
#         # Check books table
#         cursor.execute("SELECT COUNT(*) FROM books")
#         book_count = cursor.fetchone()[0]
#         print(f"{Fore.GREEN}✅ Books in database: {book_count}{Style.RESET_ALL}")
        
#         # Check sample ratings
#         cursor.execute("SELECT rating FROM books LIMIT 5")
#         sample_ratings = [row['rating'] for row in cursor.fetchall()]
#         print(f"{Fore.GREEN}📊 Sample ratings: {sample_ratings}{Style.RESET_ALL}")
        
#     finally:
#         test_conn.close()
    
    # Start chat interface
    chat_interface()