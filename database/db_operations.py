import sqlite3
import datetime
from database.db_connection import create_connection  # Ensure correct import path
from colorama import Fore, Style 


def store_user_query(db_name, user_id, query_text):  # 👈 Accept 3 parameters
    """Guaranteed storage with debug output"""
    from database.db_connection import create_connection  # Adjusted import path
    
    print(f"🔒 Attempting to store in {db_name}.sqlite")  # Use parameter
    
    conn = None  # Initialize connection
    try:
        conn = create_connection(db_name)
        if not conn:
            print(f"{Fore.RED}❌ Connection failed{Style.RESET_ALL}")
            return False
            
        sql = '''INSERT INTO user_queries(user_id, query_text)
                 VALUES(?, ?)'''  # Use parameters for both values
        
        conn.execute(sql, (user_id, query_text))  # Pass both values
        conn.commit()
        print(f"{Fore.GREEN}✅ Successfully stored query{Style.RESET_ALL}")
        return True
        
    except sqlite3.Error as e:
        print(f"{Fore.RED}❗ SQL Error: {e}{Style.RESET_ALL}")
        return False
    finally:
        if conn:
            conn.close()