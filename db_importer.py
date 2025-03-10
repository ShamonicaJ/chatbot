
import pandas as pd
import sqlite3
import os

def parquet_to_sqlite(parquet_path, db_name, table_name):
    """Convert a Parquet file to SQLite database and insert into the specified table"""
    if not os.path.exists(parquet_path):
        print(f"⚠️ Skipping {table_name}: File '{parquet_path}' not found.")
        return  # Exit function gracefully

    conn = None  # Initialize before try block
    try:
        # Read Parquet file
        df = pd.read_parquet(parquet_path)
        
        # Connect to SQLite
        conn = sqlite3.connect(db_name)
        
        # Write to SQLite
        df.to_sql(name=table_name, con=conn, if_exists='replace', index=False)

        # Commit changes
        conn.commit()
        
        print(f"✅ Successfully imported {len(df)} records to {db_name}.{table_name}")

    except Exception as e:
        print(f"❌ Error importing {table_name}: {str(e)}")

    finally:
        if conn:  # Ensure conn is not None before closing
            conn.close()

if __name__ == "__main__":
    # Database name
    db_name = "new_books_db_sqlite.db"

    # Paths to your Parquet files
    books_parquet = "processed_books.parquet"          
    ratings_parquet = "processed_ratings.parquet"  # Now it should exist!
    user_queries_parquet = "processed_user_queries.parquet"  

    # Import into respective tables
    parquet_to_sqlite(books_parquet, db_name, "books")
    parquet_to_sqlite(ratings_parquet, db_name, "ratings")  # Should now work!
    parquet_to_sqlite(user_queries_parquet, db_name, "user_queries")  # Skipped if missing

