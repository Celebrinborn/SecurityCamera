import sqlite3

def print_all_tables_and_content(db_file):
    # Create a connection to the database
    conn = sqlite3.connect(db_file)

    # Create a cursor
    cur = conn.cursor()

    # Get the list of all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()

    # Iterate over each table
    for table_name in tables:
        table_name = table_name[0]
        print(f"Table: {table_name}")

        # Get column information
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = cur.fetchall()

        # Print column names and types
        print("Columns:")
        for column in columns:
            print(f"Name: {column[1]}, Type: {column[2]}")

        # Print the contents of the table
        cur.execute(f"SELECT * from {table_name}")
        rows = cur.fetchall()

        print("Rows:")
        for row in rows:
            print(row)

        print("\n")

    # Close the connection
    conn.close()
    
import os
# Call the function
print_all_tables_and_content(os.path.join('data', "object_detection.sqlite3"))
