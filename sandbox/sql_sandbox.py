import os
import pyodbc
from dotenv import load_dotenv

import os
from dotenv import load_dotenv

load_dotenv('secrets/.env')
password = os.environ.get('SA_PASSWORD')
driver= '{ODBC Driver 17 for SQL Server}'
username = 'sa'

server = os.environ.get('Database_Server', 'localhost')
database = 'Home_Automation'

# Not specifying a database in the connection string
connection_string = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

print(connection_string)

try:
    # Establishing a connection with SQL Server
    with pyodbc.connect(connection_string) as conn:
        with conn.cursor() as cursor:
            # Executing a query not specific to any database
            cursor.execute("SELECT @@version;")
            row = cursor.fetchone()
            while row:
                print(row[0])  # Print the version of SQL Server
                row = cursor.fetchone()

except pyodbc.Error as ex:
    sqlstate = ex.args[1]
    print(sqlstate)  # Printing the error message in case of an exception
