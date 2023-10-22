import os
import pyodbc
from dotenv import load_dotenv

import os
from dotenv import load_dotenv

load_dotenv('secrets/.env')
password = os.environ.get('SA_PASSWORD')
driver= '{ODBC Driver 17 for SQL Server}'
username = 'sa'

server = 'localhost'# this needs to change for docker
database = 'Home_Automation'

# Not specifying a database in the connection string
connection_string = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

try:
    # Establishing a connection with SQL Server
    with pyodbc.connect(connection_string) as conn:
        with conn.cursor() as cursor:
            pass
except pyodbc.Error as ex:
    sqlstate = ex.args[1]
    print(sqlstate)  # Printing the error message in case of an exception
