import pandas as pd
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="Arya",
    password="root",
    database="sample_db"
)

df = pd.read_sql("SELECT * FROM orders", conn)

print(df)
conn.close()
