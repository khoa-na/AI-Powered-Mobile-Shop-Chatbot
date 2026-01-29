import pandas as pd
import sqlite3 

# connect to the sqlite database
conn = sqlite3.connect('instance/giasanpham.db')

# query the database and load into a DataFrame
df = pd.read_sql_query("SELECT * FROM thongtindienthoai", conn)

# close the connection to the database
conn.close()

# view the DataFrame
print(df)