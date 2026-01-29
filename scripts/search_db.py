import sqlite3

conn = sqlite3.connect('instance/giasanpham.db')
c = conn.cursor()
device = input("type of device want to search: ")
if device == "phone":
    table = "thongtindienthoai"
    column = "phone_names"
elif device == "laptop":
    table = "thongtinlaptop"
    column = "laptop_names"
search = input("type your name you want to search: ")
query = f"SELECT * FROM {table} WHERE {column} LIKE '%{search}%'"
c.execute(query)

rows = c.fetchall() 

if len(rows) == 0:
    print("Device not found in database")
else:
    print(rows)

conn.close()