import sqlite3

conn = sqlite3.connect('instance/giasanpham.db')
c = conn.cursor()
search = input("type your name you want to search: ")
query = f"SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%{search}%'"
c.execute(query)

rows = c.fetchall() 

if len(rows) == 0:
    print("Device not found in database")
else:
    for row in rows:
        print(row)

conn.close()