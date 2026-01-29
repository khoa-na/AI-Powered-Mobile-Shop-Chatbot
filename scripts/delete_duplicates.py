import sqlite3

conn = sqlite3.connect('instance/giasanpham.db')
cursor = conn.cursor()
cursor.execute('''
    DELETE FROM thongtinlaptop 
    WHERE ROWID NOT IN (
        SELECT MIN(ROWID) 
        FROM thongtinlaptop 
        GROUP BY laptop_names
    );
''')

conn.commit()
conn.close()

