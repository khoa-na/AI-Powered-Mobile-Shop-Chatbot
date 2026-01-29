import csv
import sqlite3

# Open the CSV file
with open('instance/thongtindienthoai.csv', 'r') as f:
    reader = csv.reader(f)
    
    # Skip the header row
    next(reader)
    
    # Connect to the SQLite database
    conn = sqlite3.connect('instance/giasanpham.db')
    c = conn.cursor()
    
    # Create the table
    c.execute('''CREATE TABLE thongtindienthoai
                 ( phone_names TEXT, phone_money TEXT, phone_chips TEXT, phone_memories TEXT, phone_screens TEXT, camera_selfies TEXT)''')
    
    # Insert each row of data
    for row in reader:
        c.execute('INSERT INTO thongtindienthoai VALUES (?,?,?,?,?,?)', row)
        
    # Save and close
    conn.commit()
    conn.close()