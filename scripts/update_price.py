import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('instance/giasanpham.db')
cursor = conn.cursor()

# Fetch the data from the thongtindienthoai table
cursor.execute('SELECT phone_names, phone_money FROM thongtindienthoai')
rows = cursor.fetchall()

# Update the phone_money values and commit the changes
for row in rows:
    phone_money = row[1].replace('.', '')  # Remove existing periods
    formatted_phone_money = '{:,.0f}'.format(float(phone_money)).replace(',', ',')  # Add periods as thousands separators
    cursor.execute('UPDATE thongtindienthoai SET phone_money = ? WHERE phone_names = ?', (formatted_phone_money, row[0]))

# Commit the changes and close the connection
conn.commit()
conn.close()
