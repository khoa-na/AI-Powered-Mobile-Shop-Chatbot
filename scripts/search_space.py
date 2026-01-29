import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('instance/giasanpham.db')
cursor = conn.cursor()

# Search term (can be "15 pro max" or "15 promax")
search_term = "15 pro max"

# Execute the query with the LIKE operator and wildcard
cursor.execute("SELECT * FROM thongtindienthoai WHERE phone_names LIKE ?", ('%' + search_term + '%',))

# Fetch and print the results
results = cursor.fetchall()
for row in results:
    print(row)

# Close the connection
conn.close()