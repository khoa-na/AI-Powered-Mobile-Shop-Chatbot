import sqlite3
import random

def price(name):
  conn = sqlite3.connect('instance/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%{name}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchone()
  result = rows
  conn.close()
  return result

def chatbot(name):
  random_number = random.randint(1, 4)
  result = price(name)
  if random_number == 1:
    answer = f"Product {result[0]}is priced at {result[1]} USD"
  elif random_number == 2:
    answer = f"{result[0]}costs {result[1]} USD"
  elif random_number == 3:
    answer = f"You can buy {result[0]}for {result[1]} USD"
  elif random_number == 4:
    answer = f"{result[0]}is being sold at {result[1]} USD"
  return answer
 