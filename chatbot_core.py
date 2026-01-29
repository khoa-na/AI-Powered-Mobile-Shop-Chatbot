import sqlite3
from model_loader import *
def price(user_input):
  conn = sqlite3.connect('instance/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT phone_names, phone_money FROM thongtindienthoai WHERE phone_names LIKE '%{user_input}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchall()
  result = list()
  for row in rows:
    result.append(list(row))
  conn.close()
  return result

def screen(user_input):
  conn = sqlite3.connect('instance/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT phone_names, phone_screens FROM thongtindienthoai WHERE phone_names LIKE '%{user_input}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchall()
  result = list()
  for row in rows:
    result.append(list(row))
  conn.close()
  return result

def memory(user_input):
  conn = sqlite3.connect('instance/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT phone_names, phone_memories FROM thongtindienthoai WHERE phone_names LIKE '%{user_input}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchall()
  result = list()
  for row in rows:
    result.append(list(row))
  conn.close()
  return result

def exist(user_input):
  conn = sqlite3.connect('instance/giasanpham.db')
  c = conn.cursor()
  query = f"SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%{user_input}%' ORDER BY phone_names ASC"
  c.execute(query)
  rows = c.fetchall()
  if len(rows) == 0:
    result = "No"
  else :
    result = "Yes"
  return result

def syntax_price(user_input):
    word_to_search1 = "what is the price of"
    word_to_search2 = "What is the price of"
    if user_input.find(word_to_search1) != -1 or user_input.find(word_to_search2) != -1:
        syntax = "yes"
    else:
        syntax = "no"
    return syntax
  
def syntax_screen(user_input):
    word_to_search1 = "what is the screen size of"
    word_to_search2 = "What is the screen size of"
    if user_input.find(word_to_search1) != -1 or user_input.find(word_to_search2) != -1:
        syntax = "yes"
    else:
        syntax = "no"
    return syntax
  
def syntax_memory(user_input):
    word_to_search1 = "what is the memory of"
    word_to_search2 = "What is the memory of"
    if user_input.find(word_to_search1) != -1 or user_input.find(word_to_search2) != -1:
        syntax = "yes"
    else:
        syntax = "no"
    return syntax

def chatbot_price(user_input, product):
  syn = syntax_price(user_input)
  if syn == "yes":
    exists = exist(product)  
    if exists == "No":
        answer = f"Product {product} at this time does not available at our store. Please type the question again!"
    else:
        info = price(product)  
        if len(info) == 1:
              answer = f"At this time, there is {len(info)} product that is similar to what you typed in:\n- {info[0][0]}is priced at {info[0][1]} USD"
        else:
            answer = f"At this time, there are {len(info)} products that are similar to what you typed in:"
            for i in info:
                answer += f"\n- {i[0]}is priced at {i[1]} USD"  # Fix loop variable issue here
  else:
    answer = "Does not correct syntax, please try again using this syntax:\n- Search price: what is the price of + product\n- Example: what is the price of iphone 15"
  return answer

def chatbot_screen(user_input, product):
  syn = syntax_screen(user_input)
  if syn == "yes":
    exists = exist(product)  
    if exists == "No":
        answer = f"Product {product} at this time does not available at our store. Please type the question again!"
    else:
        info = screen(product)  
        if len(info) == 1:
              answer = f"At this time, there is {len(info)} product that is similar to what you typed in:\n- {info[0][0]}has the screen of {info[0][1]}"
        else:
            answer = f"At this time, there are {len(info)} products that are similar to what you typed in:"
            for i in info:
                answer += f"\n- {i[0]}has the screen of {i[1]}"  
  else:
    answer = "Does not correct syntax, please try again using this syntax:\n- Search price: what is the screen size of + product\n- Example: what is the screen size of iphone 15"
  return answer

def chatbot_memory(user_input, product):
  syn = syntax_memory(user_input)
  if syn == "yes":
    exists = exist(product)  
    if exists == "No":
        answer = f"Product {product} at this time does not available at our store. Please type the question again!"
    else:
        info = memory(product)  
        if len(info) == 1:
              answer = f"At this time, there is {len(info)} product that is similar to what you typed in:\n- {info[0][0]}has memory of {info[0][1]}GB"
        else:
            answer = f"At this time, there are {len(info)} products that are similar to what you typed in:"
            for i in info:
                answer += f"\n- {i[0]}has the memory of {i[1]}GB"  
  else:
    answer = "Does not correct syntax, please try again using this syntax:\n- Search price: what is the memory of + product\n- Example: what is the memory of iphone 15"
  return answer


def chatbot(user_input):
    try:
        model = last(user_input)
        name = model[1][1]
        info = model[0]
        tag = model[1][0]
        if info == "Information Tag":
            if tag == "price_tag":
                answer = chatbot_price(user_input, name)
            elif tag == "resolution_tag":
                answer = chatbot_screen(user_input, name)
            elif tag == "memory_tag":
                answer = chatbot_memory(user_input, name)
            else:
                answer = "I do not understand the information tag. Please try again."
        elif info == "Non-Information Tag":
            answer = name
    except:
        answer = "I do not understand what you typed. Please try again."
    
    return answer
  
#print(chatbot("what is the memory of iphone 15"))