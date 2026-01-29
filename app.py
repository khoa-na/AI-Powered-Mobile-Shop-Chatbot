from flask import Flask, render_template,request,jsonify
from chatbot_core import *
# from flask_sqlalchemy import SQLAlchemy
# import datetime

app = Flask(__file__, static_folder="./static")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///instance/giasanpham.db"



@app.route('/')
def index():
    # Connect to the SQLite database
    conn = sqlite3.connect('instance/giasanpham.db')
    cursor = conn.cursor()

    # Execute the first SELECT query
    cursor.execute("SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%iPhone 15 Pro Max%'")
    data_from_table1 = cursor.fetchall()
    
    cursor.execute("SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%iPhone 15 Pro 128GB%'")
    data_from_table2 = cursor.fetchall()
    
    cursor.execute("SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%Samsung Galaxy Z Fold5 5G 256GB%'")
    data_from_table3 = cursor.fetchall()
    
    cursor.execute("SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%OPPO Find N2 Flip%'")
    data_from_table4 = cursor.fetchall()
    
    cursor.execute("SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%Samsung Galaxy S23 Plus 5G 256GB%'")
    data_from_table5 = cursor.fetchall()
    
    conn.close()
    
    return render_template('index.html', data1=data_from_table1, data2=data_from_table2, data3=data_from_table3,
                           data4=data_from_table4, data5=data_from_table5)




@app.route('/', methods=['POST'])
def post():
    data = request.json
    user_message = data['message']
    result = chatbot(user_message)
    print(user_message)  # Print user input to the server console
    # You can perform processing on user_message if needed
    # For example, you can pass it to a chatbot model for generating a response
    # For now, let's echo the user input back to the frontend
    # Construct the response data including the user input
    response_data = {
        'user_message': user_message,
        'bot_response': result  # You can replace this with the actual bot response
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)