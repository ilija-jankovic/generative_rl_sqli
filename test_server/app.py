from flask import Flask, jsonify, request
import pyodbc 

connection = pyodbc.connect("Driver={SQL Server};"
                      "Server=.\\SQLEXPRESS;"
                      "Database=StackOverflow2010;"
                      "Trusted_Connection=yes;")

cursor = connection.cursor()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route('/comments')
def get_incomes():
    score = request.args['score']
    comments = cursor.execute(f'SELECT TOP 500 * FROM Comments INNER JOIN Users ON Comments.UserId = Users.Id WHERE Comments.score = {score}').fetchall()

    return jsonify([str(comment) for comment in comments])
