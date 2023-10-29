from flask import Flask, jsonify, request
import pyodbc 

connection = pyodbc.connect("Driver={SQL Server};"
                      "Server=.\\SQLEXPRESS;"
                      "Database=StackOverflow2010;"
                      "Trusted_Connection=yes;")

cursor = connection.cursor()

app = Flask(__name__)

@app.route("/")
def home():
    return "This is a vulnerable test website based on the StackOverflow 2010 public data export dataset."

@app.route('/comments')
def get_comments():
    score = request.args['score']
    comments = cursor.execute(f'SELECT TOP 500 * FROM Comments INNER JOIN Users ON Comments.UserId = Users.Id WHERE Comments.score = {score}').fetchall()

    return jsonify([str(comment) for comment in comments])
