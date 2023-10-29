from flask import Flask, jsonify, request
import pyodbc 

cnxn = pyodbc.connect("Driver={SQL Server};"
                      "Server=.\\SQLEXPRESS;"
                      "Database=StackOverflow2010;"
                      "Trusted_Connection=yes;")

cursor = cnxn.cursor()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

incomes = [
    { 'description': 'salary', 'amount': 5000 }
]


@app.route('/incomes')
def get_incomes():
    return jsonify(incomes)


@app.route('/incomes', methods=['POST'])
def add_income():
    incomes.append(request.get_json())
    return '', 204