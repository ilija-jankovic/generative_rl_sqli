from flask import Flask, jsonify, request
import pyodbc 

cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=Test Server;"
                      "Database=database.mdf;"
                      "Trusted_Connection=yes;")

cursor = cnxn.cursor()

print(cursor.tables().description)

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