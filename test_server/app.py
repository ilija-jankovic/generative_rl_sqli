#!/usr/local/bin/python

from sqlite3 import Row
from typing import List
from flask import Flask, jsonify, request
import pyodbc 

connection = pyodbc.connect("Driver={SQL Server};"
                      "Server=.\\SQLEXPRESS;"
                      "Database=StackOverflow2010;"
                      "Trusted_Connection=yes;")

cursor = connection.cursor()

app = Flask(__name__)


def get_names(query: str):
    rows = cursor.execute(query).fetchall()
    return [row[0] for row in rows]


def print_table_and_column_names():
    print('ALL TABLES:')
    for table in get_names('SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES'):
        print(table)

    print('\nALL COLUMNS:')
    for column in get_names('SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS'):
        print(column)
    print('\n')


@app.route("/")
def home():
    return "This is a vulnerable test website based on the StackOverflow 2010 public data export dataset."


@app.route('/comments_all_columns')
def get_comments_all_columns():
    score = request.args['score']

    comments = cursor.execute(f'SELECT TOP 500 * FROM Comments WHERE score = {score}').fetchall()

    return jsonify([str(comment) for comment in comments])


@app.route('/comments_all_columns_join_user')
def get_comments_all_columns_join_user():
    score = request.args['score']

    comments = cursor.execute(f'SELECT TOP 500 * FROM Comments INNER JOIN Users ON Comments.UserId = Users.Id WHERE Comments.score = {score}').fetchall()

    return jsonify([str(comment) for comment in comments])


@app.route('/comments_single_column')
def get_comments_single_column():
    score = request.args['score']

    comments = cursor.execute(f'SELECT TOP 500 text FROM Comments WHERE score = {score}').fetchall()

    return jsonify([str(comment) for comment in comments])
