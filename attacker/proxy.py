#!/usr/local/bin/python
import os

def main():
    SQLMAP_PATH = '..\..\sqlmap-dev\sqlmap.py'
    URL = 'http://127.0.0.1:5000/comments_single_column?score=0'
    PARAMS = 'score'

    # --flush-session ensures full logs to get as much expert data for the DfDDPG as possible.
    # --batch ensures no blocking for user input.
    os.system(f'python {SQLMAP_PATH} -u {URL} -p {PARAMS} --flush-session --batch --output-dir="./sqlmap-log"')

if __name__ == "__main__":
    main()