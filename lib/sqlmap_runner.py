#!/usr/local/bin/python
import os
import subprocess
from urllib.parse import unquote

def main():
    dirname = os.path.dirname(__file__)
    
    SQLMAP_PATH = f'{dirname}\..\..\sqlmap-dev\sqlmap.py'
    IP = 'localhost'
    PARAM = 'id'
    DEFAULT_PARAM_VALUE = '1'
    URL = f'http://{IP}/products.php?{PARAM}={DEFAULT_PARAM_VALUE}'

    # --flush-session ensures full logs to get as much expert data for the DfDDPG as possible.
    # --batch ensures no blocking for user input.
    #
    # TODO: Add cookies support. Helpful resource:
    # https://stackoverflow.com/questions/24366856/how-to-inject-a-part-of-cookie-using-sqlmap
    output = subprocess.check_output(['python', SQLMAP_PATH, '-u', URL, '-p', PARAM, '--level', '5', '-v', '5', '--flush-session', '--batch', '--output-dir', '.\\sqlmap-log'])

    lines = output.decode("utf-8").split('\n')

    payload_delimiter = f'URI: {URL[:-(len(DEFAULT_PARAM_VALUE))]}'
    lines = list(filter(lambda line: line.startswith(payload_delimiter), lines))
    lines = list(map(lambda line: unquote(line.split(payload_delimiter)[1].rstrip()), lines))

    lines = '\n'.join(lines)
    
    with open(f'{dirname}/../sqlmap-log/{IP}/attempted-payloads.txt', 'w') as f:
        f.write(lines)
    f.close()

if __name__ == '__main__':
    main()