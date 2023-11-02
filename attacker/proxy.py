#!/usr/local/bin/python
import subprocess

def main():
    SQLMAP_PATH = '..\\..\\sqlmap-dev\\sqlmap.py'
    IP = '127.0.0.1'
    PORT = '5000'
    PARAM = 'score'
    DEFAULT_PARAM_VALUE = '0'
    URL = f'http://{IP}:{PORT}/comments_single_column?{PARAM}={DEFAULT_PARAM_VALUE}'

    # --flush-session ensures full logs to get as much expert data for the DfDDPG as possible.
    # --batch ensures no blocking for user input.
    output = subprocess.check_output(['python', SQLMAP_PATH, '-u', URL, '-p', PARAM, '--level', '5', '-v', '5', '--flush-session', '--batch', '--output-dir', '.\\sqlmap-log'])

    lines = output.decode("utf-8").split('\n')

    payload_delimiter = f'URI: {URL[:-(len(DEFAULT_PARAM_VALUE))]}'
    lines = list(filter(lambda line: line.startswith(payload_delimiter), lines))
    lines = list(map(lambda line: line.split(payload_delimiter)[1].rstrip(), lines))
    
    with open(f'sqlmap-log/{IP}/attempted-payloads.txt', 'w') as f:
        f.writelines(lines)
    f.close()

if __name__ == '__main__':
    main()