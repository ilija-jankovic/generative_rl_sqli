#!/usr/local/bin/python
import os
import subprocess
from typing import List
from urllib.parse import unquote, urlparse

class SqlmapRunner:
    open_url: str
    vulernable_param: str
    default_vulnerable_param_value: str
    domain_name: str

    __sqlmap_path: str
    __sqlmap_log_path: str

    def __init__(self, open_url: str, vulernable_param: str,
                 default_vulnerable_param_value: str):
        self.open_url = open_url
        self.vulernable_param = vulernable_param
        self.default_vulnerable_param_value = default_vulnerable_param_value

        self.domain_name = urlparse(open_url).netloc

        self.__dirname = os.path.dirname(__file__)
        self.__sqlmap_path = f'{self.__dirname}\..\..\sqlmap-dev\sqlmap.py'
        self.__sqlmap_log_path = f'{self.__dirname}/../sqlmap-log'

    def __write_sqlmap_output(self, output: List[str]):
        lines = '\n'.join(output)

        with open(f'{self.__sqlmap_log_path}/{self.domain_name}/sqlmap-output.txt', 'w') as f:
            f.write(lines)
        f.close()

    def __parse_attempted_sqlmap_payloads(self, output: List[str]):
        payload_delimiter = f'URI: {self.open_url}'
        payloads = list(filter(lambda line: line.startswith(payload_delimiter), output))

        return list(map(lambda line: unquote(line.split(payload_delimiter)[1].rstrip()), payloads))

    def __write_attempted_sqlmap_payloads(self, payloads: List[str]):
        lines = '\n'.join(payloads)

        with open(f'{self.__sqlmap_log_path}/{self.domain_name}/attempted-payloads.txt', 'w') as f:
            f.write(lines)
        f.close()

    def run(self, cookie: str):
        url = self.open_url + self.default_vulnerable_param_value

        # --flush-session ensures full logs to get as much expert data for the DDPGfD as possible.
        # --batch ensures no blocking for user input.
        output = subprocess.check_output([
            'python', self.__sqlmap_path,
            '-u', url,
            '-p', self.vulernable_param,
            '--headers', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            '--cookie', cookie,
            '--level', '5',
            '-v', '5',
            '--flush-session',
            '--batch',
            '--output-dir', self.__sqlmap_log_path])
        
        output = output.decode("utf-8").split('\n')
        
        self.__write_sqlmap_output(output)

        payloads = self.__parse_attempted_sqlmap_payloads(output)
        self.__write_attempted_sqlmap_payloads(payloads)
