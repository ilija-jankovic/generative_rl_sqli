#!/usr/local/bin/python
import os
import subprocess
from typing import Dict, List
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

        domain_name = urlparse(open_url).netloc
        
        # Removed port from domain name.
        self.domain_name = domain_name.split(':')[0]

        self.__dirname = os.path.dirname(__file__)
        self.__sqlmap_path = f'{self.__dirname}/../../../sqlmap-dev/sqlmap.py'
        self.__sqlmap_log_path = f'{self.__dirname}/../../sqlmap-log'

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

    def run(self, headers: Dict[str, str], cookie: str):
        headers_str = [f'{key}: {value}' for key, value in headers.items()]
        headers_str = '\n'.join(headers_str)

        # --flush-session ensures full logs to get as much expert data for the DDPGfD as possible.
        # --batch ensures no blocking for user input.
        output = subprocess.check_output([
            'python', self.__sqlmap_path,
            '-u', self.open_url,
            '-p', self.vulernable_param,
            '--data', self.default_vulnerable_param_value,
            '--method', 'POST',
            '--headers', headers_str,
            '--cookie', cookie,
            '--level', '5',
            '-v', '5',
            '--flush-session',
            '--batch',
            '--output-dir', self.__sqlmap_log_path,
            '--ignore-code', '401',
        ])
        
        output = output.decode("utf-8").split('\n')
        
        self.__write_sqlmap_output(output)

        payloads = self.__parse_attempted_sqlmap_payloads(output)
        self.__write_attempted_sqlmap_payloads(payloads)
