import os
import re

class Pretrainer():

    scraped_path: str
    output_name: str

    def __init__(self, scraped_path: str, output_name: str = 'parsed-scrape'):
        self.scraped_path = scraped_path
        self.output_name = output_name

    def __getRecursivePathGenerator(self, path: str):
        for path, _, files in os.walk(self.scraped_path):
            for file in files:
                yield os.path.join(path, file)

    def parse(self):
        unique_tokens = set();
        for path in self.__getRecursivePathGenerator(self.scraped_path):
            try:
                with open(path, 'r') as f:
                    data = f.read()
                f.close()
            except:
                f.close()
                continue
            
            for token in re.split('[^a-zA-Z]+', data):
                unique_tokens.add(token)

        token_string = ''
        for token in unique_tokens:
            token_string += token + ' '
        
        with open(f'{self.output_name}.txt', 'w') as f:
            f.write(token_string)

Pretrainer('./scraped').parse()