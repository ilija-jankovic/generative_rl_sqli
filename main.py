import os

# Scrape site solution by T0ny lombardi from:
# https://stackoverflow.com/questions/9265172/scrape-an-entire-website
os.system('wget -m -k -K -E -l 7 -t 6 -w 5 http://localhost:3000 -P ./scraped')