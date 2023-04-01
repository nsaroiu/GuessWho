"""Copyright and Usage Information
===============================
Copyright (c) 2023 Prashanth Shyamala and Nicholas Saroiu
All Rights Reserved.
This program is the property of Prashanth Shyamala and Nicholas Saroiu.
All forms of distribution of this code, whether as given or with any changes,
are expressly prohibited.
For more information on copyright for CSC111 materials, please consult our Course Syllabus.
"""
import requests
from bs4 import BeautifulSoup
import os

# specify the URL of the website to scrape
url = 'https://guesswhocharacters.info/'

# send a GET request to the website and store the response
response = requests.get(url)

# parse the HTML content of the response using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# create a directory to store the images (if it doesn't already exist)
if not os.path.exists('images'):
    os.makedirs('images')

# find all the image elements on the page and download them
for img in soup.find_all('img'):
    # get the source URL of the image
    img_url = img.get('src')
    if img_url:
        if img_url.startswith('/'):
            img_url = url + img_url[1:]

        # get the filename of the image
        filename = os.path.join('images', os.path.basename(img_url))

        # download the image and save it to the images directory
        with open(filename, 'wb') as f:
            response = requests.get(img_url)
            f.write(response.content)

        print(f'Downloaded {filename}')
