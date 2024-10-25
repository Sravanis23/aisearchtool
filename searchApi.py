from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
import time
import json
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/courses_data')
def courseList():
    return load_courses('courses_data.json');

#scraping api
@app.route('/scrape', methods=['GET'])
def scrape():
    # Define the base URL for pagination
    base_url = "https://courses.analyticsvidhya.com/collections?page="
    root_url = "https://courses.analyticsvidhya.com"

    # Function to scrape titles and links from a page
    def scrape_titles_and_links(page_num):
        url = base_url + str(page_num)
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to retrieve page {page_num}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate all course cards
        titles_and_links = []
        cards = soup.find_all('a', class_='course-card')

        if not cards:
            print(f"No more cards found on page {page_num}. Stopping.")
            return None

        for card in cards:
            title = card.find('h3')
            link = card.get('href')
            if title and link:
                titles_and_links.append((title.text.strip(), root_url + link))

        return titles_and_links

    # Function to scrape text content from the section
    def scrape_course_content(course_url):
        response = requests.get(course_url)

        if response.status_code != 200:
            print(f"Failed to retrieve course page {course_url}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the section using a structure similar to the XPath provided
        section = soup.select_one("body main section:nth-of-type(3) article section")

        if not section:
            return None

        # Extract text content
        return section.get_text(separator=' ', strip=True)

    # Collect titles and links from all pages
    page_num = 1
    all_titles_and_links = []

    while True:
        print(f"Scraping page {page_num}...")
        titles_and_links = scrape_titles_and_links(page_num)

        if titles_and_links is None:
            break

        all_titles_and_links.extend(titles_and_links)
        page_num += 1

    # Store the course data in a list of dictionaries
    courses_data = []

    # Process links in batches of 31
    batch_size = 31
    for i in range(0, len(all_titles_and_links), batch_size):
        batch = all_titles_and_links[i:i + batch_size]

        for title, link in batch:
            print(f"Scraping course: {title} | Link: {link}")
            content = scrape_course_content(link)

            if content:
                # Store course data
                courses_data.append({
                    "title": title,
                    "link": link,
                    "content": content
                })
            else:
                print(f"Failed to retrieve content for {title}")

            # To avoid overloading the server, let's wait for a few seconds between each request
            time.sleep(2)

        # Optional: Wait before starting the next batch to be polite to the server
        time.sleep(5)

    # Write the course data to a JSON file
    with open("courses_data.json", "w") as json_file:
        json.dump(courses_data, json_file, indent=4)

    return json_file;


# Load courses from JSON file
def load_courses(file_path):
    with open(file_path, 'r') as f:
        courses = json.load(f)
    return courses


# Create embeddings for titles and contents
def create_embeddings(courses):
    course_embeddings = []
    if len(courses) > 0 :
        for course in courses:
            text = f"{course['title']} {course['content']}"  # Use 'content' field
            embedding = model.encode(text, convert_to_tensor=True)
            course_embeddings.append(embedding)
        return torch.stack(course_embeddings)  # Stack embeddings into a single tensor
    return  ""

# Find relevant courses based on keyword input
def find_relevant_courses(keyword, courses, course_embeddings, top_n=5):
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(keyword_embedding, course_embeddings)[0]

    # Get the top N results
    top_results = np.argsort(similarities.numpy())[::-1][:top_n]
    relevant_courses = [(courses[i], similarities[i].item()) for i in top_results]

    return relevant_courses


# Load courses and create embeddings on startup
courses = load_courses('courses_data.json')
course_embeddings = create_embeddings(courses)


@app.route('/search', methods=['GET'])
def search():
    keyword = str(request.args.get('keyword', ''))

    if not keyword:
        return jsonify({'error': 'Keyword is required'}), 400

    # Find relevant courses
    relevant_courses = find_relevant_courses(keyword, courses, course_embeddings)

    # Prepare the response
    response = [
        {
            'title': course['title'],
            'link': course['link'],  # Include link
            'content': course['content'],  # Include content
            'similarity': similarity
        }
        for course, similarity in relevant_courses
    ]

    return jsonify(response)


if __name__ == '__main__':
    port1 = int(os.environ.get('PORT', 8000))  # Get the PORT environment variable or use 8000 as default
    app.run(debug=True, host='0.0.0.0', port=port1)
