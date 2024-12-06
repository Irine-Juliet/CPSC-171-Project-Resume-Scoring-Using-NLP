from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk

# Initialize NLTK tools
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Flask app initialization
app = Flask(__name__)

# Preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

# Function to calculate average Word2Vec vectors
def get_average_vector(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Function to calculate similarity score
def calculate_similarity_score(resume, job_title, model, vector_size=100):
    processed_resume = preprocess_text(resume)
    processed_job_title = preprocess_text(job_title)
    resume_tokens = word_tokenize(processed_resume)
    job_title_tokens = word_tokenize(processed_job_title)
    resume_vector = get_average_vector(resume_tokens, model, vector_size)
    job_title_vector = get_average_vector(job_title_tokens, model, vector_size)
    if resume_vector.any() and job_title_vector.any():
        similarity = cosine_similarity([resume_vector], [job_title_vector])[0][0]
        return round(similarity * 10, 2)  # Scale to 0–10
    else:
        return 0.0

# Load pre-trained Word2Vec model
word2vec_model = Word2Vec.load("word2vec_resume.model")  # Replace with your model path

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def get_score():
    if request.method == 'POST':
        # Get inputs from the form
        resume = request.form['resume']
        job_title = request.form['job_title']
        
        # Calculate the score
        score = calculate_similarity_score(resume, job_title, word2vec_model, vector_size=100)
        return render_template('index.html', score=score, resume=resume, job_title=job_title)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
