# AI Fact Checker Web App

A Flask-based web application that uses **Machine Learning (Random Forest + TF-IDF)** and **NLP (TextBlob, NLTK)** to analyze social media posts or news snippets for credibility.

## Features
- Single post analysis with credibility score & confidence.
- Warning flags for suspicious patterns (clickbait, all caps, etc.).
- Clean modern UI with results highlighting.

## Tech Stack
- Python, Flask
- scikit-learn, TextBlob, NLTK
- HTML/CSS/JS (custom UI)

## Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/fact-checker-app.git
cd fact-checker-app
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements_file.txt
pip install flask
python fact_checker_project.py   # Train model
python web_app.py
