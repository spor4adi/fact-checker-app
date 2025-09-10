from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our fact checker
from fact_checker_project import FactChecker

app = Flask(__name__)

# Initialize the fact checker
fact_checker = FactChecker()

# Try to load existing model, train if not found
if not fact_checker.load_model():
    print("Training model for web app...")
    fact_checker.train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        post_text = data.get('text', '')
        
        if not post_text.strip():
            return jsonify({'error': 'Please enter some text to analyze'}), 400
        
        # Analyze the post
        result = fact_checker.predict_credibility(post_text)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        posts = data.get('posts', [])
        
        if not posts:
            return jsonify({'error': 'Please provide posts to analyze'}), 400
        
        # Analyze all posts
        results = fact_checker.analyze_batch(posts)
        
        # Generate summary
        summary = {
            'total_posts': len(results),
            'credible_posts': sum(1 for r in results if r['is_credible']),
            'suspicious_posts': sum(1 for r in results if not r['is_credible']),
            'average_credibility': sum(r['credibility_score'] for r in results) / len(results)
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)