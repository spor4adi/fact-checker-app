# AI Fact-Checker for Social Media Posts
# A comprehensive ML project for detecting misinformation and fact-checking claims

import pandas as pd
import numpy as np
import re
import pickle
import requests
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

class FactChecker:
    def __init__(self):
        self.pipeline = None
        self.vectorizer = None
        self.model = None
        self.suspicious_keywords = [
            'breaking', 'urgent', 'exclusive', 'leaked', 'secret', 'hidden',
            'they dont want you to know', 'doctors hate', 'miracle cure',
            'conspiracy', 'cover up', 'fake news', 'hoax', 'scam'
        ]
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, text):
        """Extract additional features from text"""
        features = {}
        
        # Text length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        
        # Sentiment analysis
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
        
        # Suspicious keywords count
        features['suspicious_keywords'] = sum(1 for keyword in self.suspicious_keywords 
                                            if keyword in text.lower())
        
        # Capitalization features
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['exclamation_count'] = text.count('!')
        
        return features
    
    def create_synthetic_dataset(self):
        """Create a synthetic dataset for training"""
        print("Creating synthetic training dataset...")
        
        # Reliable news samples
        reliable_posts = [
            "Scientists at MIT have published a peer-reviewed study on climate change impacts",
            "The stock market closed 2% higher today according to NYSE data",
            "New archaeological discovery in Egypt reveals ancient artifacts",
            "Local hospital reports successful surgery using new technique",
            "University research team receives grant for renewable energy project",
            "Weather service predicts heavy rainfall in coastal areas this week",
            "Government announces new infrastructure spending plan",
            "Medical journal publishes findings on diabetes treatment",
            "Space agency confirms successful satellite launch",
            "Economic indicators show steady GDP growth this quarter"
        ]
        
        # Unreliable/suspicious posts
        unreliable_posts = [
            "BREAKING: Secret government documents LEAKED! They don't want you to see this!",
            "Doctors HATE this one weird trick that cures everything!",
            "URGENT: Miracle cure hidden from public for 50 years finally revealed!",
            "EXCLUSIVE: Celebrity death hoax spreads on social media",
            "Conspiracy EXPOSED: The truth about vaccines they're hiding!",
            "FAKE NEWS ALERT: Media covers up major scandal",
            "SHOCKING: This common food is actually poison!",
            "LEAKED: Government plan to control population revealed",
            "SCAM WARNING: Don't fall for this obvious hoax",
            "COVER UP: What really happened will shock you!"
        ]
        
        # Create DataFrame
        data = []
        
        # Add reliable posts (label = 1)
        for post in reliable_posts * 10:  # Multiply for more samples
            features = self.extract_features(post)
            data.append({
                'text': post,
                'processed_text': self.preprocess_text(post),
                'label': 1,  # 1 = reliable
                **features
            })
        
        # Add unreliable posts (label = 0)
        for post in unreliable_posts * 10:  # Multiply for more samples
            features = self.extract_features(post)
            data.append({
                'text': post,
                'processed_text': self.preprocess_text(post),
                'label': 0,  # 0 = unreliable
                **features
            })
        
        # Add some variations
        variations = []
        for post in reliable_posts[:5]:
            variations.append(post + " according to official sources")
            variations.append(post.replace("Scientists", "Researchers"))
        
        for post in variations:
            features = self.extract_features(post)
            data.append({
                'text': post,
                'processed_text': self.preprocess_text(post),
                'label': 1,
                **features
            })
        
        df = pd.DataFrame(data)
        return df
    
    def train_model(self):
        """Train the fact-checking model"""
        print("Training the AI fact-checking model...")
        
        # Create dataset
        df = self.create_synthetic_dataset()
        
        # Prepare features
        X_text = df['processed_text']
        feature_cols = ['char_count', 'word_count', 'polarity', 'subjectivity', 
                       'suspicious_keywords', 'caps_ratio', 'exclamation_count']
        X_features = df[feature_cols]
        y = df['label']
        
        # Split data
        X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
            X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
        X_text_train_vec = self.vectorizer.fit_transform(X_text_train)
        X_text_test_vec = self.vectorizer.transform(X_text_test)
        
        # Combine text features with other features
        X_train_combined = np.hstack([X_text_train_vec.toarray(), X_feat_train.values])
        X_test_combined = np.hstack([X_text_test_vec.toarray(), X_feat_test.values])
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_combined, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Unreliable', 'Reliable']))
        
        # Save model
        self.save_model()
        
    def predict_credibility(self, text):
        """Predict if a post is credible or not"""
        if self.model is None or self.vectorizer is None:
            print("Model not trained yet. Please train the model first.")
            return None
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract features
        features = self.extract_features(text)
        feature_values = [features[col] for col in ['char_count', 'word_count', 'polarity', 
                         'subjectivity', 'suspicious_keywords', 'caps_ratio', 'exclamation_count']]
        
        # Vectorize text
        text_vec = self.vectorizer.transform([processed_text])
        
        # Combine features
        combined_features = np.hstack([text_vec.toarray(), [feature_values]])
        
        # Predict
        prediction = self.model.predict(combined_features)[0]
        probability = self.model.predict_proba(combined_features)[0]
        
        credibility_score = probability[1]  # Probability of being reliable
        
        result = {
            'text': text,
            'is_credible': bool(prediction),
            'credibility_score': float(credibility_score),
            'confidence': float(max(probability)),
            'features': features,
            'warning_flags': self.get_warning_flags(text, features)
        }
        
        return result
    
    def get_warning_flags(self, text, features):
        """Identify potential warning flags in the text"""
        flags = []
        
        if features['suspicious_keywords'] > 0:
            flags.append("Contains suspicious keywords")
        
        if features['caps_ratio'] > 0.3:
            flags.append("Excessive use of capital letters")
        
        if features['exclamation_count'] > 3:
            flags.append("Excessive use of exclamation marks")
        
        if features['subjectivity'] > 0.8:
            flags.append("Highly subjective language")
        
        if features['word_count'] < 10:
            flags.append("Very short post - limited context")
        
        return flags
    
    def save_model(self):
        """Save the trained model and vectorizer"""
        with open('fact_checker_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print("Model saved successfully!")
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            with open('fact_checker_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("Model files not found. Please train the model first.")
            return False
    
    def analyze_batch(self, posts):
        """Analyze multiple posts at once"""
        results = []
        for i, post in enumerate(posts):
            print(f"Analyzing post {i+1}/{len(posts)}...")
            result = self.predict_credibility(post)
            results.append(result)
        
        return results
    
    def generate_report(self, results, filename='fact_check_report.json'):
        """Generate a detailed report of the analysis"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_posts_analyzed': len(results),
            'credible_posts': sum(1 for r in results if r['is_credible']),
            'suspicious_posts': sum(1 for r in results if not r['is_credible']),
            'average_credibility_score': np.mean([r['credibility_score'] for r in results]),
            'detailed_results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")
        return report

def main():
    print("=== AI-Powered Social Media Fact Checker ===\n")
    
    # Initialize fact checker
    fact_checker = FactChecker()
    
    # Try to load existing model, if not found, train new one
    if not fact_checker.load_model():
        print("Training new model...")
        fact_checker.train_model()
    
    # Test posts for demonstration
    test_posts = [
        "Scientists at Harvard University published a new study on climate change in Nature journal",
        "BREAKING: Secret government files LEAKED! They don't want you to know the TRUTH!",
        "Local hospital reports successful heart surgery using robotic assistance",
        "MIRACLE CURE doctors don't want you to see! This ONE WEIRD TRICK will shock you!",
        "The stock market closed higher today following positive economic indicators",
        "URGENT: This common household item is actually DEADLY POISON!"
    ]
    
    print("\n=== Testing Fact Checker ===\n")
    
    results = []
    for i, post in enumerate(test_posts, 1):
        print(f"Post {i}: {post}")
        result = fact_checker.predict_credibility(post)
        results.append(result)
        
        status = "✅ CREDIBLE" if result['is_credible'] else "❌ SUSPICIOUS"
        print(f"Status: {status}")
        print(f"Credibility Score: {result['credibility_score']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        if result['warning_flags']:
            print(f"Warning Flags: {', '.join(result['warning_flags'])}")
        
        print("-" * 60)
    
    # Generate report
    report = fact_checker.generate_report(results)
    
    print(f"\n=== Analysis Summary ===")
    print(f"Total posts analyzed: {report['total_posts_analyzed']}")
    print(f"Credible posts: {report['credible_posts']}")
    print(f"Suspicious posts: {report['suspicious_posts']}")
    print(f"Average credibility score: {report['average_credibility_score']:.2f}")
    
    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Enter social media posts to fact-check (type 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter post: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input:
            result = fact_checker.predict_credibility(user_input)
            status = "✅ CREDIBLE" if result['is_credible'] else "❌ SUSPICIOUS"
            print(f"\nAnalysis Result:")
            print(f"Status: {status}")
            print(f"Credibility Score: {result['credibility_score']:.2f}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            if result['warning_flags']:
                print(f"Warning Flags: {', '.join(result['warning_flags'])}")

if __name__ == "__main__":
    # Download required NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        print("NLTK data download failed, but the model will still work")
    
    main()