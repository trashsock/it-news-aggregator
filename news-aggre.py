import streamlit as st
import asyncio
import aiohttp
from newspaper import Article
import feedparser
import pandas as pd
import numpy as np

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import joblib

# Additional NLP
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import re

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)

# CIO-Focused Categories with Sophisticated Keyword Mapping
CATEGORIES = {
    'AI & Emerging Technologies': [
        'artificial intelligence', 'machine learning', 'generative ai', 'gpt', 
        'llm', 'neural networks', 'deep learning', 'predictive ai', 'cognitive computing', 
        'transformative technology', 'algorithmic innovation'
    ],
    'Big Data & Analytics': [
        'data analytics', 'big data', 'business intelligence', 'predictive analytics', 
        'data science', 'data visualization', 'machine learning analytics', 
        'enterprise data strategy', 'data-driven decision making'
    ],
    'Cybersecurity': [
        'cybersecurity', 'data breach', 'ransomware', 'threat intelligence', 
        'information security', 'network protection', 'zero trust', 
        'security strategy', 'cyber resilience', 'risk mitigation'
    ],
    'Digital Transformation': [
        'digital transformation', 'digital strategy', 'technology innovation', 
        'business model disruption', 'digital ecosystem', 'technological convergence', 
        'strategic technology', 'enterprise modernization'
    ],
    'Enterprise IT Strategy': [
        'it governance', 'technology leadership', 'enterprise architecture', 
        'strategic planning', 'digital innovation', 'it infrastructure', 
        'organizational change', 'technology roadmap'
    ],
    'Cloud & Infrastructure': [
        'cloud computing', 'hybrid cloud', 'multi-cloud', 'cloud strategy', 
        'infrastructure as code', 'serverless', 'cloud migration', 
        'edge computing', 'cloud security'
    ]
}

# Text Preprocessing Function
def preprocess_text(text):
    """
    Advanced text preprocessing for ML model
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Advanced Multi-Label Classifier
class CIONewsClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = LabelEncoder()
        self.classifier = None
    
    def train(self, texts, labels):
        """
        Train a multi-label classifier
        """
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Train SVM Classifier
        self.classifier = OneVsRestClassifier(LinearSVC(random_state=42))
        self.classifier.fit(X, y)
    
    def predict(self, text):
        """
        Predict category for a given text
        """
        processed_text = preprocess_text(text)
        X_test = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(X_test)
        return self.label_encoder.inverse_transform(prediction)[0]

# Advanced Article Fetching and Processing
async def fetch_and_process_article(session, url):
    """
    Enhanced article fetching with more robust processing
    """
    try:
        async with session.get(url, timeout=10) as response:
            html = await response.text()

        article = Article(url)
        article.set_html(html)
        article.parse()

        # Combine multiple text sources for better classification
        full_text = f"{article.title} {article.meta_description} {article.text}"
        
        # Keyword-based initial categorization as fallback
        def keyword_categorize(text):
            text_lower = text.lower()
            for category, keywords in CATEGORIES.items():
                if any(keyword in text_lower for keyword in keywords):
                    return category
            return 'Other'

        # Prepare article metadata
        return {
            'title': article.title,
            'url': url,
            'text': full_text,
            'published': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else 'Unknown',
            'source': url.split('/')[2],
            'initial_category': keyword_categorize(full_text)
        }
    
    except Exception as e:
        st.error(f"Error processing {url}: {e}")
        return None

# Main Streamlit Application
def main():
    st.title("🚀 CIO Tech Insight Aggregator")
    
    # RSS Feeds focused on enterprise and technology leadership
    RSS_FEEDS = [
            'https://www.cio.com/feed/',
            'https://techcrunch.com/feed/',
            'https://www.theverge.com/rss/index.xml',
            'https://www.zdnet.com/news/rss.xml',
            'https://www.wired.com/feed/',
            'https://arstechnica.com/feed/',
            'https://mashable.com/feed/',
            'https://www.infoworld.com/index.rss',
            'https://www.networkworld.com/news/rss.xml',
            'https://www.computerworld.com/index.rss',
            "https://asia.nikkei.com/rss",
            "https://www.bloomberg.com/feeds/bbiz.xml",
            "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
            "https://apnews.com/rss"
        ]
    # Fetch News Button
    if st.button("🔍 Fetch Latest Tech Insights"):
        # Spinner for loading
        with st.spinner('Gathering insights from top tech sources...'):
            try:
                # Fetch articles asynchronously
                async def fetch_all_articles():
                    async with aiohttp.ClientSession() as session:
                        tasks = []
                        for feed in RSS_FEEDS:
                            parsed_feed = feedparser.parse(feed)
                            for entry in parsed_feed.entries[:10]:  # Limit to 10 per feed
                                tasks.append(fetch_and_process_article(session, entry.link))
                        return [article for article in await asyncio.gather(*tasks) if article]
                
                articles = asyncio.run(fetch_all_articles())
                
                # Prepare data for classification
                texts = [article['text'] for article in articles]
                initial_categories = [article['initial_category'] for article in articles]
                
                # Train AI Classifier
                classifier = CIONewsClassifier()
                classifier.train(texts, initial_categories)
                
                # Enhance categorization
                for article in articles:
                    article['ai_category'] = classifier.predict(article['text'])
                
                # Display Results
                st.subheader("🌐 Tech Leadership Insights")
                
                # Group and display by AI-predicted categories
                for category in CATEGORIES.keys():
                    category_articles = [art for art in articles if art['ai_category'] == category]
                    
                    if category_articles:
                        st.markdown(f"### {category}")
                        
                        for article in category_articles[:5]:  # Top 5 per category
                            st.markdown(f"#### [{article['title']}]({article['url']})")
                            st.markdown(f"**Source:** {article['source']} | **Published:** {article['published']}")
                            st.markdown("---")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()