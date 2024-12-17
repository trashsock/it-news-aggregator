import asyncio
import feedparser
import aiohttp
import streamlit as st
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import langid
from functools import lru_cache
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Define the categories with more precise keywords
CATEGORIES = {
    'Cloud Computing': ['cloud', 'aws', 'azure', 'google cloud', 'virtualization', 'cloud computing', 'SaaS', 'IaaS', 'PaaS', 'cloud storage', 'multi-cloud', 'edge computing'],
    'AI/ML': ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'data science', 'natural language processing', 'computer vision', 'reinforcement learning', 'generative AI', 'GPT', 'LLM', 'predictive analytics'],
    'Cybersecurity': ['cybersecurity', 'hacking', 'data breach', 'ransomware', 'firewall', 'phishing', 'malware', 'encryption', 'identity theft', 'cyber attack', 'security breach', 'penetration testing'],
    'IT Governance & Compliance': ['governance', 'compliance', 'regulations', 'audit', 'GDPR', 'HIPAA', 'SOX', 'data privacy', 'risk management', 'internal controls', 'ISO 27001'],
    'Data Analytics & Big Data': ['data analytics', 'big data', 'business intelligence', 'data visualization', 'data mining', 'data lake', 'data pipeline', 'ETL', 'AI analytics', 'IoT data', 'structured data', 'unstructured data'],
    'Blockchain & Cryptocurrency': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'decentralized finance', 'NFT', 'smart contracts', 'blockchain technology', 'cryptocurrency security', 'tokenization'],
    'Software Development & DevOps': ['devops', 'agile', 'software development', 'CI/CD', 'microservices', 'containers', 'docker', 'kubernetes', 'serverless computing', 'API', 'code deployment', 'version control'],
    'Networking & Infrastructure': ['networking', 'network', '5G', 'network security', 'SDN', 'wifi', 'IP addressing', 'internet of things', 'network infrastructure', 'VPN', 'router', 'firewall'],
    'Tech Industry Trends': ['tech trends', 'technology news', 'innovation', 'startups', 'technology leadership', 'disruptive technology', 'future of IT', 'emerging tech', 'digital transformation']
}

class RobustNewsClassifier:
    def __init__(self):
        self.model = make_pipeline(TfidfVectorizer(stop_words='english', max_features=5000), MultinomialNB())

    def train(self, texts, labels):
        """ Train the classifier with more robust vectorization """
        self.model.fit(texts, labels)

    def predict(self, text):
        """ Predict category with fallback """
        try:
            return self.model.predict([text])[0]
        except Exception:
            return 'Tech Industry Trends'  # Default category

# Sample training data
sample_texts = [
    "The future of artificial intelligence lies in the development of explainable AI models, which will enable businesses to trust machine decisions.",
    "Quantum computing is expected to disrupt industries by solving problems that classical computers cannot, such as drug discovery and climate modeling.",
    "Cybersecurity has become a top priority for businesses as cyberattacks become more sophisticated and frequent, threatening sensitive data.",
    "Cloud computing offers businesses the flexibility to scale their IT infrastructure on demand, reducing the need for costly physical hardware.",
    "Blockchain technology is revolutionizing digital transformation by offering secure, decentralized solutions for industries like finance and logistics."
]

sample_labels = [
    'AI/ML', 'AI/ML', 
    'Cybersecurity', 
    'Cloud Computing', 
    'Blockchain & Cryptocurrency'
]

# Global classifier
classifier = RobustNewsClassifier()
classifier.train(sample_texts, sample_labels)

@lru_cache(maxsize=500)
def classify_article(text):
    """ Cached classification to reduce redundant processing """
    return classifier.predict(text)

async def fetch_article_safely(session, url, timeout=10):
    """
    Safely fetch article content with robust error handling
    """
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                logging.warning(f"Failed to fetch {url}: HTTP {response.status}")
                return None
            return await response.text()
    except asyncio.TimeoutError:
        logging.warning(f"Timeout fetching {url}")
        return None
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

async def process_article(session, url, semaphore):
    """
    Process a single article with concurrency control and efficient parsing
    """
    async with semaphore:
        try:
            html = await fetch_article_safely(session, url)
            if not html:
                return None

            # Use Article with minimal processing
            article = Article(url, fetch_images=False, fetch_videos=False)
            article.set_html(html)
            article.parse()

            # Language and content checks
            if not article.title or langid.classify(article.title)[0] != 'en':
                return None

            # Efficient categorization
            metadata_text = f"{article.title or ''} {article.meta_description or ''}"
            category = classify_article(metadata_text)

            return {
                'title': article.title or 'Untitled',
                'url': url,
                'published': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else 'Unknown',
                'source': urlparse(url).netloc,
                'category': category,
                'description': article.summary or ''
            }

        except Exception as e:
            logging.error(f"Processing error for {url}: {e}")
            return None

def main():
    st.set_page_config(page_title="Tech News Aggregator", layout="wide")
    st.title("🚀 Tech News Insight Aggregator")
    
    # Full list of RSS Feeds
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
        "https://apnews.com/rss",
        "https://www.grahamcluley.com/feed/",
        "https://feeds.feedburner.com/TheHackersNews?format=xml",
        "https://www.schneier.com/blog/atom.xml"
    ]
    
    # Create buttons for each category
    category_columns = st.columns(len(CATEGORIES))
    selected_category = None
    
    for i, (category, _) in enumerate(CATEGORIES.items()):
        with category_columns[i]:
            if st.button(category, use_container_width=True):
                selected_category = category
    
    # Add an "All News" button
    if st.button("All News", use_container_width=True):
        selected_category = None

    async def fetch_all_articles():
        # Controlled concurrency with higher limit
        semaphore = asyncio.Semaphore(20)
        articles = []
        
        async with aiohttp.ClientSession() as session:
            for feed_url in RSS_FEEDS:
                try:
                    feed = feedparser.parse(feed_url)
                    # Process more entries per feed with controlled concurrency
                    entry_tasks = [
                        process_article(session, entry.link, semaphore) 
                        for entry in feed.entries[:15]  # Increased from 10 to 15
                    ]
                    feed_articles = await asyncio.gather(*entry_tasks)
                    articles.extend([a for a in feed_articles if a is not None])
                except Exception as e:
                    st.error(f"Error fetching feed {feed_url}: {e}")
        
        return articles
    
    # Run the async function
    try:
        # Fetch all articles
        articles = asyncio.run(fetch_all_articles())

        # Filter articles if a specific category is selected
        if selected_category:
            articles = [article for article in articles if article['category'] == selected_category]

        # Display articles
        if articles:
            for article in articles:
                st.subheader(article['title'])
                st.write(f"Category: {article['category']}")
                st.write(f"Published on: {article['published']}")
                st.write(f"Source: {article['source']}")
                st.write(article['description'][:200] + '...')  # Limit to first 200 characters
                st.markdown(f"[Read full article]({article['url']})")
                st.divider()
        else:
            st.write("No articles to display.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()