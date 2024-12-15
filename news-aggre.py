import asyncio
import feedparser
import aiohttp
import streamlit as st
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import langid

# Define the categories (you can expand this list with more categories)
CATEGORIES = {
    'AI & Emerging Technologies': 'Artificial Intelligence, Machine Learning, Robotics, etc.',
    'Big Data & Analytics': 'Data Science, Big Data, Predictive Analytics, etc.',
    'Cybersecurity': 'Network Security, Data Protection, Privacy, etc.',
    'Digital Transformation': 'Cloud Computing, IT, Business Transformation, etc.',
    'Cloud & Infrastructure': 'Cloud Services, IT Infrastructure, DevOps, etc.'
}

# Classifier for categorizing news articles
class CIONewsClassifier:
    def __init__(self):
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    def train(self, texts, labels):
        """ Train the classifier on provided texts and labels """
        self.model.fit(texts, labels)

    def predict(self, text):
        """ Predict the category for a given text """
        return self.model.predict([text])[0]

# Sample text data and corresponding labels for training
sample_texts = [
    "Artificial intelligence is revolutionizing business.",
    "Data science is crucial for modern business analytics.",
    "Cybersecurity is a critical part of enterprise strategy.",
    "Digital transformation is driving change in industries.",
    "Cloud computing is essential for modern IT infrastructure."
]

sample_labels = [
    'AI & Emerging Technologies',
    'Big Data & Analytics',
    'Cybersecurity',
    'Digital Transformation',
    'Cloud & Infrastructure'
]

# Train the classifier
classifier = CIONewsClassifier()
classifier.train(sample_texts, sample_labels)

# Advanced Article Fetching and Processing
async def fetch_and_process_article(session, url, classifier):
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
        
        # Check if the article is in English
        lang, _ = langid.classify(article.title)
        if lang != 'en':
            return None

        # Use the AI-powered classifier for categorization
        category = classifier.predict(full_text)

        # Prepare article metadata
        return {
            'title': article.title,
            'url': url,
            'text': full_text,
            'published': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else 'Unknown',
            'source': url.split('/')[2],
            'category': category
        }
    
    except Exception as e:
        st.error(f"Error processing {url}: {e}")
        return None

# Main function to run the Streamlit app
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
            "https://apnews.com/rss",
            "https://www.grahamcluley.com/feed/",
            "https://feeds.feedburner.com/TheHackersNews?format=xml",
            "https://www.schneier.com/blog/atom.xml"
    ]
    
    # Fetch articles automatically when the app loads
    with st.spinner('Gathering insights from top tech sources...'):
        try:
            # Fetch articles asynchronously
            async def fetch_all_articles():
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for feed in RSS_FEEDS:
                        parsed_feed = feedparser.parse(feed)
                        for entry in parsed_feed.entries[:10]:  # Limit to 10 per feed
                            tasks.append(fetch_and_process_article(session, entry.link, classifier))
                    return [article for article in await asyncio.gather(*tasks) if article]
            
            articles = asyncio.run(fetch_all_articles())
            
            # Display Results
            st.subheader("🌐 Tech Leadership Insights")
            
            # Group and display by AI-predicted categories
            for category in CATEGORIES.keys():
                category_articles = [art for art in articles if art['category'] == category]
                
                if category_articles:
                    st.markdown(f"### {category}")
                    for article in category_articles:
                        st.markdown(f"**[{article['title']}]({article['url']})**")
                        st.markdown(f"**Publisher:** {article['source']}")
                        st.markdown(f"**Category:** {article['category']}")
                        st.markdown(f"**Published on:** {article['published']}")
                        st.markdown(f"**Description:** {article['text'][:200]}...")  # Show a short preview of the article text

                        st.markdown("---")  # Add a separator between articles
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
