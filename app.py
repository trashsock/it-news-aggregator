import streamlit as st
import asyncio
import aiohttp
from newspaper import Article
import feedparser
from langdetect import detect, LangDetectException

# Meta tag to allow iframe embedding
st.markdown(
    """
    <style>
    iframe {
        width: 100%;
        height: 100vh;
        border: none;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Predefined categories and associated keywords
categories = {
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

# Function to categorize articles based on keywords
def categorize_by_keywords(text):
    for category, keywords in categories.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            return category
    return 'Other'

# Function to detect if the article is in English
def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

# Function to fetch an article's content
async def fetch_article(session, url):
    async with session.get(url) as response:
        html = await response.text()

    article = Article(url)
    article.set_html(html)
    article.parse()

    # Get title, description, and content
    title = article.title
    meta_description = article.meta_description if article.meta_description else ""
    first_paragraph = article.text.split('\n')[0]
    last_paragraph = article.text.split('\n')[-1]

    # Combine for keyword search
    text_to_search = f"{title} {meta_description} {first_paragraph} {last_paragraph}"
    category = categorize_by_keywords(text_to_search)

    # Handle publish date
    published_date = article.publish_date
    published = published_date.strftime('%Y-%m-%d %H:%M:%S') if published_date else "Unknown"

    # Get the publisher (from the URL)
    publisher = article.source_url.split('/')[2] if article.source_url else "Unknown"

    # Check if the article is in English
    if not is_english(article.text):
        return None  # Skip non-English articles

    return {
        'title': title,
        'url': url,
        'published': published,
        'text': article.text,
        'category': category,
        'publisher': publisher
    }

# Function to fetch articles from RSS feeds
async def fetch_articles():
    articles = []
    async with aiohttp.ClientSession() as session:
        rss_feeds = [
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
        tasks = [fetch_article(session, entry.link) for feed in rss_feeds for entry in feedparser.parse(feed).entries]
        fetched_articles = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid articles and log errors
        for article in fetched_articles:
            if isinstance(article, Exception):
                st.error(f"Error fetching article: {article}")
            elif article:  # Only append non-None articles (English ones)
                articles.append(article)
    return articles

# Function to display articles
def display_news(articles):
    grouped_articles = {}
    for article in articles:
        category = article['category']
        grouped_articles.setdefault(category, []).append(article)
    
    for category, articles in grouped_articles.items():
        st.subheader(f"{category} ({len(articles)} articles)")
        for article in articles:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.markdown(f"**Publisher:** {article['publisher']}")
            st.markdown(f"**Category:** {article['category']}")
            st.markdown(f"**Published on:** {article['published']}")
            st.markdown(f"**Description:** {article['text'][:200]}...")  # Show a short preview of the article text

            # Optional: Display a button to open the full article
            if st.button(f"Read full article: {article['title']}", key=article['url']):
                st.markdown(f"[Click here to read the full article]({article['url']})")
            st.markdown("---")  # Add a separator between articles

# Main Streamlit app
def main():
    st.title("IT News Aggregator")
    if st.button("Fetch News"):
        st.write("Fetching latest IT news...")
        articles = asyncio.run(fetch_articles())
        display_news(articles)

if __name__ == '__main__':
    main()
