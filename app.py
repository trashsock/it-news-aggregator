import streamlit as st
import asyncio
import aiohttp
from newspaper import Article
from datetime import datetime
import streamlit as st
import feedparser

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

# Function to search for keywords in the article content
def categorize_by_keywords(text):
    for category, keywords in categories.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            return category
    return 'Other'  # Default category if no keywords are found

# Fetch and process articles
async def fetch_article(session, url):
    article = Article(url)
    article.download()
    article.parse()
    
    # Get the title, description, and first/last paragraphs for keyword search
    title = article.title
    meta_description = article.meta_description if article.meta_description else ""
    first_paragraph = article.text.split('\n')[0]
    last_paragraph = article.text.split('\n')[-1]
    
    # Combine text to search for keywords
    text_to_search = f"{title} {meta_description} {first_paragraph} {last_paragraph}"
    category = categorize_by_keywords(text_to_search)
    # Ensure publish_date exists before attempting to access it
    published_date = article.publish_date if article.publish_date is not None else []
    if published_date:
        # If publish_date is not None, format it
        published = datetime(*published_date[:6]).strftime('%Y-%m-%d %H:%M:%S')
    else:
        # If publish_date is None, assign a default value or handle accordingly
        published = "Unknown"
    
    return {
            'title': article.title,
            'url': article.url,
            'published': published,  # Use the formatted date
            'text': article.text,
            'source': url
        }

# Function to fetch articles from RSS
async def fetch_articles():
    articles = []
    async with aiohttp.ClientSession() as session:
        # Replace with more RSS feeds (if needed)
        rss_feeds = [
            'https://www.cio.com/feed/',
            'https://techcrunch.com/feed/',
            'https://www.theverge.com/rss/index.xml',
            'https://www.zdnet.com/news/rss.xml',
            'https://www.wired.com/feed/',
            'https://arstechnica.com/feed/',
            'https://mashable.com/feed/',
            'https://venturebeat.com/feed/',
            'https://www.infoworld.com/index.rss',
            'https://www.networkworld.com/news/rss.xml',
            'https://www.computerworld.com/index.rss'
        ]
        tasks = []
        for feed in rss_feeds:
            rss_data = feedparser.parse(feed)
            for entry in rss_data.entries:
                tasks.append(fetch_article(session, entry.link))

        # Wait for all article fetch tasks to complete
        fetched_articles = await asyncio.gather(*tasks)
        for article in fetched_articles:
            articles.append({
                'link': article['link'],
                'category': article['category'],
                'published': article['published']
            })
    return articles

# Streamlit user interface
def display_news(articles):
    # Group by categories
    grouped_articles = {}
    for article in articles:
        if article['category'] not in grouped_articles:
            grouped_articles[article['category']] = []
        grouped_articles[article['category']].append(article)
    
    for category, articles in grouped_articles.items():
        st.subheader(f"{category} ({len(articles)} articles)")
        for article in articles:
            st.write(f"[{article['link']}]({article['link']}) - Published on {article['published']}")

# Main Streamlit app
def main():
    st.title("IT News Aggregator")
    
    if st.button("Fetch News"):
        st.write("Fetching latest IT news...")
        # Fetch articles asynchronously
        articles = asyncio.run(fetch_articles())
        # Display articles in Streamlit
        display_news(articles)

if __name__ == '__main__':
    main()
