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
import time

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
    'Tech Industry Trends': ['tech trends', 'technology news', 'innovation', 'startups', 'technology leadership', 'disruptive technology', 'future of IT', 'emerging tech', 'digital transformation'],
    'Others': []
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
    # AI/ML Samples
    "Deep learning algorithms are transforming predictive analytics in healthcare, enhancing diagnosis accuracy.",
    "Neural networks power advanced image recognition technologies, enabling autonomous vehicles and facial recognition systems.",
    "Generative AI is revolutionizing customer service through intelligent chatbots, improving customer experience with personalized responses.",
    "AI-powered computer vision is enabling real-time quality control in manufacturing, reducing defects and improving efficiency.",
    "Machine learning algorithms are helping businesses predict customer behavior, improving targeted marketing strategies.",
    
    # Cybersecurity Samples
    "Zero-trust security models are becoming essential for protecting against sophisticated cyber threats, ensuring secure access control.",
    "Ransomware attacks continue to challenge enterprise security strategies, prompting businesses to invest in data recovery solutions.",
    "Blockchain technology offers new approaches to secure digital transactions, eliminating intermediaries and reducing fraud risk.",
    "Advanced encryption techniques are protecting sensitive data as cyber threats evolve and data breaches become more frequent.",
    "Penetration testing has become a crucial component of cybersecurity, helping organizations identify vulnerabilities before attackers can exploit them.",
    
    # Cloud Computing Samples
    "Hybrid cloud architectures offer flexibility for enterprise IT infrastructure, enabling better disaster recovery and business continuity.",
    "Serverless computing reduces operational overhead by enabling developers to focus on code without worrying about managing servers.",
    "Multi-cloud strategies help organizations avoid vendor lock-in by distributing workloads across different cloud providers like AWS, Azure, and Google Cloud.",
    "Edge computing complements cloud services by processing data closer to the source, reducing latency and improving performance in real-time applications.",
    "Cloud storage solutions offer scalable and secure options for storing large amounts of data, providing businesses with a cost-effective alternative to physical storage.",
    
    # Software Development & DevOps Samples
    "Continuous integration (CI) improves software deployment efficiency by automating testing and ensuring smoother updates.",
    "Microservices architecture enables scalable application development by allowing each component to be developed, tested, and deployed independently.",
    "Infrastructure as Code (IaC) transforms system administration practices by automating server provisioning and configuration management.",
    "DevOps practices promote collaboration between development and operations teams, accelerating the software development lifecycle and enhancing product quality.",
    "Docker containers allow developers to create consistent environments for deploying applications across various platforms.",
    
    # Data Analytics Samples
    "Big data analytics provide actionable insights for business strategy, enabling organizations to understand consumer trends and market demands.",
    "Predictive modeling helps companies forecast market trends accurately, giving them a competitive advantage in decision-making.",
    "Data visualization tools are transforming how businesses interpret large datasets, making complex information more accessible for executives and decision-makers.",
    "AI-driven analytics are increasingly being used to process real-time data from IoT devices, helping organizations improve operational efficiency.",
    "Data lakes enable businesses to store vast amounts of structured and unstructured data, facilitating more advanced analytics and decision-making.",
    
    # Blockchain & Cryptocurrency Samples
    "Blockchain technology is revolutionizing financial transactions by eliminating intermediaries and providing transparent, secure processes.",
    "Cryptocurrency platforms are growing rapidly, offering decentralized financial solutions that bypass traditional banking systems.",
    "Ethereum’s smart contracts are enabling self-executing agreements that don't require intermediaries, offering more secure and efficient transactions.",
    "Bitcoin has established itself as a leading digital asset, driving interest in decentralized finance (DeFi) applications.",
    "The rise of NFTs (Non-Fungible Tokens) is changing the way we think about ownership and digital assets in the creative industries.",
    
    # Networking & Infrastructure Samples
    "5G networks are set to revolutionize connectivity, enabling faster communication and greater capacity for IoT devices.",
    "Software-Defined Networking (SDN) is transforming network management, providing flexibility and scalability to modern data centers.",
    "The Internet of Things (IoT) is driving new network infrastructure demands, as more devices become connected to the cloud.",
    "Network security practices are evolving to handle sophisticated cyber threats, ensuring data privacy and protection in highly connected environments.",
    "Virtual Private Networks (VPNs) are critical for securing remote workforces, offering encrypted communication channels for sensitive data.",
    
    # IT Governance & Compliance Samples
    "Regulations like GDPR are reshaping how organizations handle personal data, requiring strict compliance to avoid penalties.",
    "SOX compliance has become an essential part of corporate governance, ensuring the accuracy and integrity of financial reporting.",
    "ISO 27001 certification is a critical standard for ensuring an organization’s information security management system (ISMS) is robust.",
    "Risk management frameworks are being adopted by companies to assess, mitigate, and monitor potential security threats to IT systems.",
    "Audit trails are essential for demonstrating compliance with data protection laws, providing transparency and accountability.",
    
    # Tech Industry Trends Samples
    "The future of IT is driven by emerging technologies such as AI, blockchain, and cloud computing, which are transforming industries.",
    "Disruptive technologies are challenging traditional business models, leading to innovation and the emergence of new startups.",
    "Digital transformation is the key to staying competitive in a fast-evolving market, with companies leveraging technology for operational efficiency.",
    "Innovation in tech leadership is helping companies foster new business strategies and adapt to changing market conditions.",
    "Startups are increasingly leveraging technology to scale quickly and disrupt traditional industries with new, agile business models.",
    
    # Others (CIO-focused but not falling into other categories)
    "Digital transformation is a crucial focus for CIOs, enabling companies to streamline operations and stay competitive in a rapidly evolving market.",
    "Cost optimization through cloud adoption and automation is a key priority for modern CIOs looking to reduce IT expenses and improve operational efficiency.",
    "Talent management and workforce development have become top priorities for CIOs as they look to attract and retain skilled professionals in a competitive job market.",
    "CIOs are increasingly turning to data-driven decision-making to drive innovation, enhance customer experiences, and optimize business performance.",
    "Sustainability is becoming a critical factor for CIOs, as companies seek to implement green IT solutions and reduce their environmental footprint.",
    "CIOs are playing a key role in aligning technology strategies with business goals, ensuring that IT investments support long-term organizational growth.",
    "The adoption of 5G technology is creating new opportunities for CIOs to enhance connectivity and accelerate digital transformation across industries.",
    "As more organizations embrace remote and hybrid work, CIOs are tasked with ensuring secure, seamless collaboration tools for their workforce.",
    "Risk management strategies are becoming increasingly important for CIOs, with a focus on managing cybersecurity threats and compliance requirements."
]

# Updated Labels with All Categories
labels = [
    'AI/ML', 'AI/ML', 'AI/ML', 'AI/ML', 'AI/ML',  # AI/ML
    'Cybersecurity', 'Cybersecurity', 'Cybersecurity', 'Cybersecurity', 'Cybersecurity',  # Cybersecurity
    'Cloud Computing', 'Cloud Computing', 'Cloud Computing', 'Cloud Computing', 'Cloud Computing',  # Cloud Computing
    'Software Development & DevOps', 'Software Development & DevOps', 'Software Development & DevOps', 'Software Development & DevOps', 'Software Development & DevOps',  # Software Development & DevOps
    'Data Analytics & Big Data', 'Data Analytics & Big Data', 'Data Analytics & Big Data', 'Data Analytics & Big Data', 'Data Analytics & Big Data',  # Data Analytics & Big Data
    'Blockchain & Cryptocurrency', 'Blockchain & Cryptocurrency', 'Blockchain & Cryptocurrency', 'Blockchain & Cryptocurrency', 'Blockchain & Cryptocurrency',  # Blockchain & Cryptocurrency
    'Networking & Infrastructure', 'Networking & Infrastructure', 'Networking & Infrastructure', 'Networking & Infrastructure', 'Networking & Infrastructure',  # Networking & Infrastructure
    'IT Governance & Compliance', 'IT Governance & Compliance', 'IT Governance & Compliance', 'IT Governance & Compliance', 'IT Governance & Compliance',  # IT Governance & Compliance
    'Tech Industry Trends', 'Tech Industry Trends', 'Tech Industry Trends', 'Tech Industry Trends', 'Tech Industry Trends',  # Tech Industry Trends
    'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others', 'Others'  # Others (CIO-specific topics)
]


# Global classifier
classifier = RobustNewsClassifier()
classifier.train(sample_texts, labels)

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
                'description': article.text or ''
            }

        except Exception as e:
            logging.error(f"Processing error for {url}: {e}")
            return None

def main():
    st.set_page_config(page_title="🚀Tech News Insights🖥️", layout="wide")
    st.title("🚀Tech News Insights🖥️")
    
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
    
       # Infinite loop to refresh every 60 minutes
    while True:
        # Fetch all articles
        try:
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
                    description_text = article['description'] or "No description available."
                    if len(description_text) > 200:
                        description_text = description_text[:200] + '...'
                    st.write(description_text)
                    st.markdown(f"[Read full article]({article['url']})")
                    st.divider()
            else:
                st.write("No articles to display.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Wait for 60 minutes before refreshing the data
        time.sleep(3600)  # Sleep for 3600 seconds (60 minutes)

if __name__ == "__main__":
    main()