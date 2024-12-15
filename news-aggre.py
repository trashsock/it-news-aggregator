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
    'Cloud Computing': ['cloud', 'aws', 'azure', 'google cloud', 'virtualization', 'cloud computing', 'SaaS', 'IaaS', 'PaaS', 'cloud storage', 'multi-cloud', 'edge computing'],
    'AI/ML': ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'data science', 'natural language processing', 'computer vision', 'reinforcement learning', 'generative AI', 'GPT', 'LLM', 'predictive analytics'],
    'Cybersecurity': ['cybersecurity', 'hacking', 'data breach', 'ransomware', 'firewall', 'phishing', 'malware', 'encryption', 'identity theft', 'cyber attack', 'security breach', 'penetration testing'],
    'IT Governance & Compliance': ['governance', 'compliance', 'regulations', 'audit', 'GDPR', 'HIPAA', 'SOX', 'data privacy', 'risk management', 'internal controls', 'ISO 27001'],
    'Data Analytics & Big Data': ['data analytics', 'big data', 'business intelligence', 'data visualization', 'data mining', 'data lake', 'data pipeline', 'ETL', 'AI analytics', 'IoT data', 'structured data', 'unstructured data'],
    'Blockchain & Cryptocurrency': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'decentralized finance', 'NFT', 'smart contracts', 'blockchain technology', 'cryptocurrency security', 'tokenization'],
    'Software Development & DevOps': ['devops', 'agile', 'software development', 'CI/CD', 'microservices', 'containers', 'docker', 'kubernetes', 'serverless computing', 'API', 'code deployment', 'version control'],
    'Networking & Infrastructure': ['networking', 'network', '5G', 'network security', 'SDN', 'wifi', 'IP addressing', 'internet of things', 'network infrastructure', 'VPN', 'router', 'firewall'],
    'Tech Industry Trends': ['tech trends', 'technology news', 'innovation', 'startups', 'technology leadership', 'disruptive technology', 'future of IT', 'emerging tech', 'digital transformation'],
    'Other': []  # Add 'Other' category as fallback
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
    "The future of artificial intelligence lies in the development of explainable AI models, which will enable businesses to trust machine decisions.",
    "Quantum computing is expected to disrupt industries by solving problems that classical computers cannot, such as drug discovery and climate modeling.",
    "Robotic process automation (RPA) is transforming how companies handle routine tasks, enabling them to achieve greater efficiency and cost savings.",
    "AI-powered chatbots are now being used in customer service, offering instant support and improving customer satisfaction across industries.",
    "Generative AI is pushing the boundaries of creativity, enabling artists to collaborate with machines in ways never thought possible before.",
    "Big data analytics allows organizations to gain insights into customer behavior, leading to more personalized and targeted marketing strategies.",
    "Predictive analytics is transforming the healthcare sector by forecasting disease outbreaks and improving patient care through data-driven insights.",
    "Data visualization tools are helping business leaders interpret complex data more effectively, making strategic decisions faster and with more confidence.",
    "Companies are increasingly using big data to optimize supply chains and inventory management, reducing costs and improving operational efficiency.",
    "With the rise of Internet of Things (IoT) devices, big data analytics is playing a key role in monitoring and optimizing smart city infrastructure.",
    "Cybersecurity has become a top priority for businesses as cyberattacks become more sophisticated and frequent, threatening sensitive data.",
    "Zero-trust security models are gaining traction as they provide a more granular level of access control, improving enterprise security posture.",
    "Ransomware attacks have surged, prompting companies to invest heavily in robust backup strategies and endpoint protection.",
    "The rise of remote work has introduced new cybersecurity challenges, making it essential for businesses to adopt secure virtual private networks (VPNs).",
    "AI is being used to detect and prevent cybersecurity threats in real time, reducing the impact of data breaches and enhancing security measures.",
    "Digital transformation is essential for organizations to stay competitive in a rapidly changing business environment, driven by technology and innovation.",
    "Cloud adoption is a key element of digital transformation, enabling businesses to scale quickly and access a wide array of computing resources on demand.",
    "The shift to mobile-first strategies is a significant aspect of digital transformation, as companies optimize their services for on-the-go customers.",
    "Automation is one of the most important drivers of digital transformation, helping companies streamline operations and reduce human error.",
    "Blockchain technology is revolutionizing digital transformation by offering secure, decentralized solutions for industries like finance and logistics.",
    "Cloud computing offers businesses the flexibility to scale their IT infrastructure on demand, reducing the need for costly physical hardware.",
    "Edge computing is complementing cloud infrastructure by processing data closer to where it's generated, reducing latency and improving performance.",
    "Hybrid cloud environments are becoming increasingly popular as businesses seek to balance the security of private clouds with the scalability of public clouds.",
    "Containerization technologies like Docker are transforming how software is developed and deployed, providing a consistent environment across different stages of production.",
    "As companies move more services to the cloud, managing cloud costs effectively has become a key challenge, requiring tools to monitor usage and optimize spending."
]

# Ensure that there are 25 corresponding labels
sample_labels = [
    'AI & Emerging Technologies', 'AI & Emerging Technologies', 'AI & Emerging Technologies', 'AI & Emerging Technologies', 'AI & Emerging Technologies',
    'Big Data & Analytics', 'Big Data & Analytics', 'Big Data & Analytics', 'Big Data & Analytics', 'Big Data & Analytics',
    'Cybersecurity', 'Cybersecurity', 'Cybersecurity', 'Cybersecurity', 'Cybersecurity',
    'Digital Transformation', 'Digital Transformation', 'Digital Transformation', 'Digital Transformation', 'Digital Transformation',
    'Cloud & Infrastructure', 'Cloud & Infrastructure', 'Cloud & Infrastructure', 'Cloud & Infrastructure', 'Cloud & Infrastructure'
]

# Train the classifier
classifier = CIONewsClassifier()
classifier.train(sample_texts, sample_labels)

# Advanced Article Fetching and Processing
async def fetch_and_process_article(session, url, classifier):
    """
    Enhanced article fetching with metadata-based categorization for speed
    """
    try:
        async with session.get(url, timeout=10) as response:
            html = await response.text()

        article = Article(url)
        article.set_html(html)
        article.parse()

        # Combine title and description for faster categorization
        metadata_text = f"{article.title} {article.meta_description}"
        
        # Check if the article is in English
        lang, _ = langid.classify(article.title)
        if lang != 'en':
            return None

        # Use the AI-powered classifier for categorization based on metadata
        category = classifier.predict(metadata_text)

        # Prepare article metadata (without the full text)
        return {
            'title': article.title,
            'url': url,
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
    
    # Streamlit UI components
    st.sidebar.header("Filter by Category:")
    selected_category = st.sidebar.selectbox("Select Category", list(CATEGORIES.keys()) + ['All'])

    # Asynchronously fetch articles
    async def fetch_all_articles():
        articles = []
        async with aiohttp.ClientSession() as session:
            for feed_url in RSS_FEEDS:
                feed = feedparser.parse(feed_url)
                tasks = [fetch_and_process_article(session, entry.link, classifier) for entry in feed.entries]
                articles.extend(await asyncio.gather(*tasks))
        return articles
    
    # Run the async function
    articles = asyncio.run(fetch_all_articles())

    # Filter articles based on selected category
    if selected_category != 'All':
        articles = [article for article in articles if article['category'] == selected_category]

    # Display articles
    if articles:
        for article in articles:
            st.subheader(article['title'])
            st.write(f"Category: {article['category']}")
            st.write(f"Published on: {article['published']}")
            st.write(f"Source: {article['source']}")
            st.write(article['text'][:200] + '...')  # Limit to first 500 characters
            st.markdown(f"[Read full article]({article['url']})")
    else:
        st.write("No articles to display.")
