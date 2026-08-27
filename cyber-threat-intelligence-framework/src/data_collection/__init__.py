 """
Data Collection Module for Cyber Threat Intelligence Framework
============================================================

This module provides comprehensive data collection capabilities for the 
Cyber Threat Intelligence (CTI) framework. It handles collection from 
various sources including surface web, deep web, dark web, and specialized
threat intelligence feeds.

Module Structure:
-----------------
- WebCrawler: Base crawler for surface web data collection
- DarkWebCrawler: Specialized crawler for dark web (Tor) sources
- ForumScraper: Specialized scraper for cybersecurity forums
- APICollector: Collector for various CTI APIs
- ThreatIntelligenceAPICollector: Advanced API collector for threat feeds
- KeywordExtractor: Extract and classify cybersecurity keywords
- DatasetBuilder: Build binary and multiclass classification datasets
- IoCExtractor: Extract Indicators of Compromise from text

Usage Example:
--------------
>>> from src.data_collection import WebCrawler, ForumScraper, APICollector
>>> 
>>> # Surface web crawling
>>> crawler = WebCrawler(max_pages=100)
>>> data = asyncio.run(crawler.crawl(['https://example.com']))
>>> 
>>> # Forum scraping
>>> scraper = ForumScraper('https://forum.example.com')
>>> posts = asyncio.run(scraper.scrape_forum())
>>> 
>>> # API collection
>>> collector = APICollector({'virustotal': 'your-api-key'})
>>> threat_data = collector.collect_virustotal('malicious-domain.com')

Source Types:
-------------
- surface_web: Standard web sources (blogs, news, public forums)
- deep_web: Subscription-based feeds, API sources
- dark_web: Tor hidden services (.onion sites)
- forum: Specialized cybersecurity forums
- api_feeds: Threat intelligence APIs (VirusTotal, AlienVault, etc.)

References:
-----------
[1] Haile, A.T., Abebe, S.L., & Melaku, H.M. (2024). Real-Time Automated 
    Cyber Threat Classification and Emerging Threat Detection Framework.
    IEEE Access (Submitted).
"""

# Version information
__version__ = "1.0.0"
__author__ = "Sachin Chakrawarti"
__email__ = "sachin.chakrawarti@example.com"

# Import core classes for easy access
from .crawler import (
    WebCrawler,
    DarkWebCrawler,
    APICollector as BaseAPICollector
)

from .forum_scraper import (
    ForumScraper,
    ThreadScraper,
    PostExtractor
)

from .api_collector import (
    ThreatIntelligenceAPICollector,
    VirusTotalCollector,
    AlienVaultCollector,
    AbuseCHCollector,
    MISPCollector,
    ThreatExchangeCollector
)

from .keyword_extractor import (
    KeywordExtractor,
    ThreatKeywordDetector,
    IoCPatternMatcher
)

from .dataset_builder import (
    DatasetBuilder,
    BinaryDatasetBuilder,
    MulticlassDatasetBuilder,
    LabelEncoder,
    DatasetBalancer
)

from .ioc_extractor import (
    IoCExtractor,
    IPExtractor,
    DomainExtractor,
    URLExtractor,
    EmailExtractor,
    HashExtractor
)

from .utils import (
    DataValidator,
    DataNormalizer,
    TextCleaner,
    URLValidator,
    RateLimiter,
    SessionManager,
    DataLogger
)

# Define what gets imported with "from src.data_collection import *"
__all__ = [
    # Crawlers
    'WebCrawler',
    'DarkWebCrawler',
    'APICollector',
    
    # Forum Scrapers
    'ForumScraper',
    'ThreadScraper',
    'PostExtractor',
    
    # API Collectors
    'ThreatIntelligenceAPICollector',
    'VirusTotalCollector',
    'AlienVaultCollector',
    'AbuseCHCollector',
    'MISPCollector',
    'ThreatExchangeCollector',
    
    # Keyword Extraction
    'KeywordExtractor',
    'ThreatKeywordDetector',
    'IoCPatternMatcher',
    
    # Dataset Builders
    'DatasetBuilder',
    'BinaryDatasetBuilder',
    'MulticlassDatasetBuilder',
    'LabelEncoder',
    'DatasetBalancer',
    
    # IoC Extractors
    'IoCExtractor',
    'IPExtractor',
    'DomainExtractor',
    'URLExtractor',
    'EmailExtractor',
    'HashExtractor',
    
    # Utilities
    'DataValidator',
    'DataNormalizer',
    'TextCleaner',
    'URLValidator',
    'RateLimiter',
    'SessionManager',
    'DataLogger'
]

# Module metadata
MODULE_INFO = {
    'name': 'Data Collection',
    'description': 'Collect CTI data from surface, deep, and dark web sources',
    'version': __version__,
    'author': __author__,
    'email': __email__,
    'dependencies': [
        'aiohttp>=3.8.0',
        'beautifulsoup4>=4.12.0',
        'requests>=2.31.0',
        'selenium>=4.15.0',
        'lxml>=4.9.0'
    ]
}

def get_module_info() -> dict:
    """
    Get module information
    
    Returns:
        Dictionary with module metadata
    """
    return MODULE_INFO.copy()

def test_module():
    """
    Test function to verify module imports work correctly
    """
    print("Testing data_collection module...")
    print(f"Module version: {__version__}")
    print(f"Author: {__author__}")
    print("Available classes:", ", ".join(__all__))
    print("Module test passed!")
    return True

# Module initialization check
try:
    import aiohttp
    import requests
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Warning: Missing dependency: {e}")
    print("Please install required dependencies:")
    print("pip install -r requirements.txt")
    raise ImportError(f"Required dependency not installed: {e}")

# Logging configuration
import logging
logger = logging.getLogger(__name__)
logger.info(f"Data Collection Module v{__version__} initialized")

# Optional: Test module on import
if __name__ == "__main__":
    test_module()