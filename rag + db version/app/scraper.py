import re
import time
import hashlib
import logging
import json 
from typing import List, Dict, Optional, Any, Set, Tuple
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import deque

from bs4 import BeautifulSoup, element
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

from config.settings import SCRAPE_CONFIG
from app.database import DatabaseManager

logger = logging.getLogger(__name__)

class NYCAdminCodeScraper:
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.base_url = SCRAPE_CONFIG["base_url"]
        self.delay = SCRAPE_CONFIG.get("delay", 1.0)
        self.processed_urls: Set[str] = set()
        self.processed_sections: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.driver = self._init_selenium()
        self.sections_saved = 0
        self.sections_skipped = 0
        self.pages_processed = 0
        self.title_patterns = set()
    
    # Initializes and returns a Selenium WebDriver instance    
    def _init_selenium(self) -> Optional[webdriver.Chrome]:
        logger.info("Initializing Selenium WebDriver...")
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)
            logger.info("Selenium WebDriver initialized successfully.")
            return driver
        except Exception as e:
            logger.error(f"FATAL: Failed to initialize Selenium driver: {e}")
            return None
        
    # Fetches a URL using Selenium with retry logic
    def _get_soup_with_selenium(self, url: str, retry_count: int = 2) -> Optional[BeautifulSoup]:
        if url in self.processed_urls or not self.driver:
            return None
        
        for attempt in range(retry_count):
            try:
                logger.info(f"Fetching: {url} (attempt {attempt + 1})")
                self.driver.get(url)
                
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "jsx-parser"))
                    )
                except TimeoutException:
                    try:
                        WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "codenav__toc"))
                        )
                    except TimeoutException:
                        time.sleep(2)
                
                time.sleep(self.delay)
                
                self.processed_urls.add(url)
                self.pages_processed += 1
                
                if self.pages_processed % 10 == 0:
                    logger.info(f"Progress: {self.pages_processed} pages processed, "
                              f"{self.sections_saved} sections saved")
                
                return BeautifulSoup(self.driver.page_source, 'lxml')
                
            except TimeoutException:
                logger.warning(f"Timeout loading {url}, attempt {attempt + 1}")
                if attempt == retry_count - 1:
                    self.failed_urls.add(url)
            except Exception as e:
                logger.error(f"Error loading {url}: {e}")
                if attempt == retry_count - 1:
                    self.failed_urls.add(url)
                    
        return None

    # Extract ALL links from a page
    @staticmethod
    def _extract_all_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        links = set()
        
        toc_container = soup.find('div', class_='codenav__toc')
        if toc_container:
            for link in toc_container.find_all('a', href=True):
                href = link.get('href')
                if href and '/NYCadmin/' in href:
                    full_url = urljoin(current_url, href)
                    links.add(full_url)
        
        breadcrumb = soup.find('nav', {'aria-label': 'breadcrumb'})
        if breadcrumb:
            for link in breadcrumb.find_all('a', href=True):
                href = link.get('href')
                if href and '/NYCadmin/' in href:
                    full_url = urljoin(current_url, href)
                    links.add(full_url)
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and '/NYCadmin/0-0-0-' in href:
                full_url = urljoin(current_url, href)
                links.add(full_url)
                
                match = re.search(r'0-0-0-(\d+)', full_url)
                if match:
                    self.title_patterns.add(match.group(1))
        
        return list(links)

    # Enhanced section data extraction
    def _extract_section_data(self, jsx_parser_div: element.Tag) -> Optional[Dict[str, Any]]:
        text = jsx_parser_div.get_text(strip=True)
        
        if len(text) < 100:
            return None
        
        text = re.sub(r'^ShareDownloadBookmarkPrint', '', text)
        text = re.sub(r'^Share Download Bookmark Print', '', text)
        
        patterns = [
            r'(§\s*[\d\w.\-]+)\s*([^.]+?)\.(.+)',
            r'(§\s*[\d\w.\-]+)\s+([A-Z][^.]+?)\s+([a-z].+)',
            r'(§\s*[\d\w.\-]+)\s*([^:]+?):(.+)',
            r'(§\s*[\d\w.\-]+)(.+)',
        ]
        
        section_match = None
        for pattern in patterns:
            section_match = re.match(pattern, text, re.DOTALL)
            if section_match:
                break
        
        if not section_match:
            return None
        
        groups = section_match.groups()
        if len(groups) >= 3:
            section_number = groups[0].strip()
            title = groups[1].strip()
            content = groups[2].strip()
        else:
            section_number = groups[0].strip()
            remaining = groups[1].strip()
            
            title_end = re.search(r'[.!?]\s+[a-z]', remaining)
            if title_end:
                title = remaining[:title_end.start() + 1].strip()
                content = remaining[title_end.start() + 1:].strip()
            else:
                lines = remaining.split('\n', 1)
                title = lines[0].strip('.:')
                content = lines[1] if len(lines) > 1 else remaining
        
        title = re.sub(r'^[.:;\s]+', '', title)
        title = re.sub(r'[.:;\s]+$', '', title)
        
        if not title or title == section_number:
            first_sentence = re.match(r'^[^.!?]+[.!?]', content)
            if first_sentence and len(first_sentence.group(0)) < 150:
                title = first_sentence.group(0).strip('.')
            else:
                title = f"Section {section_number.replace('§ ', '')}"
        
        if len(content) < 50:
            return None
        
        return {
            'section_number': section_number,
            'title': title[:500],
            'content': content
        }

    def _extract_title_number(section_number: str) -> int | None:
        if not section_number:
            return None
        m = re.search(r'(\d+)-', section_number)
        return int(m.group(1)) if m else None

    # Scrapes ALL sections from a page
    def _scrape_all_sections_from_page(self, soup: BeautifulSoup, url: str) -> int:
        sections_found = 0
        
        jsx_parsers = soup.find_all('div', class_='jsx-parser')
        
        if jsx_parsers:
            logger.debug(f"Found {len(jsx_parsers)} jsx-parser divs on {url}")
            
            for jsx_parser in jsx_parsers:
                section_data = self._extract_section_data(jsx_parser)
                
                if not section_data:
                    continue
                
                section_key = section_data['section_number']
                if section_key in self.processed_sections:
                    continue
                
                section_data['url'] = url
                section_data['content_hash'] = hashlib.sha256(
                    section_data['content'].encode('utf-8')
                ).hexdigest()
                section_data['last_updated'] = datetime.utcnow()
                
                url_match = re.search(r'0-0-0-(\d+)', url)
                if url_match:
                    section_data['metadata'] = {'url_id': url_match.group(1)}
                section_data['title_number'] = self._extract_title_number(section_data['section_number'])
                if self._save_section(section_data):
                    sections_found += 1
                    self.processed_sections.add(section_key)
        
        return sections_found

    # Cleans the extracted text
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'Â\s*', '', text)
        text = re.sub(r'[\u00A0\u2002-\u200B]', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'^\s*Back to Top\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'ShareDownloadBookmarkPrint', '', text)
        return text.strip()

    # Saves a scraped section to the database
    def _save_section(self, section_data: Dict[str, Any]) -> bool:
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("SELECT id FROM sections WHERE content_hash = %s", 
                             (section_data['content_hash'],))
                if cursor.fetchone():
                    self.sections_skipped += 1
                    return False
                
                section_data['content'] = self._clean_text(section_data['content'])
                
                metadata_json = json.dumps(section_data.get('metadata', {}))
                
                cursor.execute("""
                    INSERT INTO sections
                    (section_number, title, content, url, content_hash, last_updated, metadata, title_number)
                    VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (section_number) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    url = EXCLUDED.url,
                    content_hash = EXCLUDED.content_hash,
                    last_updated = EXCLUDED.last_updated,
                    metadata = EXCLUDED.metadata,
                    title_number = EXCLUDED.title_number;
                """, (
                    section_data['section_number'],
                    section_data['title'],
                    section_data['content'],
                    section_data['url'],
                    section_data['content_hash'],
                    section_data['last_updated'],
                    metadata_json,
                    section_data['title_number'],
                ))

                self.sections_saved += 1
                
                if self.sections_saved % 10 == 0:
                    logger.info(f"Progress: {self.sections_saved} sections saved")
                    
                return True
                
        except Exception as e:
            logger.error(f"DB error for section {section_data['section_number']}: {e}")
            return False

    # Enhanced full scrape with better coverage
    def run_full_scrape(self, max_pages: Optional[int] = None):
        if not self.driver:
            return
            
        logger.info("Starting comprehensive scrape of NYC Administrative Code")
        logger.info(f"Base URL: {self.base_url}")
        
        urls_to_process = deque([self.base_url])
        
        known_titles = [
            "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-5",
            "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-108",
            "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-2753",
            "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-39081",
            "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-42627",
        ]
        
        for url in known_titles:
            if url not in self.processed_urls:
                urls_to_process.append(url)
        
        while urls_to_process and (max_pages is None or self.pages_processed < max_pages):
            url = urls_to_process.popleft()
            
            if url in self.processed_urls:
                continue
                
            soup = self._get_soup_with_selenium(url)
            if not soup:
                continue
            
            sections_found = self._scrape_all_sections_from_page(soup, url)
            if sections_found > 0:
                logger.info(f"✓ Scraped {sections_found} sections from {url}")
            
            new_links = self._extract_all_links(soup, url)
            
            added_count = 0
            for link in new_links:
                if link not in self.processed_urls and link not in urls_to_process:
                    urls_to_process.append(link)
                    added_count += 1
            
            if added_count > 0:
                logger.debug(f"Added {added_count} new URLs to queue (queue size: {len(urls_to_process)})")
            
            if self.pages_processed % 25 == 0:
                self._log_status(len(urls_to_process))
        
        self._log_status(0)
        
        if self.failed_urls:
            logger.info(f"Retrying {len(self.failed_urls)} failed URLs...")
            for url in list(self.failed_urls):
                self.failed_urls.remove(url)
                soup = self._get_soup_with_selenium(url, retry_count=1)
                if soup:
                    sections = self._scrape_all_sections_from_page(soup, url)
                    if sections > 0:
                        logger.info(f"✓ Recovered {sections} sections from {url}")
        
        if self.driver:
            self.driver.quit()
            
    # Log current scraping status        
    def _log_status(self, queue_size: int):
        logger.info(f"""
        ========================================
        SCRAPING STATUS:
        - Pages processed: {self.pages_processed}
        - Sections saved: {self.sections_saved}
        - Sections skipped: {self.sections_skipped}
        - Failed URLs: {len(self.failed_urls)}
        - Queue size: {queue_size}
        - Unique title patterns seen: {len(self.title_patterns)}
        ========================================
        """)
    
    # Verify that we have good coverage of the NYC Admin Code    
    def verify_coverage(self):
        if not self.db:
            return
            
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    SUBSTRING(section_number FROM '§ (\d+)') as title_num,
                    COUNT(*) as section_count
                FROM sections
                WHERE section_number LIKE '§ %'
                GROUP BY title_num
                ORDER BY title_num::int
            """)
            
            results = cursor.fetchall()
            
            logger.info("\nCOVERAGE REPORT:")
            logger.info("Title | Sections")
            logger.info("------|----------")
            
            expected_titles = set(range(1, 33))
            found_titles = set()
            
            for row in results:
                if row['title_num']:
                    title = int(row['title_num'])
                    found_titles.add(title)
                    logger.info(f"  {title:2d}  | {row['section_count']:4d}")
            
            missing_titles = expected_titles - found_titles
            if missing_titles:
                logger.warning(f"\nMISSING TITLES: {sorted(missing_titles)}")
                logger.warning("Consider running scraper again or targeting missing titles")
            else:
                logger.info("\n✓ All major titles appear to be present!")