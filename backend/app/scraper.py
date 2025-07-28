import re
import time
import hashlib
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config.settings import SCRAPE_CONFIG
from app.database import DatabaseManager

logger = logging.getLogger(__name__)


class NYCAdminCodeScraper:
    """Scraper for NYC Administrative Code website"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.base_url = SCRAPE_CONFIG["base_url"]
        self.delay = SCRAPE_CONFIG["delay"]
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create HTTP session with headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'NYC Admin Code Scraper (Educational/Research)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        return session
        
    def scrape_table_of_contents(self) -> List[Dict[str, str]]:
        """Scrape the main table of contents"""
        logger.info(f"Scraping table of contents from {self.base_url}")
        
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch table of contents: {e}")
            return []
            
        soup = BeautifulSoup(response.text, 'lxml')
        sections = []
        
        # Look for the actual content area
        # The site uses a specific structure - let's find the navigation tree
        nav_tree = soup.find('div', {'class': 'navChildren'}) or \
                   soup.find('ul', {'class': 'nav-tree'}) or \
                   soup.find('div', {'id': 'toc'})
        
        if nav_tree:
            # Find all links within the navigation
            for link in nav_tree.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                # Skip empty links or navigation controls
                if not text or text in ['Off', 'Follow', 'Share', 'Download', 'Print']:
                    continue
                
                # Extract section info
                if '/NYCadmin/' in href:
                    full_url = urljoin(self.base_url, href)
                    
                    # Try to extract section number from text
                    section_match = re.search(r'(Title\s+\d+|Chapter\s+\d+|§\s*[\d\-\.]+)', text)
                    if section_match:
                        sections.append({
                            'section_number': section_match.group(1),
                            'title': text,
                            'url': full_url
                        })
        else:
            # Fallback: look for any links that seem like sections
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                if '/NYCadmin/' in href and len(text) > 10:
                    # Check if it looks like a section
                    if any(keyword in text for keyword in ['Title', 'Chapter', '§', 'Section']):
                        full_url = urljoin(self.base_url, href)
                        sections.append({
                            'section_number': text[:50],  # First 50 chars as identifier
                            'title': text,
                            'url': full_url
                        })
                    
        logger.info(f"Found {len(sections)} sections")
        return sections[:100]  # Limit for testing
        
    def scrape_section(self, section_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Scrape individual section content"""
        section_number = section_info['section_number']
        url = section_info['url']
        
        logger.info(f"Scraping section {section_number}")
        time.sleep(self.delay)  # Rate limiting
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch section {section_number}: {e}")
            return None
            
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Remove navigation and UI elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
            
        # Remove specific UI elements by class/id
        ui_selectors = [
            {'class': 'nav-tree'},
            {'class': 'toolbar'},
            {'class': 'breadcrumb'},
            {'id': 'header'},
            {'id': 'footer'},
            {'class': 'disclaimer'},
            {'class': 'annotations-toggle'},
        ]
        
        for selector in ui_selectors:
            for element in soup.find_all(attrs=selector):
                element.decompose()
        
        # Find the main content area
        content = None
        
        # Try different content selectors
        content_selectors = [
            {'class': 'content'},
            {'class': 'section-content'},
            {'class': 'code-content'},
            {'id': 'content'},
            {'id': 'main-content'},
            {'role': 'main'},
            {'class': 'ordinance-content'},
        ]
        
        for selector in content_selectors:
            content_elem = soup.find('div', selector) or soup.find('main', selector)
            if content_elem:
                content = content_elem
                break
                
        # If no specific content area found, try to get the body
        if not content:
            content = soup.find('body')
            
        if not content:
            logger.warning(f"No content found for section {section_number}")
            return None
            
        # Extract text, removing extra whitespace
        text_lines = []
        for element in content.find_all(['p', 'div', 'section', 'article']):
            text = element.get_text(strip=True)
            # Skip UI text
            if text and not any(skip in text for skip in ['Annotations', 'Off', 'Follow', 'Share', 'Download', 'Bookmark', 'Print', 'Disclaimer:']):
                text_lines.append(text)
                
        content_text = '\n\n'.join(text_lines)
        
        # If content is too short or seems wrong, log it
        if len(content_text) < 100:
            logger.warning(f"Section {section_number} has very short content: {len(content_text)} chars")
            # Try alternative extraction
            content_text = soup.get_text(separator='\n', strip=True)
            # Remove known UI text
            ui_patterns = [
                r'Annotations\s+Off\s+Follow\s+Changes\s+Share\s+Download\s+Bookmark\s+Print',
                r'Disclaimer:.*?adopted by the City\.',
                r'NY\s+New York City\s+The New York City Administrative Code',
            ]
            for pattern in ui_patterns:
                content_text = re.sub(pattern, '', content_text, flags=re.DOTALL)
                
        # Extract cross-references
        cross_refs = self._extract_cross_references(content_text)
        
        # Generate content hash for change detection
        content_hash = hashlib.sha256(content_text.encode()).hexdigest()
        
        return {
            'section_number': section_number,
            'title': section_info['title'],
            'content': content_text.strip(),
            'url': url,
            'cross_references': cross_refs,
            'content_hash': content_hash
        }
        
    def _extract_cross_references(self, content: str) -> List[str]:
        """Extract references to other sections"""
        # Pattern to match section references
        patterns = [
            r'(?:§|section)\s*([\d\-\.]+)',
            r'(?:Title|Chapter)\s+(\d+)',
            r'sections?\s+([\d\-\.]+)\s+(?:through|to)\s+([\d\-\.]+)',
        ]
        
        references = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    references.update(match)
                else:
                    references.add(match)
                    
        return list(references)
        
    def save_section(self, section_data: Dict[str, Any]) -> Optional[int]:
        """Save section to database"""
        # Only save if we have meaningful content
        if len(section_data['content']) < 50:
            logger.warning(f"Skipping section {section_data['section_number']} - content too short")
            return None
            
        try:
            with self.db.get_cursor() as cursor:
                # Check if section exists
                cursor.execute("""
                    SELECT id, content_hash FROM sections 
                    WHERE section_number = %s
                """, (section_data['section_number'],))
                
                existing = cursor.fetchone()
                
                if existing and existing['content_hash'] == section_data['content_hash']:
                    logger.info(f"Section {section_data['section_number']} unchanged")
                    return existing['id']
                    
                # Insert or update section
                cursor.execute("""
                    INSERT INTO sections (section_number, title, content, url, content_hash)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (section_number) 
                    DO UPDATE SET 
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        url = EXCLUDED.url,
                        content_hash = EXCLUDED.content_hash,
                        last_updated = CURRENT_TIMESTAMP
                    RETURNING id
                """, (
                    section_data['section_number'],
                    section_data['title'],
                    section_data['content'],
                    section_data['url'],
                    section_data['content_hash']
                ))
                
                section_id = cursor.fetchone()['id']
                
                # Save cross-references
                for ref in section_data['cross_references']:
                    cursor.execute("""
                        INSERT INTO cross_references 
                        (from_section_id, to_section_number, reference_text)
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (section_id, ref, f"References {ref}"))
                    
                logger.info(f"Saved section {section_data['section_number']}")
                return section_id
                
        except Exception as e:
            logger.error(f"Error saving section: {e}")
            return None
            
    def run_full_scrape(self):
        """Run complete scraping process"""
        logger.info("Starting full scrape")
        
        # Record scrape start
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO scrape_history (started_at) 
                VALUES (CURRENT_TIMESTAMP) 
                RETURNING id
            """)
            scrape_id = cursor.fetchone()['id']
            
        sections = self.scrape_table_of_contents()
        
        processed = 0
        updated = 0
        errors = []
        
        for section_info in sections:
            try:
                section_data = self.scrape_section(section_info)
                if section_data:
                    section_id = self.save_section(section_data)
                    if section_id:
                        processed += 1
                        # Check if it was an update
                        if section_data.get('updated'):
                            updated += 1
            except Exception as e:
                error_msg = f"Error processing {section_info['section_number']}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                
        # Update scrape history
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                UPDATE scrape_history 
                SET completed_at = CURRENT_TIMESTAMP,
                    sections_processed = %s,
                    sections_updated = %s,
                    errors = %s::jsonb
                WHERE id = %s
            """, (processed, updated, errors, scrape_id))
            
        logger.info(f"Scrape completed: {processed} processed, {updated} updated, {len(errors)} errors")


# Test scraper function
def test_scraper():
    """Test the scraper with a single section"""
    db = DatabaseManager()
    db.connect()
    
    scraper = NYCAdminCodeScraper(db)
    
    # Test with a specific URL
    test_section = {
        'section_number': 'Test',
        'title': 'Test Section',
        'url': 'https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-6'
    }
    
    result = scraper.scrape_section(test_section)
    if result:
        print(f"Content length: {len(result['content'])}")
        print(f"First 500 chars: {result['content'][:500]}")
    
    db.close()


if __name__ == "__main__":
    test_scraper()