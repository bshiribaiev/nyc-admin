import re
import time
import hashlib
import logging
from typing import List, Dict, Optional
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
        
        # Find all section links (adjust selectors based on actual HTML structure)
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            # Look for section patterns
            if '/NYCadmin/' in href and re.search(r'§?\s*\d+', text):
                full_url = urljoin(self.base_url, href)
                
                # Extract section number
                section_match = re.search(r'(§?\s*[\d\-\.]+)', text)
                if section_match:
                    section_number = section_match.group(1).strip()
                    
                    sections.append({
                        'section_number': section_number,
                        'title': text,
                        'url': full_url
                    })
                    
        logger.info(f"Found {len(sections)} sections")
        return sections
        
    def scrape_section(self, section_info: Dict[str, str]) -> Optional[Dict[str]]:
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
        
        # Extract content (adjust based on actual HTML structure)
        content_elem = soup.find('div', class_='content') or \
                      soup.find('div', id='codecontent') or \
                      soup.find('main')
                      
        if not content_elem:
            logger.warning(f"No content found for section {section_number}")
            return None
            
        content = content_elem.get_text(separator='\n', strip=True)
        
        # Extract cross-references
        cross_refs = self._extract_cross_references(content)
        
        # Generate content hash for change detection
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        return {
            'section_number': section_number,
            'title': section_info['title'],
            'content': content,
            'url': url,
            'cross_references': cross_refs,
            'content_hash': content_hash
        }
        
    def _extract_cross_references(self, content: str) -> List[str]:
        """Extract references to other sections"""
        # Pattern to match section references
        patterns = [
            r'(?:§|section)\s*([\d\-\.]+)',
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
        
    def save_section(self, section_data: Dict[str]) -> Optional[int]:
        """Save section to database"""
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