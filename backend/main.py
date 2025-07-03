import requests
from bs4 import BeautifulSoup
import re
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SearchResult:
    section_id: str
    title: str
    content: str
    url: str
    relevance_score: float

class RealTimeNYCAdminTool:
    """
    A tool that searches NYC admin code in real-time
    Goes to the website, finds relevant sections, and answers questions
    """
    
    def __init__(self):
        # NYC admin code search endpoints
        self.search_base = "https://codelibrary.amlegal.com"
        self.nyc_code_base = "/codes/newyorkcity/latest/NYCadmin"
        
        # Common headers to look like a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search_sections(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search for sections related to the query
        This simulates finding relevant sections
        """
        print(f"🔍 Searching NYC admin code for: '{query}'")
        
        # For now, let's use a mapping of common topics to known sections
        # In a real implementation, you'd scrape the site's search or table of contents
        topic_to_sections = {
            'building': ['2-01', '2-02', '28-101', '28-102'],
            'permit': ['2-01', '28-101', '28-105'],
            'fire': ['3-01', '3-02', 'FC-401'],
            'health': ['4-01', '17-101', '17-102'],
            'parking': ['14-101', '14-102', '19-101'],
            'restaurant': ['17-301', '17-302', '20-101'],
            'business': ['20-101', '20-102', '22-101'],
            'zoning': ['11-101', '11-102', '25-101'],
            'construction': ['28-101', '28-102', '28-201'],
            'noise': ['24-218', '24-219', '24-220']
        }
        
        # Find relevant sections based on keywords
        relevant_sections = []
        query_lower = query.lower()
        
        for topic, sections in topic_to_sections.items():
            if topic in query_lower:
                relevant_sections.extend(sections)
        
        # If no specific match, try general building/permit sections
        if not relevant_sections:
            relevant_sections = ['28-101', '20-101', '2-01']
        
        # Remove duplicates and limit results
        relevant_sections = list(set(relevant_sections))[:max_results]
        
        print(f"📋 Found {len(relevant_sections)} potentially relevant sections")
        
        # Fetch content for each section
        search_results = []
        for section_id in relevant_sections:
            result = self.fetch_section_content(section_id, query)
            if result:
                search_results.append(result)
        
        return search_results
    
    def fetch_section_content(self, section_id: str, query: str) -> Optional[SearchResult]:
        """
        Fetch and parse content from a specific section
        """
        # Try different URL patterns for NYC admin code
        possible_urls = [
            f"{self.search_base}{self.nyc_code_base}/{section_id}",
            f"{self.search_base}{self.nyc_code_base}/0-0-0-{section_id}",
            f"https://library.amlegal.com/nxt/gateway.dll/New%20York/admin/newyorkcityadministrativecode?f=templates$fn=default.htm$3.0$vid=amlegal:newyork_ny$anc={section_id}"
        ]
        
        for url in possible_urls:
            try:
                print(f"  📄 Fetching section {section_id}...")
                
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    content = self.parse_legal_content(response.text)
                    if content and len(content) > 100:  # Make sure we got real content
                        
                        # Calculate relevance score (simple keyword matching)
                        relevance = self.calculate_relevance(content, query)
                        
                        return SearchResult(
                            section_id=section_id,
                            title=self.extract_title(response.text, section_id),
                            content=content,
                            url=url,
                            relevance_score=relevance
                        )
                
                # Small delay to be respectful to the server
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ Error fetching {section_id} from {url}: {e}")
                continue
        
        print(f"    ⚠️  Could not fetch section {section_id}")
        return None
    
    def parse_legal_content(self, html: str) -> str:
        """
        Extract the main legal text from HTML
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove scripts, styles, and navigation
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Try to find the main content area
        content_selectors = [
            'div.content',
            'div.main-content', 
            'div.document-content',
            'main',
            'article',
            'div#content',
            'div.legal-content'
        ]
        
        content_text = ""
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                content_text = element.get_text(separator=' ', strip=True)
                break
        
        # If no specific content found, get all text but filter out navigation
        if not content_text:
            content_text = soup.get_text(separator=' ', strip=True)
        
        # Clean up the text
        content_text = re.sub(r'\s+', ' ', content_text)  # Multiple spaces to single
        content_text = re.sub(r'\n+', '\n', content_text)  # Multiple newlines to single
        
        # Remove common navigation text
        unwanted_phrases = [
            'Skip to main content',
            'Print this page',
            'Email this page',
            'Bookmark this page',
            'Table of Contents'
        ]
        
        for phrase in unwanted_phrases:
            content_text = content_text.replace(phrase, '')
        
        return content_text.strip()
    
    def extract_title(self, html: str, section_id: str) -> str:
        """
        Extract the section title from HTML
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try different title patterns
        title_selectors = ['h1', 'h2', '.title', '.section-title', 'title']
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if len(title) > 5 and len(title) < 200:  # Reasonable title length
                    return title
        
        return f"Section {section_id}"
    
    def calculate_relevance(self, content: str, query: str) -> float:
        """
        Calculate how relevant the content is to the query
        Simple keyword-based scoring
        """
        content_lower = content.lower()
        query_words = query.lower().split()
        
        score = 0
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                # Count occurrences, with diminishing returns
                count = content_lower.count(word)
                score += min(count * 0.1, 1.0)  # Max 1.0 per word
        
        # Normalize by query length
        return min(score / len(query_words), 1.0)
    
    def generate_answer(self, query: str, search_results: List[SearchResult]) -> Dict:
        """
        Generate an answer based on the search results
        """
        if not search_results:
            return {
                "answer": "I couldn't find relevant information in the NYC administrative code for your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Sort by relevance
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Create answer from most relevant sections
        answer_parts = []
        sources = []
        
        for result in search_results[:3]:  # Use top 3 results
            # Extract most relevant sentences
            sentences = result.content.split('.')
            relevant_sentences = []
            
            query_words = query.lower().split()
            for sentence in sentences:
                sentence_lower = sentence.lower()
                word_matches = sum(1 for word in query_words if word in sentence_lower)
                if word_matches >= 1 and len(sentence.strip()) > 20:
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                # Take the best sentences
                best_sentences = relevant_sentences[:2]
                section_summary = '. '.join(best_sentences)
                answer_parts.append(f"According to Section {result.section_id}: {section_summary}")
                
                sources.append({
                    "section_id": result.section_id,
                    "title": result.title,
                    "url": result.url,
                    "relevance": result.relevance_score
                })
        
        if answer_parts:
            answer = '\n\n'.join(answer_parts)
            confidence = sum(r.relevance_score for r in search_results[:3]) / 3
        else:
            answer = "I found relevant sections but couldn't extract specific information that directly answers your question. Please check the source sections for detailed information."
            confidence = 0.3
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }
    
    def ask_question(self, question: str) -> Dict:
        """
        Main method: Ask a question and get an answer from NYC admin code
        """
        print(f"\n🤔 Question: {question}")
        print("=" * 50)
        
        # Step 1: Search for relevant sections
        search_results = self.search_sections(question)
        
        # Step 2: Generate answer
        result = self.generate_answer(question, search_results)
        
        print(f"\n✅ Found answer with {len(result['sources'])} sources")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        
        return result

# Example usage and testing
def demo_real_time_tool():
    """
    Demonstrate the real-time tool
    """
    print("🏛️  NYC Admin Code Real-Time Tool Demo")
    print("=" * 50)
    
    # Create the tool
    tool = RealTimeNYCAdminTool()
    
    # Test questions
    test_questions = [
        "What are the requirements for a building permit?",
        "What are the fire safety regulations for restaurants?",
        "What are the parking violation penalties?",
        "What licenses do I need to start a business?"
    ]
    
    for question in test_questions:
        try:
            result = tool.ask_question(question)
            
            print(f"\n💬 Question: {question}")
            print(f"📝 Answer: {result['answer']}")
            print(f"📚 Sources: {[s['section_id'] for s in result['sources']]}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
        
        # Pause between questions to be respectful to the server
        time.sleep(2)

if __name__ == "__main__":
    # For testing individual questions
    tool = RealTimeNYCAdminTool()
    
    # Ask a single question
    question = input("Ask a question about NYC admin code: ")
    result = tool.ask_question(question)
    
    print(f"\n📝 Answer: {result['answer']}")
    print(f"\n📚 Sources used:")
    for source in result['sources']:
        print(f"  - Section {source['section_id']}: {source['title']}")
        print(f"    URL: {source['url']}")