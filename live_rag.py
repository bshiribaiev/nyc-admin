import os
import re
import httpx
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit("Set GEMINI_API_KEY in your env (e.g., export GEMINI_API_KEY=...)")

genai.configure(api_key=GEMINI_API_KEY)

# Known “good” URLs for permit topics (no DB, just pragmatic anchors)
PERMIT_URLS = [
    # Effective date + prior-permit rule
    "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-155416",  # § 28-101.4 Effective date
    "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-155420",  # § 28-101.4.1 Permit issued or work commenced prior to July 1, 2008

    # General permit rule + related permit sections
    "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-156206",  # § 28-105.1 General
    "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-156394",  # § 28-105.8.1 Duration of permit
    "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-156417",  # § 28-105.11 Posting
]


# Fallback when we don’t detect a special intent
TOC_URL = "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-1"

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

def pick_urls_for_query(q: str):
    ql = q.lower()
    if any(w in ql for w in ["permit", "permits", "construction"]):
        return PERMIT_URLS
    return [TOC_URL]

def fetch_url(url: str) -> str:
    with httpx.Client(timeout=30.0, headers=UA, follow_redirects=True) as cx:
        r = cx.get(url)
        r.raise_for_status()
        return r.text

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Try to keep just the main content
    main = soup.find(attrs={"role": "main"}) or soup.find("main") or soup
    text = main.get_text(" ", strip=True)
    # light normalize
    text = re.sub(r"\s+", " ", text)
    return text

def ask_gemini(question: str, context_text: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")  # fast
    prompt = f"""
You are an expert on the NYC Administrative Code.
Answer ONLY using the following text.
After EVERY sentence, include the section number (e.g., (§ 28-105.1)) and the exact source URL.
If the answer does not appear in the text, reply exactly "NO_ANSWER".

Question: {question}

Context:
{context_text}
"""
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

if __name__ == "__main__":
    q = input("Ask: ").strip()
    urls = pick_urls_for_query(q)
    combined_text = ""
    for url in urls:
        try:
            html = fetch_url(url)
            txt = html_to_text(html)
            # Tag the context with its URL so the model can cite it
            combined_text += f"\n\n[URL]: {url}\n{txt}\n"
        except Exception as e:
            print(f"Warning: failed to fetch {url}: {e}")

    # Keep context under ~60–80k chars if you like; here we keep it simple
    answer = ask_gemini(q, combined_text[:100000])
    print("\n=== ANSWER ===\n")
    print(answer)
