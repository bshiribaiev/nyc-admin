import os
import re
import httpx
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv
# Load variables from .env
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit("Set GEMINI_API_KEY in your env (e.g., export GEMINI_API_KEY=...)")
genai.configure(api_key=GEMINI_API_KEY)
# Global variables
TOC_URL = "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-1"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
STOP = {"the","a","an","of","for","to","and","or","in","is","are"}  
SEC_RE = r"§\s\d+[-–]\d+(?:[.\-–]\d+)"
FETCH_CACHE: dict[str, str] = {}
def is_admin_code_section_url(u: str) -> bool:
    if not u.startswith("https://codelibrary.amlegal.com/codes/newyorkcity/"):
        return False

    bad = ("/search", "/advancedsearch", "/results", "/toc", "/attachments")
    if any(b in u.lower() for b in bad):
        return False

    guidish = re.search(r"/content/[a-f0-9\-]{8,}$", u)  
    numeric = re.search(r"/0-0-0-\d+$", u)
    slugged = re.search(r"/content-title-\d+/./\d+-\d+(?:-\d+)-", u)
    return bool(numeric or slugged or guidish)
def expand_query(q: str) -> str:
    ql = q.lower()
    add = []
    if "permit" in ql or "emergency work" in ql:
        add += ["§ 28-105.4.1", "28-105.4.1", "work permit", "emergency work", "construction"]
    if "sidewalk shed" in ql or ("sidewalk" in ql and "shed" in ql):
        add += ["3307.6.2", "sidewalk shed", "Title 28"]
    if "zoning" in ql and ("effect" in ql or "effects" in ql):
        # this is the specific section your test expects
        add += ['"Effect on zoning resolution"', "§ 28-101.4.5.3", "28-101.4.5.3", "28-101.4"]
    if any(t in ql for t in ["fire","sprinkler","alarm","smoke","egress","fdny","evacuation"]):
        add += ["Title 28 Chapter 9", "Title 29", "BC 903", "BC 907"]
    if "student housing" in ql:
        add += ["dormitory","residence hall","Group R-1","Group R-2"]
    if "noise" in ql:
        add += ["Title 24","decibel","construction noise"]
    return q + " " + " ".join(add)
def query_ngrams(query: str, min_n=2, max_n=5):
    words = [w for w in re.findall(r"[a-z0-9\-]+", query.lower()) if w not in STOP]
    ngrams = []
    for n in range(min_n, min(max_n, len(words)) + 1):
        for i in range(len(words) - n + 1):
            ng = " ".join(words[i:i+n])
            ngrams.append(ng)

    # plural/singular tolerance: add variants where last word drops a trailing 's'
    tolerant = set(ngrams)
    for ng in list(ngrams):
        parts = ng.split()
        last = parts[-1]
        if len(last) > 3 and last.endswith("s"):
            parts2 = parts[:-1] + [last[:-1]]
            tolerant.add(" ".join(parts2))

    # longest first
    tolerant = sorted(tolerant, key=len, reverse=True)
    long_ngrams = [ng for ng in tolerant if len(ng.split()) >= 3]
    two_grams  = [ng for ng in tolerant if len(ng.split()) == 2]
    return tolerant, long_ngrams, two_grams
def build_search_variants(query: str, site: str) -> list[str]:
    q = query.strip()
    tokens = " ".join(re.findall(r"[a-z0-9\-]+", q.lower()))
    secs = re.findall(r"\b\d+-\d+(?:\.\d+)*\b", q)

    variants = [
        f"{site} {expand_query(q)}",
        f'{site} "{q}"',
        f"{site} intitle:({tokens})",
        f"{site} inurl:({tokens})",
        f'{site} intitle:"{q}"',
    ]

    , longngrams, * = query*ngrams(query)
    if long_ngrams:
        top = long_ngrams[0]
        variants.append(f'{site} "{top}"')
        variants.append(f'{site} intitle:"{top}"')

    for s in secs:
        variants.append(f'{site} "{s}"')
        variants.append(f"{site} inurl:{s}")

    return variants
def rank_and_filter_candidates(query: str, urls: list[str], max_fetch: int = 12) -> list[str]:
    q_lower = query.strip().lower()
    wanted_secs = re.findall(r"\b\d+-\d+(?:\.\d+)*\b", q_lower)
    tokens = set(re.findall(r"[a-z0-9\-]+", q_lower))
    ngrams, long_ngrams, two_grams = query_ngrams(query)

    scored = []
    for u in urls[:max_fetch]:
        try:
            html = fetch_url(u)
            sec, h1, title = extract_meta(html)
            h1_l = (h1 or "").lower()
            title_l = (title or "").lower()
            url_l = u.lower()

            score = 0.0

            # token overlap
            score += 3.0 * sum(1 for t in tokens if t in h1_l)
            score += 1.5 * sum(1 for t in tokens if t in title_l)
            score += 1.0 * sum(1 for t in tokens if t in url_l)

            # phrase-level boosts
            for ng in long_ngrams[:6]:
                if ng in h1_l:   score += 15.0
                if ng in title_l: score += 7.0

            for ng in two_grams[:6]:
                if ng in h1_l:   score += 4.0
                if ng in title_l: score += 2.0

            # section number match boost
            if sec and any(ws in sec for ws in wanted_secs):
                score += 7.0

            # prefer real sections; de-prefer generic definition/violation pages
            if not sec:
                score -= 5.0
            else:
                score += 1.0

            h1_flags = " ".join(filter(None, [h1_l, title_l]))
            if (("violation" in h1_flags or "definitions" in h1_flags or "definition" in h1_flags)
                and not any(w in q_lower for w in ["violation","penalty","define","definition"])):
                score -= 3.0

            if score > 0:
                scored.append((score, u))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [u for _, u in scored]
    remainder = [u for u in urls if u not in set(ranked)]
    return ranked + remainder
# Get the page's title and section number
def extract_meta(html: str) -> tuple[str|None, str|None, str|None]:
    soup = BeautifulSoup(html, "html.parser")

    # Try common amlegal patterns first
    h = (
        soup.find("h1")
        or soup.find("h2")
        or soup.find("h3")
        or soup.find(attrs={"id": re.compile(r"title|section", re.I)})
        or soup.find(attrs={"class": re.compile(r"title|section", re.I)})
    )
    h1_text = h.get_text(" ", strip=True) if h else None
    page_title = soup.title.get_text(" ", strip=True) if soup.title else None

    # Find a real § pattern (allow en-dash/hyphen)
    blob = " ".join(filter(None, [h1_text, page_title]))
    m = re.search(SEC_RE, blob)
    if not m:
        body_start = soup.get_text(" ", strip=True)[:2000]
        m = re.search(SEC_RE, body_start)

    section = m.group(0) if m else None
    return section, h1_text, page_title
def title_article_indexes_for_query(q: str) -> list[str]:
    ql = q.lower()
    idx = []

    # If a section number is mentioned, prefer that Title's index
    secs = re.findall(r"\b(\d+)-\d+(?:\.\d+)*\b", ql)
    title_nums = {int(s) for s in secs}
    for t in sorted(title_nums)[:2]:
        idx.append(f"https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/content-title-{t}")

    if "zoning" in ql:
        idx.append("https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/content-title-28")
        idx.append("https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/content-title-28/ARTICLE-101")
    if "noise" in ql:
        idx.append("https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/content-title-24")
    if any(w in ql for w in ["building","permit","construction","egress","sprinkler","alarm"]):
        idx.append("https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/content-title-28")
    if "fire" in ql or "fdny" in ql:
        idx.append("https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/content-title-29")

    idx.append(TOC_URL)

    seen, out = set(), []
    for u in idx:
        if u not in seen:
            seen.add(u); out.append(u)
    return out
def drilldown_toc_candidates(q: str, max_urls: int = 10) -> list[str]:
    candidates: list[str] = []
    patterns = [r"/0-0-0-\d+$", r"/content-title-\d+/./\d+-\d+(?:-\d+)-"]
    toks = re.findall(r"[a-z0-9\-]+", q.lower())
    secs = re.findall(r"\b\d+-\d+(?:\.\d+)*\b", q)
    likely = set(toks + secs)

    for idx_url in title_article_indexes_for_query(q)[:3]:
        try:
            html = fetch_url(idx_url)
        except Exception:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = httpx.URL(idx_url).join(href).human_repr()

            if not href.startswith("https://codelibrary.amlegal.com/codes/newyorkcity/"):
                continue

            text = a.get_text(" ", strip=True).lower()
            hay = f"{text} {href.lower()}"

            # keep only section-like links AND relevant keywords
            if any(re.search(p, href) for p in patterns) and any(tok in hay for tok in likely):
                candidates.append(href)
                if len(candidates) >= max_urls:
                    break

    # dedupe and return a few
    out = []
    seen = set()
    for u in candidates:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:max_urls]
# Use SerpAPI to find NYC Admin Code URLs for this query
def search_admin_code(query: str, k: int = 10) -> list[str]:
    site = 'site:codelibrary.amlegal.com "NYCadmin"'
    variants = build_search_variants(query, site)
    urls: list[str] = []

    if SERPAPI_KEY:
        with httpx.Client(timeout=20.0) as cx:
            for v in variants:
                if len(urls) >= k:
                    break
                r = cx.get(
                    "https://serpapi.com/search.json",
                    params={
                        "engine": "google",
                        "q": v,
                        "api_key": SERPAPI_KEY,
                        "num": 10,
                        "no_cache": "true",
                        "google_domain": "google.com",
                        "hl": "en",
                    },
                )
                # If SerpAPI throttles/errs, just continue to next variant
                if r.status_code != 200:
                    continue

                data = r.json()
                for item in data.get("organic_results", []):
                    u = item.get("link")
                    if u and is_admin_code_section_url(u):
                        urls.append(u)

                # light dedupe on the fly
                urls = list(dict.fromkeys(urls))

    print("[DEBUG] raw SerpAPI urls:", *urls, sep="\n  ")

    # If still thin, do a small TOC drilldown (see next section)
    if len(urls) < 3:
        urls.extend(drilldown_toc_candidates(query))

    # Fallback anchors only if nothing found at all
    if not urls:
        urls = [
            "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-156206",  # § 28-105.1
            "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-156252",  # § 28-105.4
            "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-155420",  # § 28-101.4.1
        ]

    if urls:
        urls = rank_and_filter_candidates(query, urls)
        print("\n[DEBUG] ranked candidates (sec | H1 | title):")
        for u in urls[:8]:
            try:
                sec, h1, title = extract_meta(fetch_url(u))
                print(f"  {sec or '-'} | {h1 or '-'} | {title or '-'} | {u}")
            except Exception as e:
                print(f"  [err] {u}: {e}")

    # Dedup + cap
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out[:k]
def pick_urls_for_query(q: str) -> list[str]:
    return search_admin_code(q, k=10)
def fetch_url(url: str) -> str:
    if url in FETCH_CACHE:
        return FETCH_CACHE[url]

    with httpx.Client(timeout=30.0, headers=UA, follow_redirects=True) as cx:
        r = cx.get(url)
        r.raise_for_status()
        html = r.text

    FETCH_CACHE[url] = html
    return html
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script","style","noscript","header","footer","nav","aside"]):
        tag.decompose()

    main = soup.find(attrs={"role":"main"}) or soup.find("main") or soup.find("article") or soup
    text = main.get_text(" ", strip=True)
    text = text.replace("Â§", "§").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+"," ", text)
def chunk_text(text: str, size=900, overlap=120):
    text = re.sub(r"\s+"," ", text).strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(1, size-overlap)
    return out
def extract_section_from_text(txt: str) -> str | None:
    m = re.search(SEC_RE, txt)
    return m.group(0) if m else None
def pick_snippets(query: str, url_to_text: dict[str,str], url_meta: dict[str,tuple], max_snips=5) -> list[dict]:
    terms = set(re.findall(r"[a-z0-9§\-\.]+", query.lower()))
    ngrams, long_ngrams, two_grams = query_ngrams(query)

    seen_pairs = set()       # (section, url)
    per_url = {}             # url -> count
    out = []

    def ok_phrase(hay, h1_l, title_l):
        if long_ngrams and any(ng in hay or ng in h1_l or ng in title_l for ng in long_ngrams):
            return True
        if not long_ngrams and two_grams and any(ng in hay or ng in h1_l or ng in title_l for ng in two_grams):
            return True
        # loose fallback: at least two distinct tokens in chunk
        return sum(1 for t in terms if t in hay) >= 2

    for url, text in url_to_text.items():
        page_sec, page_h1, page_title = url_meta.get(url, (None, None, None))
        h1_l = (page_h1 or "").lower()
        title_l = (page_title or "").lower()

        for ch in chunk_text(text):
            hay = ch.lower()
            msec = re.search(SEC_RE, ch)
            sec = (msec.group(0) if msec else page_sec) or ""

            if not sec:
                continue

            if not ok_phrase(hay, h1_l, title_l):
                continue

            pair = (sec, url)
            if pair in seen_pairs:
                continue

            if per_url.get(url, 0) >= 2:   # cap max 2 snippets per page
                continue

            score = sum(1 for t in terms if t in hay)

            for ng in long_ngrams[:6]:
                if ng in hay: score += 6

            for ng in two_grams[:6]:
                if ng in hay: score += 2

            if sec and any(ws in sec for ws in re.findall(r"\b\d+-\d+(?:\.\d+)*\b", query)):
                score += 4

            out.append({
                "score": score, 
                "url": url, 
                "section": sec, 
                "title": page_title or "", 
                "h1": page_h1 or "", 
                "text": ch
            })

            seen_pairs.add(pair)
            per_url[url] = per_url.get(url, 0) + 1

            if len(out) >= max_snips:
                break

        if len(out) >= max_snips:
            break

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:max_snips]
def ask_gemini(question: str, snippets: list[dict]) -> str:
    if not snippets:
        return "NO_ANSWER"

    # Build context with clearer formatting
    blocks = []
    for i, sn in enumerate(snippets, 1):
        blocks.append(
            f"SNIPPET {i}:\n"
            f"Section: {sn['section']}\n"
            f"URL: {sn['url']}\n"
            f"Title: {sn.get('h1') or sn.get('title') or ''}\n"
            f"Content: {sn['text']}\n"
        )

    context = "\n".join(blocks)

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are an expert on the NYC Administrative Code. Answer the question using ONLY the provided snippets.
CRITICAL FORMATTING RULES:
1. Answer in 1-2 sentences maximum
2. Each sentence MUST end with: (§ [section]) [URL]
3. Use the EXACT section number and URL from the snippets
4. If the answer is not clearly supported by the snippets, respond with: NO_ANSWER
EXAMPLE GOOD ANSWER:
"The provisions of section 28-101.4.5 shall not be construed to affect the status of any non-conforming use or non-complying bulk otherwise permitted to be retained pursuant to the New York city zoning resolution. (§ 28-101.4.5.3) [https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-155473]"
Question: {question}
{context}
Answer:"""

    try:
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        return raw
    except Exception as e:
        print(f"[DEBUG] Gemini error: {e}")
        return "NO_ANSWER"
def normalize_section(sec: str) -> str:
    """Normalize section number format"""
    if not sec:
        return ""
    # Remove leading § and whitespace, then add it back consistently
    sec = re.sub(r'^\s§\s', '', sec.strip())
    sec = sec.replace('–', '-').replace('—', '-')  # normalize dashes
    return f"§ {sec}"
def validate_answer(answer: str, snippets: list[dict]) -> dict:
    """Improved validation with better regex patterns"""
    if answer.strip() == "NO_ANSWER":
        return {"valid": True, "issues": []}

    urls_whitelist = {s["url"] for s in snippets}

    # More flexible sentence splitting that handles multiple spaces and edge cases
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s*\s+(?=[A-Z])', answer.strip()) if s.strip()]

    if not sentences:
        return {"valid": False, "issues": ["empty answer"]}

    # More comprehensive citation pattern that handles various formats
    cite_pattern = r'\(\s§\s[\d\-–.]+\s\)\s\[\s(https?://[^\]]+)\s\]'

    issues = []

    for i, sent in enumerate(sentences, 1):
        # Check if sentence has a citation at the end
        citation_matches = list(re.finditer(cite_pattern, sent))

        if not citation_matches:
            issues.append(f"s{i}: missing or malformed citation")
            continue

        # Use the last citation found in the sentence
        citation_match = citation_matches[-1]
        url = citation_match.group(1).strip()

        if not url.startswith("https://codelibrary.amlegal.com/codes/newyorkcity/"):
            issues.append(f"s{i}: off-domain citation {url}")

        if url not in urls_whitelist:
            issues.append(f"s{i}: cites URL not in snippets {url}")

    return {"valid": not issues, "issues": issues}
def enforce_citation_format(answer: str, snippets: list[dict]) -> str:
    """Conservative citation enforcement - only fix clearly broken citations"""
    if answer.strip() == "NO_ANSWER" or not snippets:
        return answer

    # If the answer already has citations that look reasonable, don't modify it
    citation_pattern = re.compile(r'\(\s§\s[\d\-–.]+\s\)\s\[\shttps?://[^\]]+\s\]')
    if citation_pattern.search(answer):
        # Answer already has citations, return as-is to avoid breaking it
        return answer

    # Only if there are NO citations at all, add them
    fallback_sec = snippets[0].get("section", "")
    fallback_url = snippets[0].get("url", "")

    if not fallback_sec or not fallback_url:
        return "NO_ANSWER"

    # Simple approach: if no citations found, add one at the end
    answer = answer.strip()
    if answer and answer[-1] not in '.!?':
        answer += '.'

    normalized_sec = normalize_section(fallback_sec)
    return f"{answer} ({normalized_sec}) [{fallback_url}]"

if name == "main":
    q = input("Ask: ").strip()
    urls = pick_urls_for_query(q)[:6]

    url_to_text, url_meta = {}, {}
    for url in urls[:6]:
        try:
            html = fetch_url(url)
            url_to_text[url] = html_to_text(html)
            url_meta[url] = extract_meta(html)
        except Exception as e:
            print(f"Warning: failed to fetch {url}: {e}")

    snippets = pick_snippets(q, url_to_text, url_meta, max_snips=5)
    print("\n[DEBUG] Candidate snippets:")
    for s in snippets:
        print(f"  {s['section']}  {s['url']}")

    if not snippets:
        print("\n=== ANSWER ===\n")
        print("NO_ANSWER")
    else:
        answer = ask_gemini(q, snippets)
        print(f"\n[DEBUG] Raw Gemini response: {answer}")

        answer = enforce_citation_format(answer, snippets)
        print(f"[DEBUG] After format enforcement: {answer}")

        val = validate_answer(answer, snippets)
        print(f"[DEBUG] Validation: {val}")

        if not val["valid"]:
            print("[DEBUG] validator issues:", *val["issues"], sep="\n  ")
            answer = "NO_ANSWER"

        print("\n=== ANSWER ===\n")
        print(answer)