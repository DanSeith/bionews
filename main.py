import asyncio
import json
import os
import random
import time
from datetime import datetime, timedelta
import arxiv
import feedparser
import aiohttp
from jinja2 import Environment, FileSystemLoader
from google import genai
from google.genai import types
import pytz
from dateutil import parser
from dotenv import load_dotenv

# Load local .env if it exists
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

ARXIV_QUERIES = [
    "ti:\"protein design\"",
    "ti:\"molecular generation\"",
    "ti:\"cheminformatics\"",
    "ti:\"alphafold\"",
    "ti:\"diffusion model\" AND (cat:q-bio.BM OR cat:cs.LG)",
    "ti:\"drug discovery\" AND cat:cs.LG",
    "ti:\"boltz\" AND cat:q-bio.BM" # Specific check for Boltz-1 types
]

ARXIV_CATEGORIES = ["q-bio.BM", "cs.LG", "q-bio.QM"]

RSS_FEEDS = {
    "FierceBiotech": "https://www.fiercebiotech.com/rss/biotech",
    "FDA": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    "Stat News": "https://www.statnews.com/category/biotech/feed",
    "Endpoints News": "https://endpts.com/feed/"
}

async def fetch_arxiv_papers():
    """Fetch ArXiv papers from the last 24 hours."""
    print("Fetching ArXiv papers...")
    search_client = arxiv.Client()
    papers = []
    since = datetime.now(pytz.utc) - timedelta(days=1)
    
    for query in ARXIV_QUERIES:
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        for result in search_results(search_client, search):
            if result.published > since:
                papers.append({
                    "title": result.title,
                    "summary": result.summary,
                    "link": result.entry_id,
                    "source": "ArXiv",
                    "tags": result.categories
                })
    return papers

def search_results(client, search):
    """Bridge for arxiv library to get results."""
    return client.results(search)

async def fetch_biorxiv_papers():
    """Fetch BioRxiv papers from the last 24 hours via API."""
    print("Fetching BioRxiv papers...")
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"https://api.biorxiv.org/details/biorxiv/{yesterday}/{today}/0/json"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                papers = []
                keywords = ["protein", "molecular", "cheminformatics", "alphafold", "diffusion"]
                for item in data.get("collection", []):
                    title_lower = item["title"].lower()
                    if any(kw in title_lower for kw in keywords):
                        papers.append({
                            "title": item["title"],
                            "summary": item["abstract"],
                            "link": f"https://doi.org/{item['doi']}",
                            "source": "BioRxiv",
                            "tags": ["Biology"]
                        })
                return papers
            return []

async def fetch_reddit_posts():
    """Fetch top posts from r/biotech and r/chempros."""
    subreddits = ["biotech", "chempros"]
    posts = []
    headers = {"User-Agent": "BioSmolBot/1.0"}
    
    print("Fetching Reddit posts...")
    async with aiohttp.ClientSession() as session:
        for sub in subreddits:
            url = f"https://www.reddit.com/r/{sub}/top.json?t=day&limit=5"
            try:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data['data']['children']:
                            p = item['data']
                            posts.append({
                                "title": p['title'],
                                "summary": p['selftext'][:500] + "..." if p['selftext'] else "Link post",
                                "link": f"https://www.reddit.com{p['permalink']}",
                                "source": f"r/{sub}",
                                "tags": ["Community"],
                                "relevance_score": 7 # Default high for community signal
                            })
            except Exception as e:
                print(f"Error fetching r/{sub}: {e}")
    return posts

async def fetch_rss_feeds():
    """Fetch RSS feeds."""
    print("Fetching RSS feeds...")
    articles = []
    for source, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            published = None
            if hasattr(entry, 'published_parsed'):
                published = datetime(*entry.published_parsed[:6], tzinfo=pytz.utc)
            
            since = datetime.now(pytz.utc) - timedelta(days=1)
            if not published or published > since:
                articles.append({
                    "title": entry.title,
                    "summary": entry.get("summary", "") or entry.get("description", ""),
                    "link": entry.link,
                    "source": source,
                    "tags": [source]
                })
    return articles

async def summarize_articles(articles):
    """Use Gemini to summarize and score articles."""
    if not articles:
        return []

    client = get_gemini_client()
    if not client:
        print("Warning: No Gemini API key found. Returning raw articles.")
        return [{**a, "relevance_score": 7, "summary": a["summary"][:200]} for a in articles]

    print(f"Summarizing {len(articles)} articles with Gemini (genai)...")
    summarized = []
    
    for i in range(0, len(articles), 5):
        # Rate limit to avoid 429 errors
        if i > 0:
            time.sleep(1) # Minimal buffer for paid tier
            
        batch = articles[i:i+5]
        prompt = """Summarize the following biotech/research articles. Return a JSON list of objects with 'title', 'summary' (max 2 sentences), 'tags', and 'relevance_score' (1-10).
        SCORING CRITERIA:
        - 10: HISTORIC/BREAKING (e.g. AlphaFold 4 release, >$10B Merger, Major FDA Approval/Rejection). This gets featured at the top.
        - 8-9: High Impact (e.g. Phase 3 reads, >$1B deal, SOTA ML Paper).
        - 6-7: Standard Industry News.
        - <6: Noise/PR fluff.
        """
        for idx, art in enumerate(batch):
            prompt += f"--- Article {idx+1} ---\nTitle: {art['title']}\nContent: {art['summary'][:1000]}\n\n"
        
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                ),
            )
            
            # Use json.loads because response.parsed doesn't work the same in all SDK versions
            batch_results = json.loads(response.text)
            
            if isinstance(batch_results, dict):
                results_list = batch_results.get("articles", batch_results.get("results", batch_results.get("list", [])))
                # If still dict and looks like a list might be indexed or something, handle it
                if not results_list and len(batch_results) > 0 and all(isinstance(v, dict) for v in batch_results.values()):
                     results_list = list(batch_results.values())
            else:
                results_list = batch_results
            
            if not isinstance(results_list, list):
                print(f"Unexpected JSON format from Gemini: {batch_results}")
                results_list = []

            for idx, res in enumerate(results_list):
                if idx < len(batch):
                    # We keep everything >= 6, but filtered usually happens later or we allow high recall here
                    res["link"] = batch[idx]["link"]
                    res["source"] = batch[idx]["source"]
                    summarized.append(res)
        except Exception as e:
            print(f"Error in Gemini summarization: {e}")
            for art in batch:
                summarized.append({
                    "title": art["title"],
                    "summary": art["summary"][:200] + "...",
                    "tags": art["tags"],
                    "relevance_score": 7,
                    "link": art["link"],
                    "source": art["source"]
                })
                
    return summarized

async def fetch_pubchem_data(molecule_name):
    """Fetch structure image and basic properties from PubChem."""
    # Clean name: "Ozempic (Semaglutide)" -> "Semaglutide", "Keytruda (Pembrolizumab)" -> "Pembrolizumab"
    # If generic is in parens, use that.
    clean_name = molecule_name
    if "(" in molecule_name:
        # Extract text inside parens
        import re
        match = re.search(r'\((.*?)\)', molecule_name)
        if match:
            clean_name = match.group(1)
    
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    print(f"Fetching PubChem data for {clean_name} (from {molecule_name})...")
    
    async with aiohttp.ClientSession() as session:
        # 1. Get CID
        cid_url = f"{base_url}/compound/name/{clean_name}/cids/JSON"
        async with session.get(cid_url) as resp:
            if resp.status != 200:
                print(f"Failed to find CID for {clean_name}")
                return None
            try:
                data = await resp.json()
                cid = data['IdentifierList']['CID'][0]
            except (KeyError, IndexError, TypeError):
                 print(f"Error parsing CID data for {clean_name}")
                 return None

        # 2. Get Properties
        props_url = f"{base_url}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES/JSON"
        async with session.get(props_url) as resp:
            if resp.status != 200:
                props = {}
            else:
                data = await resp.json()
                props = data['PropertyTable']['Properties'][0]

    image_url = f"{base_url}/compound/cid/{cid}/PNG?record_type=2d&image_size=large"
    
    return {
        "cid": cid,
        "image_url": image_url,
        "formula": props.get("MolecularFormula", "N/A"),
        "weight": props.get("MolecularWeight", "N/A"),
        "iupac": props.get("IUPACName", "N/A"),
        "smiles": props.get("IsomericSMILES", "N/A")
    }

async def generate_global_summary(articles, paper_count, feed_count):
    """Generate a news.smol.ai style 'Top of the Fold' summary."""
    client = get_gemini_client()
    if not client:
        return {
            "stats": f"Scanned {paper_count} papers and {feed_count} news feeds.",
            "vibe": "AI summarization unavailable. Check API key."
        }
        
    titles = [a['title'] for a in articles[:15] if a.get('relevance_score', 0) > 7]
    prompt = f"""
    Write a 2-3 sentence 'Daily Vibe' summary for a biotech dashboard based on these high-impact headlines: {titles}.
    
    INSTRUCTIONS:
    - If the list is empty or contains only minor news, explicitly state: "Quiet day in the industry." or "Slow news day." and suggest a general theme (e.g. "Focus remains on macro trends").
    - If there are MAJOR headlines, summarize the key event immediately.
    - Tone: Sophisticated, skeptical, industry-insider.
    
    Format as JSON: {{"vibe": "..."}}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
            ),
        )
        data = json.loads(response.text)
        return {
            "stats": f"Scanned {paper_count} papers and {feed_count} industry sources.",
            "vibe": data.get("vibe", "Biotech never sleeps.")
        }
    except Exception as e:
        print(f"Error generating summary: {e}")
        return {
            "stats": f"Scanned {paper_count} papers.",
            "vibe": "Today's biotech landscape at a glance."
        }

async def get_dynamic_molecule_candidate():
    """Ask Gemini for a clinically significant small molecule to feature."""
    client = get_gemini_client()
    if not client:
        return "Caffeine" # Fallback
        
    history_file = "molecule_history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []

    exclude_list = ", ".join(history[-30:]) if history else "None"
    
    COOL_FALLBACKS = [
        "Sotorasib", "Niraparib", "Risdiplam", "Venetoclax", 
        "Ubrogepant", "Capmatinib", "Tepotinib", "Lonafarnib"
    ]
    fallback = random.choice([m for m in COOL_FALLBACKS if m not in history] or COOL_FALLBACKS)

    prompt = f"""
    Propose 1 SMALL MOLECULE drug (organic chemistry, <1000 Da) that is either:
    1. A recent significant approval (2020-2025)
    2. A "Hot" mechanism (e.g. molecular glue, KRAS, radioligand)
    3. A NOTABLE FAIL or withdrawal (e.g. unexpected toxicity, Phase 3 bust).
    
    CRITICAL: Avoid biologics (mAbs, proteins, gene therapy). Must have a PubChem CID.
    Do NOT choose: Aspirin, Caffeine, Tylenol, or standard ancient generics.
    ALSO, do NOT choose any of these recently featured molecules: {exclude_list}.
    
    Return JSON: {{"molecule": "Name"}}
    """
    
    # Remove artificial delay for paid tier
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
            ),
        )
        data = json.loads(response.text)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        return data.get("molecule", fallback) if isinstance(data, dict) else fallback
    except Exception as e:
        print(f"Gemini Candidate API Error: {e}")
        return fallback

async def generate_molecule_of_day():
    """Generate a DrugHunter-style profile for a dynamic molecule."""
    # 1. Get a fresh candidate
    molecule = await get_dynamic_molecule_candidate()
    print(f"Generating profile for {molecule}...")
    
    # Save to history
    history_file = "molecule_history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    
    if molecule not in history:
        history.append(molecule)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    # Minimal 1s delay for paid tier burst protection
    time.sleep(1)

    # Fetch real chemical data
    pubchem_data = await fetch_pubchem_data(molecule)
    
    client = get_gemini_client()
    if not client:
        return {
            "name": molecule,
            "image": pubchem_data.get('image_url') if pubchem_data else "",
            "year_discovered": "N/A",
            "year_approved": "N/A",
            "origin_story": "API Key missing.",
            "mechanism": "Please check configuration.",
            "significance": "Mechanism of action summary.",
            "properties": pubchem_data
        }

    prompt = f"""
    Write a sophisticated 'Molecule of the Day' profile for {molecule}.
    Target Audience: Medicinal Chemists.
    
    Return ONLY JSON with these exact keys:
    {{
      "name": "{molecule}",
      "year_discovered": "YYYY (concise, e.g. '2003')",
      "year_approved": "YYYY or 'N/A' (concise, e.g. '2023')",
      "origin_story": "One sentence on the discovery/lead optimization (max 20 words).",
      "mechanism": "MOA - mention specific target/pocket (max 25 words).",
      "significance": "Why it matters/Market impact (max 20 words)."
    }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
            ),
        )
        mol_data = json.loads(response.text)
        if isinstance(mol_data, list):
            mol_data = mol_data[0]
            
        mol_data['image'] = pubchem_data.get('image_url') if pubchem_data else ""
        mol_data['properties'] = pubchem_data
        return mol_data
        
    except Exception as e:
        print(f"Error generating molecule: {e}")
        return {
            "name": molecule,
            "image": pubchem_data.get('image_url') if pubchem_data else "",
            "year_discovered": "Error",
            "year_approved": "Error",
            "origin_story": "Error generating profile.",
            "mechanism": "Consult pharmacopeia.",
            "significance": "Consult literature.",
            "properties": pubchem_data
        }

async def main():
    research_papers = await asyncio.gather(fetch_arxiv_papers(), fetch_biorxiv_papers())
    all_research = research_papers[0] + research_papers[1]
    
    business_news = await fetch_rss_feeds()
    reddit_posts = await fetch_reddit_posts()
    
    all_news = business_news + reddit_posts
    
    # Generate Global Summary
    global_summary = await generate_global_summary(
        all_research + all_news, 
        len(all_research), 
        len(all_news)
    )

    summarized_research = await summarize_articles(all_research)
    summarized_business = await summarize_articles(all_news)
    molecule = await generate_molecule_of_day()
    # Filter Hero Article
    summarized_all = summarized_research + summarized_business
    hero_article = None
    
    # Sort by score descending
    summarized_all.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # If top article is highly relevant (>= 9), promote it
    if summarized_all and summarized_all[0].get("relevance_score", 0) >= 9:
        hero_article = summarized_all[0]
        # Remove from regular lists so it's not duplicated
        if hero_article in summarized_research:
            summarized_research.remove(hero_article)
        if hero_article in summarized_business:
            summarized_business.remove(hero_article)
            
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('index.html')
    output = template.render(
        summary=global_summary,
        hero_article=hero_article,
        molecule=molecule,
        research=summarized_research,
        business=summarized_business,
        date=datetime.now().strftime("%B %d, %Y")
    )
    
    # Create public directory if it doesn't exist
    os.makedirs("public", exist_ok=True)
    
    with open("public/index.html", "w") as f:
        f.write(output)
    print("Dashboard generated: public/index.html")

if __name__ == "__main__":
    asyncio.run(main())
