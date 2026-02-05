import asyncio
import json
import os
import random
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
    "protein design",
    "molecular generation",
    "cheminformatics",
    "alphafold",
    "diffusion model"
]

RSS_FEEDS = {
    "FierceBiotech": "https://www.fiercebiotech.com/rss/biotech",
    "FDA": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    "Stat News": "https://www.statnews.com/category/biotech/feed"
}

MOLECULES = [
    "Keytruda (Pembrolizumab)", "Thalidomide", "Imatinib (Gleevec)", "Aspirin",
    "Humira (Adalimumab)", "Metformin", "Penicillin", "Insulin",
    "Ozempic (Semaglutide)", "Paxlovid", "Epipen", "Sildenafil",
    "Prozac (Fluoxetine)", "Lipitor (Atorvastatin)", "Remicade"
]

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
                    "tags": [t.term for t in result.tags]
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
        batch = articles[i:i+5]
        prompt = "Summarize the following biotech/research articles. Return a JSON list of objects with 'title', 'summary' (max 2 sentences), 'tags', and 'relevance_score' (1-10). Keep the density high.\n\n"
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
                    if res.get("relevance_score", 0) >= 6:
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

async def generate_molecule_of_day():
    """Generate a DrugHunter-style profile for a random molecule."""
    molecule = random.choice(MOLECULES)
    print(f"Generating profile for {molecule}...")
    
    client = get_gemini_client()
    if not client:
        return {
            "name": molecule,
            "discovery": "API Key missing.",
            "mechanism": "Please check configuration.",
            "significance": "Mechanism of action summary."
        }

    prompt = f"Write a '1-minute read' profile for {molecule}. Include 'discovery story', 'mechanism of action', and 'why it matters'. Tone: industry-insider. Return JSON with keys: 'name', 'discovery', 'mechanism', 'significance'."
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating molecule: {e}")
        return {
            "name": molecule,
            "discovery": "A landmark therapeutic discovery.",
            "mechanism": "Consult pharmacopeia for details.",
            "significance": "Foundational therapy in its class."
        }

async def main():
    research_papers = await asyncio.gather(fetch_arxiv_papers(), fetch_biorxiv_papers())
    all_research = research_papers[0] + research_papers[1]
    business_news = await fetch_rss_feeds()
    
    summarized_research = await summarize_articles(all_research)
    summarized_business = await summarize_articles(business_news)
    molecule = await generate_molecule_of_day()
    
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('index.html')
    output = template.render(
        molecule=molecule,
        research=summarized_research,
        business=summarized_business,
        date=datetime.now().strftime("%B %d, %Y")
    )
    
    with open("index.html", "w") as f:
        f.write(output)
    print("Dashboard generated: index.html")

if __name__ == "__main__":
    asyncio.run(main())
