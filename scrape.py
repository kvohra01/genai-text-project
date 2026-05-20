"""
Family Guy Script Scraper — Springfield! Springfield!
=====================================================
Scrapes all Family Guy episode transcripts and saves them as:
  - Individual .txt files per episode in ./scripts/
  - A single concatenated corpus file: family_guy_corpus.txt

Usage:
    pip install requests beautifulsoup4 tqdm
    python scrape_family_guy.py

Respect the site: the scraper includes delays between requests.
"""

import os
import re
import time
import random
import logging
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────

BASE_URL = "https://www.springfieldspringfield.co.uk"
SHOW_SLUG = "family-guy"
EPISODE_LIST_URL = f"{BASE_URL}/episode_scripts.php?tv-show={SHOW_SLUG}"
SCRIPT_URL_TPL = f"{BASE_URL}/view_episode_scripts.php?tv-show={SHOW_SLUG}&episode={{ep_slug}}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

OUTPUT_DIR = Path("scripts")
CORPUS_FILE = Path("family_guy_corpus.txt")
DELAY_RANGE = (1.5, 3.5)  # seconds between requests — be polite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def polite_get(url: str, session: requests.Session, retries: int = 3) -> requests.Response | None:
    """GET with retries, backoff, and random delay."""
    for attempt in range(1, retries + 1):
        try:
            time.sleep(random.uniform(*DELAY_RANGE))
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            wait = 2 ** attempt + random.random()
            log.warning(f"Attempt {attempt}/{retries} failed for {url}: {e}  — retrying in {wait:.1f}s")
            time.sleep(wait)
    log.error(f"Gave up on {url}")
    return None


def get_episode_slugs(session: requests.Session) -> list[dict]:
    """
    Scrape the episode-list page to get every episode slug.
    Returns list of dicts: {slug, label}  e.g. {slug: "s01e01", label: "S01E01"}
    """
    log.info(f"Fetching episode list from {EPISODE_LIST_URL}")
    resp = polite_get(EPISODE_LIST_URL, session)
    if not resp:
        raise RuntimeError("Could not fetch episode list page.")

    soup = BeautifulSoup(resp.text, "html.parser")

    # The site lists episodes as <a> tags whose href contains the show slug
    episodes = []
    seen = set()

    for a_tag in soup.select("a.season-episode-title"):
        href = a_tag.get("href", "")
        # Extract episode slug from href
        # Typical href: /view_episode_scripts.php?tv-show=family-guy&episode=s01e01
        match = re.search(r"episode=(s\d+e\d+)", href, re.IGNORECASE)
        if match:
            slug = match.group(1).lower()
            if slug not in seen:
                seen.add(slug)
                label = slug.upper()
                episodes.append({"slug": slug, "label": label})

    # Fallback: if the CSS selector didn't work, try a broader search
    if not episodes:
        log.warning("Primary selector found 0 episodes — trying fallback selector")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if SHOW_SLUG in href and "episode=" in href:
                match = re.search(r"episode=(s\d+e\d+)", href, re.IGNORECASE)
                if match:
                    slug = match.group(1).lower()
                    if slug not in seen:
                        seen.add(slug)
                        episodes.append({"slug": slug, "label": slug.upper()})

    # Sort by season and episode number
    def sort_key(ep):
        m = re.match(r"s(\d+)e(\d+)", ep["slug"])
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

    episodes.sort(key=sort_key)
    log.info(f"Found {len(episodes)} episodes")
    return episodes


def scrape_script(slug: str, session: requests.Session) -> str | None:
    """Fetch and extract the script text for one episode."""
    url = SCRIPT_URL_TPL.format(ep_slug=slug)
    resp = polite_get(url, session)
    if not resp:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # The script text lives in <div class="scrolling-script-container">
    container = soup.find("div", class_="scrolling-script-container")

    # Fallback selectors if the site layout changed
    if not container:
        container = soup.find("div", class_="movie_script")
    if not container:
        container = soup.find("div", id="textblock")

    if not container:
        log.warning(f"Could not find script container for {slug}")
        return None

    text = container.get_text(separator="\n", strip=True)
    # Normalise whitespace: collapse runs of blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape Family Guy scripts")
    parser.add_argument("--start-season", type=int, default=1, help="Start from this season")
    parser.add_argument("--end-season", type=int, default=99, help="Stop after this season")
    parser.add_argument("--delay", type=float, default=None, help="Override min delay between requests (secs)")
    parser.add_argument("--output-dir", type=str, default="scripts", help="Directory for individual episode files")
    args = parser.parse_args()

    global DELAY_RANGE
    if args.delay is not None:
        DELAY_RANGE = (args.delay, args.delay + 1.5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # 1. Get episode list
    episodes = get_episode_slugs(session)
    if not episodes:
        log.error("No episodes found. The site layout may have changed.")
        return

    # Filter by season range
    def season_num(slug):
        m = re.match(r"s(\d+)", slug)
        return int(m.group(1)) if m else 0

    episodes = [
        ep for ep in episodes
        if args.start_season <= season_num(ep["slug"]) <= args.end_season
    ]
    log.info(f"Scraping {len(episodes)} episodes (seasons {args.start_season}–{args.end_season})")

    # 2. Scrape each episode
    failed = []
    corpus_parts = []

    for ep in tqdm(episodes, desc="Scraping", unit="ep"):
        slug, label = ep["slug"], ep["label"]
        out_path = output_dir / f"{slug}.txt"

        # Skip if already downloaded
        if out_path.exists() and out_path.stat().st_size > 100:
            log.debug(f"Skipping {label} (already exists)")
            text = out_path.read_text(encoding="utf-8")
        else:
            text = scrape_script(slug, session)
            if text:
                out_path.write_text(text, encoding="utf-8")
                log.info(f"✓ {label}  ({len(text):,} chars)")
            else:
                failed.append(label)
                continue

        # Build corpus entry with a clear episode delimiter
        header = f"\n{'='*60}\n{label}\n{'='*60}\n"
        corpus_parts.append(header + text)

    # 3. Write combined corpus
    corpus = "\n".join(corpus_parts)
    CORPUS_FILE.write_text(corpus, encoding="utf-8")

    # 4. Summary
    total_chars = len(corpus)
    approx_tokens = total_chars // 4  # rough BPE estimate
    log.info("─" * 50)
    log.info(f"Done!  {len(episodes) - len(failed)}/{len(episodes)} episodes scraped")
    log.info(f"Corpus size: {total_chars:,} chars  (~{approx_tokens:,} tokens)")
    log.info(f"Individual files: {output_dir}/")
    log.info(f"Full corpus:      {CORPUS_FILE}")
    if failed:
        log.warning(f"Failed episodes:  {', '.join(failed)}")

if __name__ == "__main__":
    main()