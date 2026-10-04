"""Download wiki pages and save them as clean text: one file per entity.

Source: Wookieepedia (starwars.fandom.com), text is licensed under CC BY-SA.
Output: data/raw/<Page_title>.txt with the original (not yet renamed) terms.

Usage: python scripts/download_pages.py
"""

import json
import re
import subprocess
import time
from urllib.parse import urlencode
from pathlib import Path

from bs4 import BeautifulSoup

API_URL = "https://starwars.fandom.com/api.php"
USER_AGENT = "Mozilla/5.0 (kb-builder; educational RAG project)"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Sections after the in-universe part of an article: real-world notes, lists of sources.
STOP_SECTIONS = {"behind the scenes", "appearances", "sources", "notes and references", "external links"}
# Articles are very long, so we keep the lead and the beginning of every section.
PARAGRAPHS_PER_SECTION = 2
MAX_CHARS = 20_000

PAGES = [
    # characters
    "Anakin Skywalker", "Luke Skywalker", "Leia Organa", "Han Solo", "Obi-Wan Kenobi",
    "Yoda", "Darth Sidious", "Chewbacca", "R2-D2", "C-3PO", "Padmé Amidala", "Boba Fett",
    "Jabba Desilijic Tiure", "Lando Calrissian", "Ahsoka Tano", "Maul",
    # planets
    "Tatooine", "Coruscant", "Hoth", "Endor", "Naboo", "Alderaan", "Dagobah",
    # technology and objects
    "Death Star", "DS-1 Death Star Mobile Battle Station", "Millennium Falcon", "Lightsaber", "The Force", "Hyperdrive",
    "T-65B X-wing starfighter", "Blaster",
    # organizations
    "Jedi Order", "Sith", "Galactic Empire", "Alliance to Restore the Republic", "Galactic Republic",
    # events
    "Clone Wars", "Battle of Yavin", "Battle of Hoth", "Battle of Endor", "Order 66",
    # species
    "Wookiee", "Hutt",
]


def fetch_html(page: str, retries: int = 4) -> tuple[str, str]:
    """Return (resolved title, rendered HTML) of a wiki page.

    The API is called with curl: the site's CDN rejects the Python HTTP clients (403).
    """
    params = {"action": "parse", "page": page, "prop": "text", "redirects": 1,
              "format": "json", "formatversion": 2}
    url = f"{API_URL}?{urlencode(params)}"
    for attempt in range(retries):
        result = subprocess.run(["curl", "-sS", "--fail", "-A", USER_AGENT, url],
                                capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if "error" in data:
                raise RuntimeError(f"{page}: {data['error'].get('info')}")
            return data["parse"]["title"], data["parse"]["text"]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{page}: {result.stderr.strip()}")


def clean_text(node) -> str:
    text = node.get_text()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.replace(" 's", "'s").replace(" ' s", "'s").strip()


def html_to_text(title: str, html: str) -> str:
    """Keep headings and paragraphs, drop infoboxes, quotes, images, references and navigation."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    for junk in root.select("sup, .mw-editsection, aside, table, figure, style, script, .noprint"):
        junk.decompose()

    lines = [f"# {title}", ""]
    paragraphs_in_section = 0
    in_lead = True
    total = 0
    for node in root.find_all(["h2", "h3", "p"]):
        # Skip paragraphs nested in quote or notice blocks.
        if node.name == "p" and node.find_parent("div", class_=re.compile("quote|notice|hatnote")):
            continue
        text = clean_text(node)
        if not text:
            continue
        if node.name in ("h2", "h3"):
            if node.name == "h2" and text.lower() in STOP_SECTIONS:
                break
            # Drop the previous heading if its section turned out to be empty.
            if lines[-2].startswith("#") and lines[-2] != lines[0] and paragraphs_in_section == 0:
                del lines[-2:]
            lines += [("## " if node.name == "h2" else "### ") + text, ""]
            paragraphs_in_section = 0
            in_lead = False
            continue
        if len(text) < 40:
            continue
        if not in_lead and paragraphs_in_section >= PARAGRAPHS_PER_SECTION:
            continue
        if total + len(text) > MAX_CHARS:
            break
        lines += [text, ""]
        paragraphs_in_section += 1
        total += len(text)

    while lines[-2].startswith("#") and lines[-2] != lines[0]:
        del lines[-2:]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for page in PAGES:
        title, html = fetch_html(page)
        text = html_to_text(title, html)
        path = RAW_DIR / (re.sub(r"[^\w-]+", "_", title).strip("_") + ".txt")
        path.write_text(text, encoding="utf-8")
        print(f"{title:45} {len(text):>7} chars -> {path.name}")
        time.sleep(1)


if __name__ == "__main__":
    main()
