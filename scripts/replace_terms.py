"""Rename the key terms of the downloaded universe and build the knowledge base.

Input:  data/raw/*.txt (output of download_pages.py)
Output: knowledge_base/*.md and terms_map.json (original term -> invented term)

The map has two parts:
1. CURATED - hand-picked replacements for the main terms and for terms that are
   ordinary English words (Force, Empire, Solo), which cannot be detected automatically.
2. Auto-detected proper nouns: words that are always capitalized in the corpus and are
   absent from the English dictionary (Tatooine, Kenobi). They get a generated name.

Usage: python scripts/replace_terms.py
"""

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
KB_DIR = ROOT / "knowledge_base"
MAP_PATH = ROOT / "terms_map.json"
DICT_PATH = Path("/usr/share/dict/words")

CURATED = {
    # --- the universe, factions, eras ---
    "Star Wars": "Astral Saga",
    "Death Star": "Void Core", "Death Stars": "Void Cores",
    "Force": "Synth Flux",
    "dark side": "umbral tide", "Dark Side": "Umbral Tide",
    "light side": "radiant tide", "Light Side": "Radiant Tide",
    "Jedi": "Veyari", "Sith": "Morkai", "Darth": "Xarn",
    "Padawan": "Tirren", "Padawans": "Tirrens", "padawan": "tirren",
    "Republic": "Concord", "Empire": "Hegemony", "Emperor": "Hegemon",
    "Imperial": "Hegemonic", "Imperials": "Hegemonics",
    "Rebel": "Insurgent", "Rebels": "Insurgents", "Rebellion": "Insurgency",
    "rebel": "insurgent", "rebels": "insurgents", "rebellion": "insurgency",
    "Alliance": "Pact",
    "Separatist": "Secessionist", "Separatists": "Secessionists",
    "Confederacy": "League", "Confederate": "League",
    "Trade Federation": "Merchant Combine",
    "First Order": "Prime Mandate", "Resistance": "Defiance",
    "Order 66": "Directive 91",
    "Clone": "Replica", "Clones": "Replicas", "clone": "replica", "clones": "replicas",
    "Moff": "Vizor", "Moffs": "Vizors",
    "Outer Rim": "Far Verge", "Mid Rim": "Mid Verge", "Inner Rim": "Near Verge", "Rim": "Verge",
    "Cloud City": "Sky Haven", "Rogue One": "Stray One",
    "BBY": "BVC", "ABY": "AVC",
    # --- characters whose names are ordinary English words or the main heroes ---
    "Vader": "Velgor", "Anakin": "Edrik", "Skywalker": "Vantreil", "Skywalkers": "Vantreils",
    "Luke": "Corin", "Leia": "Mirae", "Organa": "Tessaly",
    "Han": "Dax", "Solo": "Varro", "Obi": "Oru", "Wan": "Kei", "Kenobi": "Talvane",
    "Maul": "Skorn", "Boba": "Teko", "Ben": "Tov", "Ren": "Vash", "Bail": "Dorn",
    "Mon": "Sel", "Gon": "Rhal", "Jinn": "Ormis", "Finn": "Brannoc", "Poe": "Arlo",
    "Grievous": "Kharzul", "Rex": "Tarn", "Bridger": "Hollin", "Bane": "Skar", "Cad": "Zev",
    "Lars": "Odrin", "Owen": "Halder", "Mace": "Joran", "Snoke": "Vrask",
    "Jar Jar": "Nib Nib", "Binks": "Tolk", "Veers": "Drask", "Rax": "Venn", "Sing": "Lyth",
    "Cassian": "Teodric", "Shand": "Morrow", "Thrawn": "Vyrell", "Wren": "Aldis",
    "Sabine": "Maris", "Hondo": "Barrik", "Kanan": "Jorem", "Savage": "Brakk",
    "Din": "Kol", "Saw": "Dorrak", "Wedge": "Tobin", "Bib": "Ose", "Galen": "Aldric",
    "Gardulla": "Hesska", "Chopper": "Rivet", "Vos": "Drel", "Cal": "Wes", "Bibble": "Tharn",
    "Koss": "Derr", "Bey": "Ulla", "Kes": "Ovan", "Iden": "Sarra", "Wicket": "Pimm",
    "Yoda": "Olvek", "Chewbacca": "Grawlokk", "Chewie": "Grawl",
    "Padmé": "Naira", "Padme": "Naira", "Amidala": "Corvelle",
    "Sidious": "Nyxarion", "Palpatine": "Malverin", "Jabba": "Gorvo",
    "Lando": "Berrin", "Calrissian": "Maldonne", "Ahsoka": "Ilyra", "Tano": "Seshu",
    "Fett": "Drayk", "Jango": "Rogan", "Dooku": "Varnoth", "Tarkin": "Helvar",
    "Windu": "Ashkar", "Kylo": "Zerek", "Rey": "Kessa", "Qui": "Sael", "Mothma": "Aldren",
    "Artoo": "Kayseven", "Threepio": "Eightel",
    # --- places and peoples ---
    "Tatooine": "Sarrakesh", "Naboo": "Ilveren", "Hoth": "Kryost", "Endor": "Thalwen",
    "Alderaan": "Belisar", "Dagobah": "Murgath", "Yavin": "Orvax", "Geonosis": "Kharrab",
    "Kashyyyk": "Ruwokka", "Mustafar": "Pyrrhax", "Kamino": "Thessil", "Mandalore": "Kordath",
    "Hutt": "Zogg", "Ewok": "Tumli", "Jawa": "Skrit", "Tusken": "Dhural",
    "Coruscant": "Velaris", "Bespin": "Velmora", "Theed": "Quillan", "Mos": "Kar",
    "Polis": "Ostra", "Massa": "Dreya", "Lasan": "Ummar", "Teth": "Orrin",
    "Wookiee": "Varruk", "Trandoshan": "Skarrim", "Pyke": "Vosk", "Whills": "Ennari",
    # --- technology, ships, droids ---
    "Millennium Falcon": "Meridian Kestrel", "Falcon": "Kestrel",
    "Star Destroyer": "Nova Dreadnought", "Star Destroyers": "Nova Dreadnoughts",
    "Venator": "Aegis",
    "TIE": "VEX", "TIEs": "VEXes",
    "X-wing": "K-lance", "X-wings": "K-lances", "Y-wing": "J-lance", "Y-wings": "J-lances",
    "A-wing": "E-lance", "A-wings": "E-lances", "B-wing": "G-lance", "B-wings": "G-lances",
    "U-wing": "O-lance", "U-wings": "O-lances",
    "AT-AT": "HK-RA", "AT-ATs": "HK-RAs", "AT-ST": "HK-VO", "AT-STs": "HK-VOs",
    "DS-1": "VC-1", "DS-2": "VC-2", "T-65B": "K-41B", "T-65": "K-41", "T-70": "K-52", "T-85": "K-63",
    "YT-1300": "NV-2700", "YT-1300f": "NV-2700f", "YT-1300p": "NV-2700p",
    "R2-D2": "K7-M4", "R2": "K7", "C-3PO": "T-8LX", "C-3P0": "T-8LX", "3PO": "8LX", "BB-8": "DD-5",
    "lightsaber": "arcblade", "lightsabers": "arcblades",
    "Lightsaber": "Arcblade", "Lightsabers": "Arcblades",
    "blaster": "pulser", "blasters": "pulsers", "Blaster": "Pulser", "Blasters": "Pulsers",
    "turbolaser": "turbopulser", "turbolasers": "turbopulsers",
    "superlaser": "nova lance", "Superlaser": "Nova lance",
    "droid": "mechanoid", "droids": "mechanoids", "Droid": "Mechanoid", "Droids": "Mechanoids",
    "astromech": "navimech",
    "hyperdrive": "slipdrive", "hyperdrives": "slipdrives", "Hyperdrive": "Slipdrive",
    "Hyperdrives": "Slipdrives",
    "hyperspace": "slipspace", "Hyperspace": "Slipspace",
    "hyperlane": "sliplane", "hyperlanes": "sliplanes", "lightspeed": "slipspeed",
    "stormtrooper": "shocktrooper", "stormtroopers": "shocktroopers",
    "Stormtrooper": "Shocktrooper", "Stormtroopers": "Shocktroopers",
    "kyber": "vethra", "Kyber": "Vethra",
    "midi-chlorian": "flux-mote", "midi-chlorians": "flux-motes",
    "holocron": "mnemolith", "holocrons": "mnemoliths",
    "coaxium": "quorite", "carbonite": "ferrolite", "beskar": "ironveil", "bacta": "mendgel",
    "tibanna": "aerith", "bowcaster": "quarrelgun", "dejarik": "holochess",
    "sabacc": "vantik", "Sabacc": "Vantik",
    "sarlacc": "gorrath", "Sarlacc": "Gorrath",
    "tauntaun": "frostloper", "tauntauns": "frostlopers",
    "wampa": "snowmaw", "wampas": "snowmaws", "bantha": "dunebeast", "banthas": "dunebeasts",
    "krayt": "sarn",
    "podrace": "skimrace", "podracer": "skimracer", "podracers": "skimracers",
    "podracing": "skimracing", "Podrace": "Skimrace",
}

# Capitalized tokens that must stay as they are (Roman numerals, real-world abbreviations).
KEEP = {"II", "III", "IV", "VI", "VII", "VIII", "IX", "DNA"}
# Word endings used to link derived forms to the base term: Hutt -> Hutts, Alderaan -> Alderaanian.
DERIVED_SUFFIXES = ("ians", "ian", "ans", "an", "ese", "ites", "ite", "es", "ns", "s", "n", "i")
ENGLISH_SUFFIXES = ("s", "es", "ed", "d", "ing", "ly", "er", "ers", "n", "ies", "ied")

ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "h", "j", "k", "kr", "l", "m", "n", "p", "qu",
          "r", "s", "sh", "sk", "t", "th", "tr", "v", "vr", "z", "zh"]
VOWELS = ["a", "e", "i", "o", "u", "a", "e", "o", "ia", "ei"]
CODAS = ["", "", "", "l", "n", "r", "s", "th", "x", "sk", "rn", "m"]


def load_english_words() -> set[str]:
    return {w.strip() for w in DICT_PATH.read_text().splitlines() if w.strip().islower()}


def is_english(token: str, words: set[str]) -> bool:
    token = token.lower()
    if token in words:
        return True
    for suffix in ENGLISH_SUFFIXES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            if stem in words or stem + "e" in words or stem + "y" in words:
                return True
    return False


def invent_name(term: str, taken: set[str], words: set[str]) -> str:
    """Deterministic pseudo-random name: the same term always gets the same replacement."""
    for attempt in range(100):
        digest = hashlib.sha256(f"{term}:{attempt}".encode()).digest()
        if term.isupper():
            # Abbreviations and model codes stay abbreviations: BB -> QT.
            letters = "BDFGHJKLMNPQRSTVXZ"
            name = "".join(letters[b % len(letters)] for b in digest[: len(term)])
        else:
            syllables = 2 if len(term) <= 7 else 3
            name = ""
            for i in range(syllables):
                onset = ONSETS[digest[i * 3] % len(ONSETS)]
                vowel = VOWELS[digest[i * 3 + 1] % len(VOWELS)]
                coda = CODAS[digest[i * 3 + 2] % len(CODAS)] if i == syllables - 1 else ""
                name += onset + vowel + coda
            name = name.capitalize()
        if name not in taken and name.lower() not in words:
            taken.add(name)
            return name
    raise RuntimeError(f"cannot invent a name for {term}")


def find_base(token: str, known: dict[str, str]) -> tuple[str, str] | None:
    """Return (base term, suffix) if the token is a derived form of an already mapped term."""
    for suffix in DERIVED_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if stem in known:
                return stem, suffix
            if suffix.startswith("ian") and stem + "e" in known:
                return stem + "e", suffix
    return None


def detect_proper_nouns(text: str, words: set[str]) -> Counter:
    """Tokens that are always capitalized and are not English words."""
    tokens = re.findall(r"[^\W\d_]+", text)
    lowercase_seen = {t for t in tokens if t[0].islower()}
    capitalized = Counter(t for t in tokens if t[0].isupper())
    return Counter({
        t: n for t, n in capitalized.items()
        if t not in KEEP and t.lower() not in lowercase_seen and not is_english(t, words)
    })


def build_terms_map(text: str) -> dict[str, str]:
    words = load_english_words()
    terms = dict(CURATED)
    proper_nouns = detect_proper_nouns(text, words)
    # An invented name must not coincide with any original term.
    taken = set(terms.values()) | set(proper_nouns)
    # Shorter tokens first, so that base forms are mapped before derived ones.
    for token in sorted(proper_nouns, key=lambda t: (len(t), t)):
        if token in terms:
            continue
        base = find_base(token, terms)
        if base:
            terms[token] = terms[base[0]] + base[1]
        else:
            terms[token] = invent_name(token, taken, words)
    return terms


def compile_pattern(terms) -> re.Pattern:
    # Longest terms first: "Death Star" must win over "Star", "R2-D2" over "R2".
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")


def slugify(title: str) -> str:
    return re.sub(r"[^\w-]+", "_", title).strip("_").lower()


def main() -> None:
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        raise SystemExit("data/raw is empty: run scripts/download_pages.py first")
    documents = {path: path.read_text(encoding="utf-8") for path in raw_files}

    terms = build_terms_map("\n".join(documents.values()))
    pattern = compile_pattern(terms)

    shutil.rmtree(KB_DIR, ignore_errors=True)
    KB_DIR.mkdir(parents=True)
    replaced = 0
    leftovers = Counter()
    for text in documents.values():
        new_text, count = pattern.subn(lambda m: terms[m.group(0)], text)
        replaced += count
        leftovers.update(pattern.findall(new_text))
        title = new_text.splitlines()[0].lstrip("# ").strip()
        (KB_DIR / f"{slugify(title)}.md").write_text(new_text, encoding="utf-8")

    MAP_PATH.write_text(json.dumps(terms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"documents: {len(documents)}, terms in map: {len(terms)} "
          f"(curated: {len(CURATED)}), replacements made: {replaced}")
    # A leftover means an invented name coincides with an original term: fix CURATED.
    print(f"original terms still present after replacement: {dict(leftovers) or 'none'}")


if __name__ == "__main__":
    main()
