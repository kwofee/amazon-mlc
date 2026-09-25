from collections import defaultdict
from pathlib import Path
import re
import unicodedata
from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from IPython.display import display
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
NUMBER_OF_GROUND_TRUTH_ROWS = 1_000
CHUNK_SIZE                  = 250_000

BASE_TRAIN   = '/home/laalenthika/Downloads/student_resource/dataset/train/'
OUTPUT_DIR   = "/home/laalenthika/Downloads/student_resource/output/"

SOURCE1_PATH      = BASE_TRAIN + 'train_source1.tsv'
SOURCE2_PATH      = BASE_TRAIN + 'train_source2.tsv'
SOURCE3_PATH      = BASE_TRAIN + 'train_source3.tsv'
GROUND_TRUTH_PATH = BASE_TRAIN + 'train_ground_truth.tsv'

# Thresholds — tune these to trade recall vs precision
TOKEN_SORT_THRESHOLD  = 85   # rapidfuzz token_sort_ratio
NGRAM_THRESHOLD       = 0.4  # character trigram jaccard
TFIDF_THRESHOLD       = 0.6  # cosine similarity
FUZZY_THRESHOLD       = 80   # rapidfuzz partial_ratio



# ── Text helpers ──────────────────────────────────────────────────────────────
LEGAL_AND_DOMAIN_WORDS = {
    "co", "com", "company", "corp", "corporation", "inc", "llc", "llp",
    "ltd", "limited", "net", "org", "private", "pvt", "sas", "sasu",
    "sarl", "sci", "www",
}

def normalize_text(value):
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return " ".join(text.split())

def remove_accents(value):
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return normalize_text(text)

def core_name(value):
    tokens = normalize_text(value).split()
    return "".join(t for t in tokens if t not in LEGAL_AND_DOMAIN_WORDS)

def token_sorted_name(value):
    tokens = normalize_text(value).split()
    tokens = [t for t in tokens if t not in LEGAL_AND_DOMAIN_WORDS]
    return " ".join(sorted(tokens))

def country_key(country, value):
    return f"{country}\x1f{value}"

def char_ngrams(text, n=3):
    text = text.replace(" ", "")
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def ngram_jaccard(a, b, n=3):
    sa, sb = char_ngrams(a, n), char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

# ── Load ground truth & S1 ────────────────────────────────────────────────────
ground_truth = pd.read_csv(
    GROUND_TRUTH_PATH, sep='\t', dtype=str,
    keep_default_na=False, nrows=NUMBER_OF_GROUND_TRUTH_ROWS
)
source1_ids = set(ground_truth['source1_entity_id'])

def read_selected_source1():
    chunks = []
    for chunk in pd.read_csv(SOURCE1_PATH, sep='\t', dtype=str,
                              keep_default_na=False, chunksize=CHUNK_SIZE):
        sel = chunk[chunk['entity_id'].isin(source1_ids)]
        if not sel.empty:
            chunks.append(sel)
    return pd.concat(chunks, ignore_index=True)

source1 = read_selected_source1()
print(f"S1 records loaded: {len(source1):,}")

# ── Build exact indexes ───────────────────────────────────────────────────────
exact_name_index       = defaultdict(set)
core_name_index        = defaultdict(set)
accent_core_name_index = defaultdict(set)
exact_address_index    = defaultdict(set)
token_sorted_index     = defaultdict(set)

for rec in source1.itertuples(index=False):
    country, entity_id = rec.country, rec.entity_id
    en  = normalize_text(rec.business_name)
    cn  = core_name(rec.business_name)
    acn = core_name(remove_accents(rec.business_name))
    ea  = normalize_text(rec.business_address)
    ts  = token_sorted_name(rec.business_name)

    if en:  exact_name_index      [country_key(country, en)] .add(entity_id)
    if cn:  core_name_index       [country_key(country, cn)] .add(entity_id)
    if acn: accent_core_name_index[country_key(country, acn)].add(entity_id)
    if ea:  exact_address_index   [country_key(country, ea)] .add(entity_id)
    if ts:  token_sorted_index    [country_key(country, ts)] .add(entity_id)

print("Exact indexes built.")

# ── Build TF-IDF index ────────────────────────────────────────────────────────
s1_records = source1.copy()
s1_records['norm_name'] = s1_records['business_name'].map(normalize_text)

tfidf_indexes = {}
for country, grp in s1_records.groupby('country'):
    names      = grp['norm_name'].tolist()
    entity_ids = grp['entity_id'].tolist()
    if len(names) < 2:
        continue
    vec    = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    matrix = vec.fit_transform(names)
    tfidf_indexes[country] = {'vectorizer': vec, 'matrix': matrix, 'entity_ids': entity_ids}

print(f"TF-IDF indexes built for: {list(tfidf_indexes.keys())}")

# ── Build token inverted index ────────────────────────────────────────────────
token_inverted_index = defaultdict(set)
for rec in source1.itertuples(index=False):
    tokens = [t for t in normalize_text(rec.business_name).split()
              if t not in LEGAL_AND_DOMAIN_WORDS and len(t) > 2]
    for token in tokens:
        token_inverted_index[f"{rec.country}\x1f{token}"].add(rec.entity_id)

s1_norm_lookup = {
    rec.entity_id: normalize_text(rec.business_name)
    for rec in source1.itertuples(index=False)
}
print(f"Token inverted index size: {len(token_inverted_index):,} keys")

# ── Candidate store ───────────────────────────────────────────────────────────
candidate_channels = defaultdict(set)

def add_candidates(source1_matches, candidate_id, channel_name):
    for s1_id in source1_matches:
        candidate_channels[(s1_id, candidate_id)].add(channel_name)

def collect_exact_matches(keys, candidate_ids, lookup, channel_name):
    matched_mask = keys.isin(lookup)
    for key, cid in zip(keys[matched_mask], candidate_ids[matched_mask]):
        add_candidates(lookup[key], cid, channel_name)

# ── Channel functions (ALL defined before scan_target_source) ─────────────────
def tfidf_scan(chunk):
    for country, grp in chunk.groupby('country'):
        if country not in tfidf_indexes:
            continue
        idx         = tfidf_indexes[country]
        cand_names  = grp['business_name'].map(normalize_text).tolist()
        cand_ids    = grp['entity_id'].tolist()
        cand_matrix = idx['vectorizer'].transform(cand_names)
        sims        = cosine_similarity(cand_matrix, idx['matrix'])
        for i, cand_id in enumerate(cand_ids):
            for j in np.where(sims[i] >= TFIDF_THRESHOLD)[0]:
                add_candidates({idx['entity_ids'][j]}, cand_id, 'tfidf_cosine')

def fast_fuzzy_ngram_scan(chunk):
    for row in chunk.itertuples(index=False):
        cand_norm = normalize_text(row.business_name)
        tokens    = [t for t in cand_norm.split()
                     if t not in LEGAL_AND_DOMAIN_WORDS and len(t) > 2]
        if not tokens:
            continue

        s1_candidates = set()
        for token in tokens:
            s1_candidates |= token_inverted_index.get(f"{row.country}\x1f{token}", set())

        for s1_id in s1_candidates:
            s1_norm = s1_norm_lookup.get(s1_id, "")
            if not s1_norm:
                continue
            if fuzz.token_sort_ratio(cand_norm, s1_norm) >= TOKEN_SORT_THRESHOLD:
                add_candidates({s1_id}, row.entity_id, 'token_sort_fuzzy')
            elif len(cand_norm) >= 3 and ngram_jaccard(cand_norm, s1_norm) >= NGRAM_THRESHOLD:
                add_candidates({s1_id}, row.entity_id, 'char_ngram')

# ── Main scan (defined last, after all helpers) ───────────────────────────────
MAX_CHUNKS = 2  # change this to however many you want to test with

def scan_target_source(path, max_chunks=MAX_CHUNKS):
    print(f"\nScanning {path} ...")
    for chunk_num, chunk in enumerate(
        pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE), 1
    ):
        print(f"  chunk {chunk_num} ({len(chunk):,} rows) ...", end=' ')

        exact_names        = chunk['business_name'].map(normalize_text)
        compact_names      = chunk['business_name'].map(core_name)
        accent_compact     = chunk['business_name'].map(lambda v: core_name(remove_accents(v)))
        exact_addresses    = chunk['business_address'].map(normalize_text)
        token_sorted_names = chunk['business_name'].map(token_sorted_name)

        collect_exact_matches(chunk['country'] + "\x1f" + exact_names,        chunk['entity_id'], exact_name_index,       'exact_name')
        collect_exact_matches(chunk['country'] + "\x1f" + compact_names,      chunk['entity_id'], core_name_index,        'core_name')
        collect_exact_matches(chunk['country'] + "\x1f" + accent_compact,     chunk['entity_id'], accent_core_name_index, 'accent_core_name')
        collect_exact_matches(chunk['country'] + "\x1f" + exact_addresses,    chunk['entity_id'], exact_address_index,    'exact_address')
        collect_exact_matches(chunk['country'] + "\x1f" + token_sorted_names, chunk['entity_id'], token_sorted_index,     'token_sorted')

        fast_fuzzy_ngram_scan(chunk)
        tfidf_scan(chunk)

        print(f"candidates so far: {len(candidate_channels):,}")

        if chunk_num >= max_chunks:
            print(f"  Stopped early at chunk {chunk_num}")
            break

# ── Run ───────────────────────────────────────────────────────────────────────
candidate_channels = defaultdict(set)

scan_target_source(SOURCE2_PATH, max_chunks=2)
scan_target_source(SOURCE3_PATH, max_chunks=2)

print(f"\nTotal candidate pairs: {len(candidate_channels):,}")

import os

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Build candidate_details dataframe ────────────────────────────────────────
candidate_details = pd.DataFrame([
    {
        'source1_entity_id':  s1_id,
        'candidate_entity_id': cand_id,
        'channels':            ','.join(sorted(channels)),
        'channel_count':       len(channels),
    }
    for (s1_id, cand_id), channels in candidate_channels.items()
])

# ── Save candidate_details ────────────────────────────────────────────────────
candidate_details.to_csv(
    OUTPUT_DIR + 'development_candidate_details.tsv',
    sep='\t', index=False
)
print(f"Saved candidate_details: {len(candidate_details):,} rows")

# ── Build and save candidate_pairs (submission format) ───────────────────────
candidate_lists = (
    candidate_details
    .groupby('source1_entity_id')['candidate_entity_id']
    .agg(lambda values: ','.join(dict.fromkeys(values)))
    .rename('candidate_entity_ids')
    .reset_index()
)

# Every S1 must appear even if no candidates found
candidate_lists = (
    ground_truth[['source1_entity_id']]
    .merge(candidate_lists, on='source1_entity_id', how='left')
    .fillna({'candidate_entity_ids': ''})
)

candidate_lists.to_csv(
    OUTPUT_DIR + 'development_candidate_pairs_2.tsv',
    sep='\t', index=False
)
print(f"Saved candidate_pairs:   {len(candidate_lists):,} rows")
print(f"\nFiles saved to: {OUTPUT_DIR}")
# ── Run ───────────────────────────────────────────────────────────────────────
# candidate_channels = defaultdict(set)   # reset before running

# scan_target_source(SOURCE2_PATH)
# scan_target_source(SOURCE3_PATH)

# print(f"\nTotal candidate pairs: {len(candidate_channels):,}")