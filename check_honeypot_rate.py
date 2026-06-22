import sys
import csv
sys.path.insert(0, 'src')
from data_loader import load_candidates  # adjust import if your full-pool loader has a different name
from score import honeypot_guard

# --- 1. Load the full candidate pool so we can look up each ranked candidate's full record ---
print("Loading full candidate pool (this may take a moment)...")
all_candidates = load_candidates('data/candidates.jsonl')  # if this function name is different in your data_loader.py, tell me
cand_map = {c['candidate_id']: c for c in all_candidates}

# --- 2. Load your output CSV ---
OUTPUT_CSV = "submission.csv"  # change this to your actual output filename/path

flagged = []
total_rows = 0

with open(OUTPUT_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_rows += 1
        cid = row['candidate_id']
        cand = cand_map.get(cid)
        if cand is None:
            print(f"WARNING: {cid} not found in candidate pool!")
            continue
        if honeypot_guard(cand):
            flagged.append(cid)

rate = len(flagged) / total_rows * 100 if total_rows else 0

print()
print(f"Total rows checked: {total_rows}")
print(f"Honeypot-flagged candidates in your top 100: {len(flagged)}")
print(f"Honeypot rate: {rate:.1f}%")
print()
if flagged:
    print("Flagged candidate IDs:", flagged)
print()
if rate > 10:
    print("⚠️  WARNING: rate exceeds 10% — this would trigger disqualification per spec section 7.")
else:
    print("✅ Under the 10% disqualification threshold.")
