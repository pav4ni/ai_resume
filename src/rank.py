"""
rank.py — Full-pool candidate ranking pipeline.

Usage:
  # Score 100K candidates (offline; requires precomputed embeddings):
  python src/rank.py --source full --out submission.csv

  # Smoke-test on the 50-candidate sample (NOTE: validator will fail —
  # it expects exactly 100 rows; sample mode is code-path only):
  python src/rank.py --source sample --out submission_sample.csv

Offline mode:
  Env vars are set at the top of this file so no network calls are made
  at ranking time.  Embeddings must be precomputed (run embed.py first).

Pipeline:
  1. Load candidates
  2. Ensure embedding cache exists (calls embed if needed — but assumes
     embeddings were precomputed in a prior setup step)
  3. Score every candidate with combine_score (see score.py)
  4. Sort descending by score; break ties by candidate_id ascending
  5. Take top 100, assign rank 1–100
  6. Write CSV, run validate_submission.py, print result
"""

import argparse
import csv
import os
import subprocess
import sys

# ---- Offline env vars (ranking must not touch network) ---------------------
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# ---- Path setup ------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_candidates, load_candidates_sample, load_job_description
from score import (
    combine_score,
    semantic_fit_score,
    consistency_score,
    structured_rule_score,
    honeypot_guard,
    behavioral_modifier,
)

VALIDATOR_PATH = os.path.join('data', 'validate_submission.py')


# ============================================================
# Reasoning template
# ============================================================

def _build_reasoning(candidate: dict, sf: float, cs: float, sr: float,
                     bm: float, is_honeypot: bool) -> str:
    """
    Produce a grounded, human-readable reasoning string.

    Rules:
    - Reference REAL fields from the candidate dict (title, yoe, skills, company).
    - Vary wording based on actual characteristics (location, open_to_work, score tier).
    - No LLM calls — pure string formatting from known fields.
    - Keep under ~250 chars so the CSV is readable.
    """
    if is_honeypot:
        return "Excluded: candidate flagged as honeypot (implausible skill durations or role overlap)."

    profile = candidate.get('profile', {})
    rs = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])

    title = profile.get('current_title', 'Unknown Role')
    yoe = float(profile.get('years_of_experience', 0) or 0)
    location = profile.get('location', '')
    country = profile.get('country', '')
    open_flag = rs.get('open_to_work_flag', False)
    resp_rate = float(rs.get('recruiter_response_rate', 0.0) or 0.0)
    last_active = rs.get('last_active_date', '')

    # Pick top skill by endorsements
    top_skill = ''
    if skills:
        best = max(skills, key=lambda s: s.get('endorsements', 0))
        top_skill = best.get('name', '')

    # Current company (most recent)
    current_co = profile.get('current_company', '')
    if career:
        for r in career:
            if r.get('is_current'):
                current_co = r.get('company', current_co)
                break

    # Semantic fit tier
    if sf >= 0.70:
        jd_phrase = "strong JD alignment on NLP/ranking/retrieval"
    elif sf >= 0.65:
        jd_phrase = "good JD fit for AI/ML engineering"
    else:
        jd_phrase = "partial JD overlap"

    # Location note
    loc_lower = location.lower()
    if country.lower() == 'india':
        if any(k in loc_lower for k in ('pune', 'noida')):
            loc_note = f"based in {location} (preferred location)"
        elif any(k in loc_lower for k in ('hyderabad', 'mumbai', 'delhi', 'gurgaon', 'gurugram')):
            loc_note = f"based in {location} (Tier-1 city)"
        else:
            loc_note = f"based in {location}, India"
    else:
        loc_note = f"located outside India ({country})"

    # Behavioral note
    if resp_rate >= 0.7 and open_flag:
        beh_note = f"highly responsive (rate={resp_rate:.0%}), actively open to work"
    elif resp_rate >= 0.5:
        beh_note = f"moderately responsive (rate={resp_rate:.0%})"
    elif not open_flag:
        beh_note = "not currently marked open-to-work"
    else:
        beh_note = f"low response rate ({resp_rate:.0%})"

    # Vary template by score tier
    if combine_score(candidate) >= 0.65:
        template = (
            f"{title} with {yoe:.1f} yrs experience at {current_co}. "
            f"Top skill: {top_skill}. "
            f"{jd_phrase.capitalize()}; {loc_note}. "
            f"{beh_note.capitalize()}."
        )
    else:
        template = (
            f"{title} ({yoe:.1f} yrs, {current_co}). "
            f"{jd_phrase.capitalize()}. "
            f"{loc_note}. "
            f"{beh_note.capitalize()}."
        )

    return template


# ============================================================
# Main ranking function
# ============================================================

def run(source: str = 'full', out: str = 'submission.csv') -> None:
    """
    Score all candidates, produce top-100 ranked CSV, validate.

    Args:
        source: 'full' reads candidates.jsonl; 'sample' reads sample_candidates.json.
        out:    Output CSV path.
    """
    print(f"[rank] Source: {source}  ->  output: {out}")

    # --- 1. Load candidates --------------------------------------------------
    if source == 'sample':
        print("[rank] Loading 50-candidate sample (note: validator will fail — expects 100 rows)")
        candidates = load_candidates_sample()
    else:
        print("[rank] Streaming full candidates.jsonl …")
        candidates = load_candidates('data/candidates.jsonl')
    print(f"[rank] Loaded {len(candidates)} candidates")

    # --- 2. Ensure embeddings cached -----------------------------------------
    # Check if cache exists; if not, run embed.
    from embed import load_cache, embed_candidates, embed_jd
    from data_loader import load_job_description

    cache_ids_path = os.path.join('data', 'cache', 'candidate_ids.json')
    if not os.path.exists(cache_ids_path):
        print("[rank] Embedding cache not found — running embed_candidates …")
        jd_text = load_job_description()
        embed_jd(jd_text)
        embed_candidates(candidates)
    else:
        import json
        with open(cache_ids_path) as f:
            cached_ids = set(json.load(f))
        new_count = sum(1 for c in candidates if c['candidate_id'] not in cached_ids)
        if new_count > 0:
            print(f"[rank] {new_count} candidates not yet embedded — running embed_candidates …")
            jd_text = load_job_description()
            embed_jd(jd_text)
            embed_candidates(candidates)
        else:
            print("[rank] Embedding cache is complete — skipping embed step")

    # --- 3. Score every candidate -------------------------------------------
    print("[rank] Scoring candidates …")
    scored = []
    for i, cand in enumerate(candidates):
        if i % 10000 == 0 and i > 0:
            print(f"  … {i}/{len(candidates)}")
        try:
            is_hp = honeypot_guard(cand)
            sf = semantic_fit_score(cand) if not is_hp else 0.0
            cs = consistency_score(cand)  if not is_hp else 0.0
            sr = structured_rule_score(cand) if not is_hp else 0.0
            bm = behavioral_modifier(cand) if not is_hp else 0.6
            cm = 0.0 if is_hp else combine_score(cand)
            reasoning = _build_reasoning(cand, sf, cs, sr, bm, is_hp)
            scored.append((cand['candidate_id'], cm, reasoning))
        except Exception as exc:
            # Do not let one bad record crash the whole pipeline
            print(f"  WARN: scoring failed for {cand.get('candidate_id', '?')}: {exc}")
            scored.append((cand.get('candidate_id', 'UNKNOWN'), 0.0, f"Scoring error: {exc}"))

    print(f"[rank] Scored {len(scored)} candidates")

    # --- 4. Sort: descending score, ascending candidate_id for ties ----------
    scored.sort(key=lambda x: (-x[1], x[0]))

    # --- 5. Top 100 with ranks -----------------------------------------------
    top100 = scored[:100]
    # Sanity: ensure monotonically non-increasing scores (they are, given sort above)
    # and tie-break order (candidate_id asc for equal scores — guaranteed by sort key)

    # --- 6. Write CSV --------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(out)) if os.path.dirname(out) else '.', exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, (cid, score, reasoning) in enumerate(top100, start=1):
            # Round score to 6 decimal places for clean output
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    print(f"[rank] Wrote {len(top100)} rows to {out}")

    # --- 7. Validate -----------------------------------------------------------
    print("[rank] Running validate_submission.py …")
    result = subprocess.run(
        [sys.executable, VALIDATOR_PATH, out],
        capture_output=True, text=True
    )
    print(result.stdout.strip() if result.stdout else "(no stdout)")
    if result.stderr:
        print("STDERR:", result.stderr.strip())
    if result.returncode != 0:
        print(f"[rank] Validator returned exit code {result.returncode}")
    else:
        print("[rank] Validation PASSED")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rank candidates and write submission CSV')
    parser.add_argument(
        '--source',
        choices=['full', 'sample'],
        default='full',
        help='full: stream candidates.jsonl (100K); sample: use sample_candidates.json (50)',
    )
    parser.add_argument(
        '--out',
        default='submission.csv',
        help='Output CSV path (default: submission.csv)',
    )
    args = parser.parse_args()
    run(source=args.source, out=args.out)
