# Hackathon Recruiter AI — Candidate Ranking Pipeline

## Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. One-time setup: download MiniLM model and precompute embeddings
#    (requires network access for the first run only)
python src/embed.py --source full

# 3. Offline ranking: score 100K candidates, write submission.csv, validate
python src/rank.py --source full --out submission.csv
```

- Step 2 takes ~5–10 min on CPU (100K candidates, batch_size=256).
- Step 3 takes ~2–4 min on CPU (pure numpy dot products, no network).
- Total wall-clock is well under 5 min once embeddings are cached.

---

## Architecture: 6-Layer Scoring

Each candidate is scored through six independent layers, then combined into a
final score in [0, 1].

### Layer 1 — Semantic Fit (weight 0.60)
`semantic_fit_score` computes cosine similarity between the candidate's
**narrative embedding** (summary + all role descriptions) and the JD embedding.
Narratives that closely match the JD's language around NLP, retrieval, ranking,
and recommendation systems score highest.

Mapping: `(cos + 1) / 2` maps [-1, 1] to [0, 1].

### Layer 2 — Consistency (weight 0.25)
`consistency_score` computes cosine similarity between the candidate's
**claims embedding** (title + headline + skills) and their **narrative embedding**.
A large mismatch flags resume inflation — e.g. a candidate who lists expert-level
NLP skills but whose job descriptions describe sales or support work.

### Layer 3 — Structured Rule Score (weight 0.15)
`structured_rule_score` applies deterministic heuristics:
- **YOE fit** (soft): full credit for 5–9 years; linear falloff outside the band.
- **Services-firm penalty**: if every employer is a known IT services firm (TCS,
  Infosys, Wipro, Accenture, Cognizant, etc.) with no product company.
- **CV/Robotics-only penalty**: if skills show only computer-vision/speech/robotics
  focus with zero NLP/IR signal.
- **Location boost/penalty**: Pune/Noida (+0.15), Hyderabad/Mumbai/Delhi-NCR (+0.10),
  outside India (-0.15, no visa sponsorship).

### Layer 4 — Honeypot Guard (exclude = 0.0)
`honeypot_guard` returns `True` (candidate excluded, score forced to 0.0) if:
- Any skill rated expert/advanced has `duration_months == 0` or missing.
- Any single career role's `duration_months` exceeds `years_of_experience * 12 + 12`.

### Layer 5 — Behavioral Modifier (multiplier in [0.6, 1.0])
`behavioral_modifier` is a multiplicative factor built from platform signals:
recruiter response rate, days since last activity, open-to-work flag, and
profile completeness.  It can never zero out a good candidate — it scales
from 0.6 (stale/unresponsive) to 1.0 (active/responsive/open/complete).

### Layer 6 — Combine Score
```
final = (0.60 * semantic_fit + 0.25 * consistency + 0.15 * structured_rule)
        * behavioral_modifier
```
Honeypot candidates return 0.0 immediately.

---

## Caching

All embeddings are precomputed once and stored under `data/cache/`:

| File                          | Description                          |
|-------------------------------|--------------------------------------|
| `candidate_ids.json`          | Ordered list of embedded candidate IDs |
| `candidate_claims_emb.npy`    | (N, 384) float32, L2-normalised      |
| `candidate_narrative_emb.npy` | (N, 384) float32, L2-normalised      |
| `jd_emb.npy`                  | (384,) float32, L2-normalised        |

`embed.py` is **incremental**: re-running it only encodes candidates whose IDs
are not yet in `candidate_ids.json`.  The `data/cache/` directory is git-ignored.

The MiniLM model (~80 MB) is downloaded once into `./model_cache/`.  At ranking
time, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set so no network
call is attempted.

---

## Constraints

- CPU only, no GPU required.
- No API calls at ranking time.
- Peak RAM < 16 GB (100K x 384-dim float32 = ~150 MB per embedding matrix).
- Full 100K run target: <= 5 min wall-clock.
