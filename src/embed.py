"""
embed.py — Precompute and cache MiniLM embeddings for candidates + JD.

First run (network required):
  python src/embed.py --source sample
  This downloads all-MiniLM-L6-v2 (~80 MB) into ./model_cache and encodes
  the 50-candidate sample.  Subsequent runs skip the download and stay offline.

Ranking time:
  rank.py sets HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 before importing this
  module so no network is attempted during the scoring pipeline.

Cache layout under data/cache/:
  candidate_ids.json          — ordered list of embedded candidate_ids
  candidate_claims_emb.npy    — (N, 384) float32, L2-normalised
  candidate_narrative_emb.npy — (N, 384) float32, L2-normalised
  jd_emb.npy                  — (384,) float32, L2-normalised

Incremental caching:
  embed_candidates checks which ids are already in candidate_ids.json and only
  encodes the new ones, then appends and re-saves.
"""

import argparse
import json
import os

import numpy as np

# Cache directory (relative to project root, i.e. where you run from)
CACHE_DIR = os.path.join('data', 'cache')
IDS_PATH = os.path.join(CACHE_DIR, 'candidate_ids.json')
CLAIMS_PATH = os.path.join(CACHE_DIR, 'candidate_claims_emb.npy')
NARRATIVE_PATH = os.path.join(CACHE_DIR, 'candidate_narrative_emb.npy')
JD_PATH = os.path.join(CACHE_DIR, 'jd_emb.npy')

MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_CACHE = './model_cache'


def _load_model():
    """Load the SentenceTransformer model from the local cache folder."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)
    return model


def embed_candidates(candidates: list, batch_size: int = 256, checkpoint_every: int = 20) -> None:
    """
    Encode claims_text and narrative_text for all candidates and cache to disk,
    saving progress every `checkpoint_every` batches so an interruption never
    loses more than a few minutes of work.

    Only newly-seen candidate_ids are encoded; existing cache entries are kept.

    Args:
        candidates:        List of candidate dicts.
        batch_size:        Sentence-transformer encoding batch size.
        checkpoint_every:  Save to disk after this many batches.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from build_evidence_text import build_evidence

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load existing cache if available
    if os.path.exists(IDS_PATH):
        with open(IDS_PATH, encoding='utf-8') as f:
            cached_ids: list = json.load(f)
        cached_set = set(cached_ids)
        cached_claims = np.load(CLAIMS_PATH)
        cached_narrative = np.load(NARRATIVE_PATH)
    else:
        cached_ids = []
        cached_set = set()
        cached_claims = None
        cached_narrative = None

    # Filter to candidates not yet cached
    new_candidates = [c for c in candidates if c['candidate_id'] not in cached_set]
    print(f"Total candidates supplied: {len(candidates)}; "
          f"already cached: {len(candidates) - len(new_candidates)}; "
          f"new to embed: {len(new_candidates)}")

    if not new_candidates:
        print("0 new candidates to embed (all cached)")
        return

    model = _load_model()

    new_ids = [c['candidate_id'] for c in new_candidates]
    evidences = [build_evidence(c) for c in new_candidates]
    claims_texts = [e['claims_text'] for e in evidences]
    narrative_texts = [e['narrative_text'] for e in evidences]

    n = len(new_ids)
    num_batches = (n + batch_size - 1) // batch_size

    def _save_progress(up_to_idx: int):
        """Save everything encoded so far (both cached + newly done up to up_to_idx)."""
        done_ids = new_ids[:up_to_idx]
        done_claims = claims_buffer[:up_to_idx]
        done_narrative = narrative_buffer[:up_to_idx]

        if cached_claims is not None:
            all_claims = np.vstack([cached_claims, done_claims])
            all_narrative = np.vstack([cached_narrative, done_narrative])
        else:
            all_claims = done_claims
            all_narrative = done_narrative

        all_ids = cached_ids + done_ids

        with open(IDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_ids, f)
        np.save(CLAIMS_PATH, all_claims)
        np.save(NARRATIVE_PATH, all_narrative)
        print(f"  [checkpoint] saved {len(all_ids)} total candidates to disk")

    print(f"Encoding {n} candidates in {num_batches} batches "
          f"(batch_size={batch_size}, checkpoint every {checkpoint_every} batches)...")

    claims_buffer = np.zeros((n, 384), dtype=np.float32)
    narrative_buffer = np.zeros((n, 384), dtype=np.float32)

    for batch_num, start in enumerate(range(0, n, batch_size), start=1):
        end = min(start + batch_size, n)

        batch_claims = model.encode(
            claims_texts[start:end],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        batch_narrative = model.encode(
            narrative_texts[start:end],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        claims_buffer[start:end] = batch_claims
        narrative_buffer[start:end] = batch_narrative

        print(f"  batch {batch_num}/{num_batches} done ({end}/{n} candidates)")

        if batch_num % checkpoint_every == 0 or batch_num == num_batches:
            _save_progress(end)

    print(f"Cache updated: {len(cached_ids) + n} candidates total")


def embed_jd(jd_text: str) -> None:
    """
    Encode the JD and cache to data/cache/jd_emb.npy.  Skips if already cached.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(JD_PATH):
        print(f"JD embedding already cached at {JD_PATH} — skipping.")
        return

    model = _load_model()
    jd_emb = model.encode(
        [jd_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)[0]  # shape (384,)

    np.save(JD_PATH, jd_emb)
    print(f"JD embedding cached at {JD_PATH}; shape={jd_emb.shape}")


def load_cache() -> tuple:
    """
    Load all cached embeddings.

    Returns:
        (ids, claims_matrix, narrative_matrix, jd_vector, id_to_row)

        ids             — list[str]  of candidate_ids in row order
        claims_matrix   — np.ndarray (N, 384)
        narrative_matrix — np.ndarray (N, 384)
        jd_vector       — np.ndarray (384,)
        id_to_row       — dict[str, int]
    """
    with open(IDS_PATH, encoding='utf-8') as f:
        ids = json.load(f)
    claims    = np.load(CLAIMS_PATH)
    narrative = np.load(NARRATIVE_PATH)
    jd        = np.load(JD_PATH)
    id_to_row = {cid: i for i, cid in enumerate(ids)}
    return ids, claims, narrative, jd, id_to_row


# ---- CLI / self-test --------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_candidates_sample, load_candidates, load_job_description

    parser = argparse.ArgumentParser(description='Precompute candidate + JD embeddings')
    parser.add_argument(
        '--source',
        choices=['sample', 'full'],
        default='sample',
        help='sample: use data/sample_candidates.json (50 cands, fast); '
             'full: stream data/candidates.jsonl (100K, slow)',
    )
    args = parser.parse_args()

    if args.source == 'sample':
        print('Loading 50-candidate sample...')
        candidates = load_candidates_sample()
    else:
        print('Streaming full candidates.jsonl (100 K)...')
        candidates = load_candidates('data/candidates.jsonl')

    jd_text = load_job_description()

    # Embed JD first (cheap, skipped if cached)
    embed_jd(jd_text)

    # Embed candidates (incremental — skips already-cached ids)
    embed_candidates(candidates)

    # Verify by loading the cache
    ids, claims, narrative, jd, id_to_row = load_cache()
    print(f"\nCache verification:")
    print(f"  candidate_ids  : {len(ids)} entries")
    print(f"  claims_emb     : {claims.shape}")
    print(f"  narrative_emb  : {narrative.shape}")
    print(f"  jd_emb         : {jd.shape}")
