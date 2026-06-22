"""
score.py — 6-layer scoring for the candidate ranking pipeline.

Architecture:
  1. semantic_fit_score  — narrative vs JD cosine similarity
  2. consistency_score   — claims vs narrative cosine similarity
  3. structured_rule_score — deterministic heuristics (YOE, company type, location, skills)
  4. honeypot_guard       — True if candidate looks fabricated; returns 0.0 from combine_score
  5. behavioral_modifier  — platform-signal multiplier [0.6, 1.0]
  6. combine_score        — final weighted combination

Cosine similarity note:
  All embeddings are L2-normalised (done in embed.py), so cosine = dot product.
  Raw clamped cosine: max(0.0, cos) — values below 0 are floored to 0.

Layer weights (in combine_score):
  W_SEMANTIC  = 0.40   -- direct JD narrative relevance
  W_STRUCTURED = 0.35  -- rules / heuristics (raised to carry more signal)
  W_CONSISTENCY = 0.25 -- penalise resume inflation
"""

import os
import sys
from datetime import date
from functools import lru_cache
from typing import Optional

import numpy as np

# ---- Weight constants -------------------------------------------------------
# Weights sum to 1.0.  Structured raised to 0.35 so deterministic rules
# (location, YOE, keyword-stuffer penalty) carry meaningful signal alongside
# embedding layers.
W_SEMANTIC    = 0.25   # narrative embedding vs JD — reduced after empirical testing
                        # showed embedding similarity alone unreliably ranked genuine
                        # retrieval/ranking fits below generic technical candidates
W_STRUCTURED  = 0.50   # deterministic heuristics — raised to primary signal after
                        # outperforming embeddings on known sanity-check candidates
W_CONSISTENCY = 0.25   # claims embedding vs narrative embedding

# ---- Reference date for recency calculations --------------------------------
REFERENCE_DATE = date(2026, 6, 20)

# ---- Services-firm blocklist -----------------------------------------------
SERVICES_FIRMS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'hcl', 'tech mahindra', 'mindtree', 'mphasis',
}

# ---- CV/speech/robotics keywords (disqualify if no NLP/IR signal) ----------
CV_KEYWORDS = {
    'opencv', 'image classification', 'yolo', 'cnn', 'gans',
    'speech recognition', 'tts', 'robotics',
}
NLP_IR_KEYWORDS = {
    'nlp', 'information retrieval', 'embeddings', 'retrieval', 'search',
    'ranking', 'recommendation', 'transformers', 'sentence transformers',
    'semantic search', 'vector search', 'faiss', 'pinecone',
}

# ---- AI/ML/retrieval terms for keyword-stuffer rule ------------------------
# Used in structured_rule_score to detect non-technical titles claiming heavy
# advanced AI expertise (a documented red flag in the JD spec).
AI_ML_TERMS = {
    'embeddings', 'embedding', 'retrieval', 'ranking', 'recommendation',
    'recommendation systems', 'information retrieval', 'faiss', 'pinecone',
    'weaviate', 'qdrant', 'milvus', 'sentence transformers', 'transformers',
    'hugging face', 'nlp', 'llm', 'llms', 'fine-tuning llms', 'lora', 'qlora',
    'rag', 'semantic search', 'vector', 'machine learning', 'deep learning',
    'neural', 'pytorch', 'tensorflow', 'scikit-learn', 'mlops', 'mlflow',
    'learning to rank', 'bm25', 'cnn', 'gan', 'gans', 'diffusion',
}

GENERIC_AI_BUZZWORDS = {
    'ai', 'a.i.', 'artificial intelligence', 'ai/ml', 'ai tools', 'chatgpt',
    'gpt', 'genai', 'generative ai', 'ai-assisted', 'ai capabilities',
    'ai-powered', 'machine learning', 'ml',
}

RETRIEVAL_ML_CORE_TERMS = {
    'embeddings', 'embedding', 'retrieval', 'information retrieval', 'ranking',
    'recommendation', 'recommendation systems', 'recommender', 'faiss',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'vector search', 'vector database',
    'semantic search', 'hybrid search', 'bm25', 'learning to rank', 'ndcg',
    'mrr', 'map', 'sentence transformers', 'rag',
}

# Technical title qualifiers: a title is TECHNICAL if it contains any of these words,
# OR if it contains 'analyst' together with a tech qualifier.
_TECHNICAL_TITLE_WORDS = {'engineer', 'developer', 'scientist', 'architect', 'researcher'}
_ANALYST_TECH_QUALIFIERS = {'data', 'ml', 'machine learning', 'ai', 'quantitative'}

# ---- Location tables -------------------------------------------------------
PUNE_NOIDA_KEYWORDS   = {'pune', 'noida'}
MEDIUM_CITY_KEYWORDS  = {
    'hyderabad', 'mumbai', 'delhi', 'gurgaon', 'gurugram', 'ghaziabad', 'faridabad',
}


# ============================================================
# Cache loader (lazy, loaded once per Python process)
# ============================================================

_cache = None

def _get_cache():
    """Load embedding cache once and keep in memory."""
    global _cache
    if _cache is None:
        sys.path.insert(0, os.path.dirname(__file__))
        from embed import load_cache
        ids, claims, narrative, jd, id_to_row = load_cache()
        _cache = {
            'ids': ids,
            'claims': claims,
            'narrative': narrative,
            'jd': jd,
            'id_to_row': id_to_row,
        }
    return _cache


def _get_embeddings(candidate: dict):
    """Return (claims_vec, narrative_vec, jd_vec) for a candidate."""
    cache = _get_cache()
    cid = candidate['candidate_id']
    row = cache['id_to_row'].get(cid)
    if row is None:
        raise KeyError(f"Candidate {cid} not in embedding cache. Run embed.py first.")
    return cache['claims'][row], cache['narrative'][row], cache['jd']


def _cos_clamp(cos: float) -> float:
    """Clamp raw cosine similarity to [0, 1]: max(0.0, cos).
    Embeddings are L2-normalised so cosine = dot product, range [-1, 1].
    Negative cosines (opposite-direction vectors) are treated as zero signal.
    """
    return max(0.0, float(cos))


# ============================================================
# Layer 1 — Semantic Fit
# ============================================================

def semantic_fit_score(candidate: dict) -> float:
    """
    Cosine similarity between the candidate's narrative_emb and the JD embedding.
    Uses raw clamped cosine: max(0.0, cos) — negative similarities are zeroed.
    Higher = narrative content aligns with the JD role.
    """
    _, narrative_vec, jd_vec = _get_embeddings(candidate)
    cos = float(np.dot(narrative_vec, jd_vec))
    return _cos_clamp(cos)


# ============================================================
# Layer 2 — Consistency
# ============================================================

def consistency_score(candidate: dict) -> float:
    """
    Cosine similarity between claims_emb and narrative_emb.
    Uses raw clamped cosine: max(0.0, cos) — negative similarities are zeroed.
    Low similarity → claims don't match history → lower score (resume inflation red flag).
    """
    claims_vec, narrative_vec, _ = _get_embeddings(candidate)
    cos = float(np.dot(claims_vec, narrative_vec))
    return _cos_clamp(cos)


# ============================================================
# Layer 3 — Structured Rule Score
# ============================================================

def structured_rule_score(candidate: dict) -> float:
    """
    Deterministic heuristic score in [0, 1].

    Rules applied (additive/subtractive from a 0.5 base):
      1. YOE fit to 5–9 band: soft scoring — full credit inside, linear falloff outside.
      2. Services-firm-only penalty: if every employer is a known IT services firm.
      3. CV/speech/robotics-only penalty: if skills/narrative show only CV/robotics with
         zero NLP/IR signal.
      3b. Keyword-stuffer penalty: non-technical title + 3+ advanced/expert AI/ML skills.
      4. Location boost/penalty:
           Pune/Noida         → +0.15
           Hyderabad/Mumbai/Delhi-NCR → +0.10
           Outside India      → −0.15 (no visa sponsorship)
    """
    BASE = 0.5
    score = BASE

    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])

    # --- 1. YOE soft scoring ---
    yoe = float(profile.get('years_of_experience', 0) or 0)
    YOE_LOW, YOE_HIGH = 5.0, 9.0
    YOE_MAX_DELTA = 0.15  # max contribution from YOE

    if YOE_LOW <= yoe <= YOE_HIGH:
        yoe_score = 1.0
    elif yoe < YOE_LOW:
        # Linear falloff from 5 down to 0: at 0 yrs → score = 0
        yoe_score = max(0.0, yoe / YOE_LOW)
    else:
        # yoe > 9: gentle falloff — at 15+ years score drops to 0
        yoe_score = max(0.0, 1.0 - (yoe - YOE_HIGH) / 6.0)

    # Map [0, 1] yoe_score to a contribution centered at 0:
    # perfect band → +0.15; worst → -0.15
    score += (yoe_score - 0.5) * (2 * YOE_MAX_DELTA)

    # --- 2. Services-firm-only penalty ---
    companies = [r.get('company', '').lower() for r in career]
    is_services = [
        any(sf in comp for sf in SERVICES_FIRMS)
        for comp in companies
    ]
    if companies and all(is_services):
        score -= 0.10  # all employers are services firms

    # --- 3. CV/speech/robotics only penalty ---
    # Look at skill names and narrative text
    from build_evidence_text import narrative_text as get_narrative
    narrative = get_narrative(candidate).lower()
    skill_names_lower = {s.get('name', '').lower() for s in skills}
    all_text_tokens = skill_names_lower | set(narrative.split())

    has_cv_signal = bool(CV_KEYWORDS & all_text_tokens) or any(
        kw in narrative for kw in CV_KEYWORDS
    )
    has_nlp_signal = bool(NLP_IR_KEYWORDS & all_text_tokens) or any(
        kw in narrative for kw in NLP_IR_KEYWORDS
    )
    # Also check skill names explicitly (multi-word keywords)
    skill_text = ' '.join(s.get('name', '').lower() for s in skills)
    if not has_nlp_signal:
        has_nlp_signal = any(kw in skill_text for kw in NLP_IR_KEYWORDS)

    if has_cv_signal and not has_nlp_signal:
        score -= 0.10  # pure CV/speech/robotics, no NLP/IR signal

    # --- 3b. Narrative AI-vocabulary inflation penalty ---
    # Catches candidates whose narrative/summary leans on GENERIC AI buzzwords
    # ("ai tools", "chatgpt", "ai/ml") without any SPECIFIC technical vocabulary
    # (embeddings, retrieval, transformers, etc.) or skills backing. This is the
    # "content writer talking about AI" pattern — distinct from a skills-list stuffer.
    title_lower = profile.get('current_title', '').lower()
 
    is_technical_title = any(word in title_lower for word in _TECHNICAL_TITLE_WORDS)
    if not is_technical_title and 'analyst' in title_lower:
        is_technical_title = any(q in title_lower for q in _ANALYST_TECH_QUALIFIERS)
 
    generic_ai_hits = sum(1 for term in GENERIC_AI_BUZZWORDS if term in narrative)
    specific_ai_hits = sum(1 for term in AI_ML_TERMS if term in narrative)
 
    skill_ai_terms_backed = {
        s.get('name', '').lower() for s in skills
        if s.get('name', '').lower() in AI_ML_TERMS
        and s.get('proficiency', '').lower() in ('intermediate', 'advanced', 'expert')
    }
 
    if not is_technical_title:
        # Generic AI talk, no real technical depth, no skills backing → strong stuffer signal
        if generic_ai_hits >= 2 and specific_ai_hits <= 1 and len(skill_ai_terms_backed) == 0:
            score -= 0.25
        elif generic_ai_hits >= 1 and specific_ai_hits == 0:
            score -= 0.10
    else:
        # Technical title but narrative is still generic-AI-heavy with no real depth
        if generic_ai_hits >= 3 and specific_ai_hits == 0 and len(skill_ai_terms_backed) == 0:
            score -= 0.10
 
    # Original skills-only stuffer check, kept as a secondary signal:
    # non-technical title + 3+ advanced/expert AI/ML skills.
    if not is_technical_title:
        strong_ai_skill_count = sum(
            1 for s in skills
            if s.get('proficiency', '').lower() in ('advanced', 'expert')
            and s.get('name', '').lower() in AI_ML_TERMS
        )
        if strong_ai_skill_count >= 3:
            score -= 0.25

    # --- 4. Location boost/penalty ---
    location_lower = profile.get('location', '').lower()
    country = profile.get('country', '').strip()

    if country.lower() != 'india':
        score -= 0.15
    elif any(kw in location_lower for kw in PUNE_NOIDA_KEYWORDS):
        score += 0.15
    elif any(kw in location_lower for kw in MEDIUM_CITY_KEYWORDS):
        score += 0.10
    # Other Indian cities: no change from base

    # --- 5. Genuine retrieval/ranking/ML relevance boost ---
    # The JD's actual hard requirements are narrow and specific. Generic technical
    # titles (Frontend, Mobile, Java Dev, etc.) should NOT outscore a candidate with
    # real evidence of embeddings/retrieval/ranking work just because they clear the
    # YOE/location/title-format bar. This rule rewards specific relevant evidence.
    relevance_hits_narrative = sum(1 for term in RETRIEVAL_ML_CORE_TERMS if term in narrative)
 
    relevant_skills = [
        s for s in skills
        if s.get('name', '').lower() in RETRIEVAL_ML_CORE_TERMS
        and s.get('proficiency', '').lower() in ('intermediate', 'advanced', 'expert')
    ]
 
    relevance_title_words = {'recommendation', 'search', 'ranking', 'retrieval', 'nlp', 'ml', 'ai'}
    title_is_directly_relevant = any(w in title_lower for w in relevance_title_words)
 
    relevance_boost = 0.0
    if relevance_hits_narrative >= 2 and len(relevant_skills) >= 1:
        relevance_boost = 0.25
    elif relevance_hits_narrative >= 1 or len(relevant_skills) >= 1:
        relevance_boost = 0.12
    if title_is_directly_relevant:
        relevance_boost += 0.10
 
    score += relevance_boost

    return max(0.0, min(1.0, score))

# ============================================================
# Layer 4 — Honeypot Guard
# ============================================================

def honeypot_guard(candidate: dict) -> bool:
    """
    Returns True (honeypot — exclude this candidate) if:
      (a) Any skill has proficiency 'expert' or 'advanced' with
          duration_months == 0 or None/missing.
      (b) Any single career_history entry's duration_months exceeds
          years_of_experience * 12 by more than 12 months of slack.

    Returns False for legitimate candidates.
    """
    profile = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])

    yoe = float(profile.get('years_of_experience', 0) or 0)
    yoe_months = yoe * 12

    # Check (a): expert/advanced with zero/missing duration
    for s in skills:
        prof = s.get('proficiency', '').lower()
        dur = s.get('duration_months')
        if prof in ('expert', 'advanced'):
            if dur is None or dur == 0:
                return True

    # Check (b): single role duration > total_yoe + 12 months slack
    for role in career:
        role_dur = role.get('duration_months')
        if role_dur is None:
            continue
        if role_dur > yoe_months + 12:
            return True

    return False


# ============================================================
# Layer 5 — Behavioral Modifier
# ============================================================

def behavioral_modifier(candidate: dict) -> float:
    """
    Platform-signal multiplier in [0.6, 1.0].

    Inputs (from redrob_signals):
      - recruiter_response_rate [0, 1]
      - last_active_date recency (relative to 2026-06-20)
      - open_to_work_flag (bool)
      - profile_completeness_score [0, 100]

    Formula (all components equal weight 0.25 each, combined → [0, 1],
    then scaled to [0.6, 1.0]):
      raw = 0.25*response + 0.25*recency + 0.25*open + 0.25*completeness
      modifier = 0.6 + 0.4 * raw    (floors at 0.6, ceilings at 1.0)
    """
    rs = candidate.get('redrob_signals', {})

    # --- Response rate ---
    response = float(rs.get('recruiter_response_rate', 0.0) or 0.0)
    response = max(0.0, min(1.0, response))

    # --- Recency --- last_active_date days since reference date
    last_active_str = rs.get('last_active_date', '')
    if last_active_str:
        try:
            last_date = date.fromisoformat(last_active_str)
            days_ago = (REFERENCE_DATE - last_date).days
            # Fresh: ≤30 days = 1.0; stale: ≥365 days = 0.0; linear between
            recency = max(0.0, 1.0 - days_ago / 365.0)
        except ValueError:
            recency = 0.5
    else:
        recency = 0.5

    # --- Open to work ---
    open_flag = rs.get('open_to_work_flag', False)
    open_score = 1.0 if open_flag else 0.3  # still possible, just lower signal

    # --- Profile completeness ---
    completeness_raw = float(rs.get('profile_completeness_score', 50.0) or 50.0)
    completeness = max(0.0, min(1.0, completeness_raw / 100.0))

    raw = 0.25 * response + 0.25 * recency + 0.25 * open_score + 0.25 * completeness
    modifier = 0.6 + 0.4 * raw
    return max(0.6, min(1.0, modifier))


# ============================================================
# Layer 6 — Combine Score
# ============================================================

def combine_score(candidate: dict) -> float:
    """
    Final composite score.

    If honeypot_guard fires → 0.0 (excluded from ranking).
    Otherwise:
      final = (W_SEMANTIC_FIT * semantic_fit
               + W_CONSISTENCY * consistency
               + W_STRUCTURED * structured_rule) * behavioral_modifier

    Returns float in [0.0, 1.0].
    """
    if honeypot_guard(candidate):
        return 0.0

    sf = semantic_fit_score(candidate)
    cs = consistency_score(candidate)
    sr = structured_rule_score(candidate)
    bm = behavioral_modifier(candidate)

    return (W_SEMANTIC * sf + W_CONSISTENCY * cs + W_STRUCTURED * sr) * bm


# ============================================================
# Inline test
# ============================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_candidates_sample

    TEST_IDS = ['CAND_0000031', 'CAND_0000001', 'CAND_0000026']
    sample = load_candidates_sample()
    cand_map = {c['candidate_id']: c for c in sample}

    print('=== score.py self-test ===')
    print()
    header = (
        f"{'candidate':<15} | {'sem_fit':>8} | {'consist':>8} | "
        f"{'struc_rule':>10} | {'honeypot':>8} | {'beh_mod':>8} | {'combine':>8}"
    )
    print(header)
    print('-' * len(header))

    for cid in TEST_IDS:
        cand = cand_map[cid]
        sf  = semantic_fit_score(cand)
        cs  = consistency_score(cand)
        sr  = structured_rule_score(cand)
        hg  = honeypot_guard(cand)
        bm  = behavioral_modifier(cand)
        cm  = combine_score(cand)

        print(
            f"{cid:<15} | {sf:>8.4f} | {cs:>8.4f} | {sr:>10.4f} | "
            f"{'TRUE' if hg else 'false':>8} | {bm:>8.4f} | {cm:>8.4f}"
        )
