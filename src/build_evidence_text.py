"""
build_evidence_text.py — Convert a candidate dict into two distinct text representations.

Two-channel design:
  claims_text   — what the candidate SAYS they are (title, headline, skill names).
  narrative_text — what their history SHOWS (summary + role descriptions).

Keeping them separate lets the scorer detect mismatches (consistency_score) and
measure semantic alignment with the JD independently for each channel.
"""


def claims_text(candidate: dict) -> str:
    """
    Build the "claims" channel: structured assertions about the candidate.

    Sources (all from the candidate dict):
      - profile.current_title
      - profile.headline
      - skills[].name + proficiency (e.g. "NLP (advanced)")

    Returns:
        A single string concatenating these fields.
    """
    parts = []

    profile = candidate.get('profile', {})
    if profile.get('current_title'):
        parts.append(profile['current_title'])
    if profile.get('headline'):
        parts.append(profile['headline'])

    skills = candidate.get('skills', [])
    skill_strs = []
    for s in skills:
        name = s.get('name', '').strip()
        prof = s.get('proficiency', '').strip()
        if name:
            skill_strs.append(f"{name} ({prof})" if prof else name)
    if skill_strs:
        parts.append(', '.join(skill_strs))

    return '. '.join(parts)


def narrative_text(candidate: dict) -> str:
    """
    Build the "narrative" channel: what the work history actually demonstrates.

    Sources:
      - profile.summary
      - career_history[].description (all roles, space-separated)

    Returns:
        A single string concatenating these fields.
    """
    parts = []

    profile = candidate.get('profile', {})
    if profile.get('summary'):
        parts.append(profile['summary'].strip())

    for role in candidate.get('career_history', []):
        desc = role.get('description', '').strip()
        if desc:
            parts.append(desc)

    return ' '.join(parts)


def build_evidence(candidate: dict) -> dict:
    """
    Return both text channels for a candidate.

    Returns:
        {'claims_text': str, 'narrative_text': str}
    """
    return {
        'claims_text': claims_text(candidate),
        'narrative_text': narrative_text(candidate),
    }


# ---- inline test ---------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_candidates_sample

    TEST_IDS = {'CAND_0000031', 'CAND_0000001', 'CAND_0000026'}
    sample = load_candidates_sample()

    print('=== build_evidence_text self-test ===')
    for cand in sample:
        if cand['candidate_id'] not in TEST_IDS:
            continue
        ev = build_evidence(cand)
        print(f"\n--- {cand['candidate_id']} ({cand['profile']['current_title']}) ---")
        print(f"CLAIMS  ({len(ev['claims_text'])} chars):\n  {ev['claims_text'][:300]}")
        print(f"NARRATIVE ({len(ev['narrative_text'])} chars):\n  {ev['narrative_text'][:300]}")
