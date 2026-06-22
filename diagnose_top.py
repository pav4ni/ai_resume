import sys
sys.path.insert(0, 'src')
from data_loader import load_candidates_sample
from score import (
    semantic_fit_score, consistency_score, structured_rule_score,
    honeypot_guard, behavioral_modifier, combine_score,
    W_SEMANTIC, W_CONSISTENCY, W_STRUCTURED
)

sample = load_candidates_sample()
cand_map = {c['candidate_id']: c for c in sample}

print(f"Weights: semantic={W_SEMANTIC}  structured={W_STRUCTURED}  consistency={W_CONSISTENCY}")
print()

for cid in ['CAND_0000048', 'CAND_0000014', 'CAND_0000031']:
    c = cand_map[cid]
    sf = semantic_fit_score(c)
    cs = consistency_score(c)
    sr = structured_rule_score(c)
    hg = honeypot_guard(c)
    bm = behavioral_modifier(c)
    cm = combine_score(c)

    print(f"{cid}  ({c['profile'].get('current_title')})")
    print(f"  YOE: {c['profile'].get('years_of_experience')}  Location: {c['profile'].get('location')}  Country: {c['profile'].get('country')}")
    print(f"  semantic_fit={sf:.4f}  consistency={cs:.4f}  structured_rule={sr:.4f}  honeypot={hg}  behavioral={bm:.4f}")
    print(f"  weighted: sem={W_SEMANTIC*sf:.4f}  struc={W_STRUCTURED*sr:.4f}  cons={W_CONSISTENCY*cs:.4f}  -> sum*beh = {cm:.4f}")
    print()