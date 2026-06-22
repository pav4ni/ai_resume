import sys
sys.path.insert(0, 'src')
from data_loader import load_candidates_sample
from score import combine_score

sample = load_candidates_sample()
results = []
for c in sample:
    title = c['profile'].get('current_title', '')
    results.append((c['candidate_id'], title, combine_score(c)))

results.sort(key=lambda x: -x[2])

print(f"{'rank':<5}{'candidate':<16}{'title':<30}{'score':>8}")
for i, (cid, title, sc) in enumerate(results, 1):
    print(f"{i:<5}{cid:<16}{title[:28]:<30}{sc:>8.4f}")
