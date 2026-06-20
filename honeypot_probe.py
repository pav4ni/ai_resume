import json
PATH="data/candidates.jsonl"
expert_zero=[]; tenure_gt=[]; both=[]
with open(PATH,encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        c=json.loads(line); p=c["profile"]; yoe=p.get("years_of_experience") or 0
        ch=c.get("career_history",[]); skills=c.get("skills",[])
        ez=[s for s in skills if s.get("proficiency") in ("expert","advanced") and s.get("duration_months")==0]
        tg=[j for j in ch if (j.get("duration_months") or 0) > yoe*12+12]
        if ez: expert_zero.append((c,ez))
        if tg: tenure_gt.append((c,tg))
        if ez and tg: both.append(c["candidate_id"])

print(f"expert/advanced 0-month skills: {len(expert_zero)} candidates")
print(f"single-job tenure > career   : {len(tenure_gt)} candidates")
print(f"BOTH conditions              : {len(both)} -> {both}")

print("\n--- expert/advanced-with-0-months candidates (count of such skills each) ---")
for c,ez in expert_zero[:25]:
    p=c["profile"]
    names=", ".join(f"{s['name']}({s['proficiency']})" for s in ez[:6])
    print(f"  {c['candidate_id']} | {p['current_title']:24s} | yoe {p['years_of_experience']:.1f} | {len(ez)} zero-skills: {names}")

print("\n--- single-job-tenure > total-career candidates ---")
for c,tg in tenure_gt[:25]:
    p=c["profile"]; yoe=p['years_of_experience']
    j=max(tg,key=lambda x:x.get('duration_months',0))
    print(f"  {c['candidate_id']} | {p['current_title']:24s} | yoe {yoe:.1f} | job '{j['title']}' @ {j['company']} = {j['duration_months']}mo ({j['duration_months']/12:.1f}y)")

# count how many skills marked expert/advanced with 0 months, distribution
from collections import Counter
cnt=Counter(len(ez) for _,ez in expert_zero)
print("\ndistribution of #zero-month expert/advanced skills per flagged candidate:", dict(sorted(cnt.items())))
