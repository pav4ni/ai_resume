import json, re, statistics
from collections import Counter
from datetime import date

TODAY = date(2026, 6, 20)
PATH = "data/candidates.jsonl"

def pdate(s):
    try:
        y,m,d = map(int, s.split("-")); return date(y,m,d)
    except Exception:
        return None

# --- vocab ---
IR_EVIDENCE = [
    "recommendation system","recommender","ranking model","ranking layer","learning-to-rank",
    "learning to rank","retrieval","embedding","vector search","semantic search","search relevance",
    "information retrieval","nearest neighbor","bm25","elasticsearch","opensearch","recsys",
    "personalization","search engine","matching system","re-rank","reranking","ranker","ann index",
    "faiss","pinecone","relevance label","discovery feed","search product","ctr","ndcg"
]
AI_SKILLS = {
    "embeddings","faiss","pinecone","weaviate","qdrant","milvus","sentence transformers",
    "information retrieval","recommendation systems","learning to rank","semantic search",
    "hugging face transformers","rag","fine-tuning llms","transformers","elasticsearch","bm25",
    "nlp","vector databases","llm","llms","langchain","retrieval","embedding"
}
SERVICES = ["tcs","tata consultancy","infosys","wipro","accenture","cognizant","capgemini",
            "hcl","tech mahindra","mindtree","ltimindtree","mphasis","igate","syntel"]
NONTECH_TITLES = {"marketing manager","graphic designer","accountant","hr manager","customer support",
                  "operations manager","content writer","sales","business development","recruiter",
                  "project manager","civil engineer","mechanical engineer"}

CORE_CITIES = ["pune","noida"]
NCR_WELCOME = ["hyderabad","mumbai","delhi","gurgaon","gurugram","ghaziabad","faridabad"]

n=0
titles=Counter(); industries=Counter(); countries=Counter()
real_ir=0; kw_only=0; ai_skill_heavy=0; plain_lang=0
loc_core=0; loc_ncr=0; loc_other_india=0; loc_outside=0
services_only=0; services_any_current=0
# signals
resp=[]; complete=[]; o2w=0; gh_none=0; gh_pos=[]
last_active_buckets=Counter()
# honeypot-ish
salary_inv=0; expert_zero=0; skill_gt_career=0; tenure_gt_career=0
kwstuffer_nontech=0
ir_skill_counts=[]

with open(PATH,encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        c=json.loads(line); n+=1
        p=c["profile"]; sig=c["redrob_signals"]
        title=p["current_title"].strip(); titles[title]+=1
        industries[p["current_industry"].strip()]+=1
        country=p.get("country","").strip(); countries[country]+=1
        yoe=p.get("years_of_experience") or 0

        # text blob from summary + all career descriptions + titles
        ch=c.get("career_history",[])
        text=" ".join([p.get("summary","")] + [j.get("description","") for j in ch] +
                      [j.get("title","") for j in ch]).lower()
        has_ir_evidence = any(k in text for k in IR_EVIDENCE)

        skills=c.get("skills",[])
        sk_names=[s["name"].strip().lower() for s in skills]
        ai_hits=sum(1 for s in sk_names if s in AI_SKILLS)
        ir_skill_counts.append(ai_hits)
        if ai_hits>=3: ai_skill_heavy+=1

        if has_ir_evidence: real_ir+=1
        if ai_hits>=3 and not has_ir_evidence: kw_only+=1
        if has_ir_evidence and ai_hits<=1: plain_lang+=1

        # location
        loc=(p.get("location","")+" "+country).lower()
        if country.lower()!="india":
            loc_outside+=1
        elif any(ci in loc for ci in CORE_CITIES):
            loc_core+=1
        elif any(ci in loc for ci in NCR_WELCOME):
            loc_ncr+=1
        else:
            loc_other_india+=1

        # services
        comps=[ (j.get("company","") or "").lower() for j in ch ]
        cur=(p.get("current_company","") or "").lower()
        def is_svc(name): return any(s in name for s in SERVICES)
        if comps and all(is_svc(x) for x in comps):
            services_only+=1
        if is_svc(cur):
            services_any_current+=1

        # signals
        rr=sig.get("recruiter_response_rate")
        if rr is not None: resp.append(rr)
        pc=sig.get("profile_completeness_score")
        if pc is not None: complete.append(pc)
        if sig.get("open_to_work_flag"): o2w+=1
        gh=sig.get("github_activity_score")
        if gh==-1: gh_none+=1
        elif gh is not None: gh_pos.append(gh)
        la=pdate(sig.get("last_active_date",""))
        if la:
            d=(TODAY-la).days
            if d<=30: last_active_buckets["<=30d"]+=1
            elif d<=90: last_active_buckets["31-90d"]+=1
            elif d<=180: last_active_buckets["91-180d"]+=1
            elif d<=365: last_active_buckets["181-365d"]+=1
            else: last_active_buckets[">365d"]+=1

        # honeypot-ish
        sal=sig.get("expected_salary_range_inr_lpa") or {}
        if sal.get("min") is not None and sal.get("max") is not None and sal["min"]>sal["max"]:
            salary_inv+=1
        if any(s.get("proficiency") in ("expert","advanced") and s.get("duration_months")==0 for s in skills):
            expert_zero+=1
        if any((s.get("duration_months") or 0) > yoe*12+12 for s in skills):
            skill_gt_career+=1
        if any((j.get("duration_months") or 0) > yoe*12+12 for j in ch):
            tenure_gt_career+=1
        if ai_hits>=4 and title.lower() in NONTECH_TITLES:
            kwstuffer_nontech+=1

def show(counter,k=20):
    tot=sum(counter.values())
    for name,ct in counter.most_common(k):
        print(f"    {name:38s} {ct:6d}  {100*ct/tot:5.1f}%")

def dist(vals,label):
    vals=sorted(vals)
    if not vals: print(label,"(none)"); return
    print(f"  {label}: n={len(vals)} mean={statistics.mean(vals):.3f} median={statistics.median(vals):.3f} min={vals[0]:.3f} max={vals[-1]:.3f}")

print("="*70)
print("1. TOTAL CANDIDATES:", n)
print("="*70)
print("\n2a. TOP 20 current_title:")
show(titles)
print("\n2b. TOP 20 current_industry:")
show(industries)
print("\n3. RETRIEVAL/RANKING EXPERIENCE:")
print(f"    real IR/ranking evidence in career/summary text : {real_ir:6d}  ({100*real_ir/n:.1f}%)")
print(f"    AI-skill-heavy (>=3 AI skill names)             : {ai_skill_heavy:6d}  ({100*ai_skill_heavy/n:.1f}%)")
print(f"    keyword-only (>=3 AI skills, NO text evidence)  : {kw_only:6d}  ({100*kw_only/n:.1f}%)")
print(f"    plain-language fit (text evidence, <=1 AI skill): {plain_lang:6d}  ({100*plain_lang/n:.1f}%)")
print(f"    avg AI-skill-name hits per candidate            : {statistics.mean(ir_skill_counts):.2f}")
print("\n4. KEY SIGNAL DISTRIBUTIONS:")
dist(resp,"recruiter_response_rate")
print("     buckets:", {b:sum(1 for x in resp if lo<=x<hi) for b,(lo,hi) in
      {"0-0.1":(0,0.1),"0.1-0.3":(0.1,0.3),"0.3-0.5":(0.3,0.5),"0.5-0.7":(0.5,0.7),"0.7-1.01":(0.7,1.01)}.items()})
dist(complete,"profile_completeness_score")
print(f"  open_to_work_flag TRUE: {o2w} ({100*o2w/n:.1f}%)")
print(f"  github_activity_score: none(-1)={gh_none} ({100*gh_none/n:.1f}%); linked n={len(gh_pos)}", end="")
if gh_pos: print(f" mean={statistics.mean(gh_pos):.1f} median={statistics.median(gh_pos):.1f}")
else: print()
print("  last_active recency buckets:", dict(last_active_buckets))
print("\n5. LOCATION SPREAD:")
print(f"    Pune/Noida (core)        : {loc_core:6d}  ({100*loc_core/n:.1f}%)")
print(f"    Other NCR/Hyd/Mumbai     : {loc_ncr:6d}  ({100*loc_ncr/n:.1f}%)")
print(f"    Other India              : {loc_other_india:6d}  ({100*loc_other_india/n:.1f}%)")
print(f"    Outside India            : {loc_outside:6d}  ({100*loc_outside/n:.1f}%)")
print("    Top countries:")
show(countries,8)
print("\n6. SERVICES-FIRM CAREERS:")
print(f"    every company in history is a services firm: {services_only:6d}  ({100*services_only/n:.1f}%)")
print(f"    currently AT a services firm               : {services_any_current:6d}  ({100*services_any_current/n:.1f}%)")
print("\n7. HONEYPOT / TRAP-PATTERN COUNTS (whole pool):")
print(f"    salary min>max                              : {salary_inv:6d}  ({100*salary_inv/n:.1f}%)")
print(f"    expert/advanced skill w/ 0 months used      : {expert_zero:6d}  ({100*expert_zero/n:.1f}%)")
print(f"    a skill used > career length (+12mo slack)  : {skill_gt_career:6d}  ({100*skill_gt_career/n:.1f}%)")
print(f"    single-job tenure > total career (+12mo)    : {tenure_gt_career:6d}  ({100*tenure_gt_career/n:.1f}%)")
print(f"    keyword-stuffer (>=4 AI skills, nontech title): {kwstuffer_nontech:6d}  ({100*kwstuffer_nontech/n:.1f}%)")
