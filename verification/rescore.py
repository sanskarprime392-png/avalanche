"""Re-score the 100 reviewed chips against a stated rule, and report the rater disagreement.

Rule — a detection counts as VERIFIED only if all three hold:
  (i)   snow is present on the slope in the BEFORE image (a bare slope has nothing to release);
  (ii)  a localized, rough, darker feature appears inside the outline in AFTER;
  (iii) the surrounding slope is essentially unchanged between the two dates.
Frames too cloudy or too washed out to assess are UNJUDGEABLE and excluded, not counted as failures.
"""
import json, os, math, glob, collections
import pandas as pd

D = r"C:\Users\Sanskar\Documents\avalanche_data"
VD = (r"C:\Users\Sanskar\AppData\Local\Temp\claude\C--Users-Sanskar-Documents-avalanche"
      r"\a470dfce-a258-4b98-9552-15641e0024c7\scratchpad\verdicts\verdicts")

rater1 = {}
for f in glob.glob(os.path.join(VD, "*.json")):
    d = json.load(open(f))
    rater1[int(os.path.basename(f)[:-5])] = d.get("data", d)["verdict"]

# second pass: the 24 chips scored as avalanche, re-examined against the rule above
fail_no_snow = [2, 17, 41, 51, 55, 64, 80, 92, 95, 96]      # (i) fails: bare slope in BEFORE
fail_scene   = [19, 21, 39, 47, 70, 73, 84, 94]             # (iii) fails: whole frame changed
unjudgeable  = [33]                                          # BEFORE blank / cloud
no_change    = [20, 67]                                      # nothing new inside the outline
keep         = [11, 42, 57]                                  # all three criteria met

rater2 = dict(rater1)
for i in fail_no_snow + fail_scene + no_change:
    rater2[i] = "meltout"
for i in unjudgeable:
    rater2[i] = "unclear"

def wilson(k, n, z=1.96):
    if not n: return (0.0, 0.0)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0.0, c-h), min(1.0, c+h))

rows = []
for name, R in (("first pass (unaided)", rater1), ("rule-based re-scoring", rater2)):
    c = collections.Counter(R.values())
    a, m, u = c["avalanche"], c["meltout"], c["unclear"]
    lo, hi = wilson(a, a+m)
    rows.append(dict(scoring=name, verified=a, rejected=m, unjudgeable=u,
                     n_judged=a+m, precision=a/(a+m), ci_low=lo, ci_high=hi))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(D, "results", "precision_verification.csv"), index=False)
print(res.round(3).to_string(index=False))

agree = sum(1 for i in rater1 if rater1[i] == rater2[i])
print(f"\nagreement between the two scorings: {agree}/100 ({agree}%)")
# Cohen's kappa between the two passes
cats = ["avalanche", "meltout", "unclear"]
n = len(rater1)
po = agree / n
p1 = collections.Counter(rater1.values()); p2 = collections.Counter(rater2.values())
pe = sum((p1[c]/n) * (p2[c]/n) for c in cats)
print(f"Cohen's kappa (rater agreement): {(po-pe)/(1-pe):.3f}")

json.dump({str(k): v for k, v in rater2.items()},
          open(os.path.join(D, "verification", "verdicts_rulebased.json"), "w"), indent=1)
json.dump({str(k): v for k, v in rater1.items()},
          open(os.path.join(D, "verification", "verdicts_firstpass.json"), "w"), indent=1)
print("\nsaved both scorings to avalanche_data/verification/")
