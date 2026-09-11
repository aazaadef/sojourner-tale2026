#!/usr/bin/env python3
"""
Replication Analysis Script
Paper: Teaching Software Testing and Debugging Through a Serious Game:
       An Empirical Classroom Study with Sojourner under Sabotage
Authors: Aazaade Faraji, Enrico Nunes, Francisco Reis, Nuno Pombo,
         Christina Andersson

This script reproduces all statistical results reported in the paper using:
  - data_pre_questionnaire.csv   (pre-session questionnaire, N=22)
  - data_post_questionnaire.csv  (post-session questionnaire, N=21)
  - data_telemetry.json          (in-game behavioral telemetry)

All inferential statistics are computed with SciPy. Every test reports the
test statistic, the handling of zero differences and ties, whether the p-value
is exact, permutation-based or asymptotic, and the effect-size formula, so the
numbers in the paper can be traced to a specific procedure.

Usage:
  pip install scipy matplotlib numpy
  python analysis.py
"""

import csv, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths — adjust if running from a different directory
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Everything the script needs sits next to it, so the package runs standalone.
# telemetry.json is the pseudonymized export; see resolve_user below.
PRE_CSV = BASE / "data_pre_questionnaire.csv"
POST_CSV = BASE / "data_post_questionnaire.csv"
TELE_JSON = BASE / "telemetry.json"

# ---------------------------------------------------------------------------
# Correct answers for 11 MCQ items
# ---------------------------------------------------------------------------
CORRECT_PRE = {
    23: "Testing",
    26: "Debugging",
    29: "Testing a single function in isolation from other components",
    32: "Testing interactions between components",
    35: "Black-box testing",
    38: "Designing tests based on internal code logic",
    41: "Verifying that login works correctly",
    44: "Mutation testing",
    47: "assertEquals(2, sum(1,1))",
    50: "Investigate whether the failure is due to a bug in the code or the test",
    53: "The tests are not effective enough to detect the bug"
}
CORRECT_POST = {
    11: "Testing",
    14: "Debugging",
    17: "Testing a single function in isolation from other components",
    20: "Testing interactions between components",
    23: "Black-box testing",
    26: "Designing tests based on internal code logic",
    29: "Verifying that login works correctly",
    32: "Mutation testing",
    35: "assertEquals(2, sum(1,1))",
    38: "Investigate whether the failure is due to a bug in the code or the test",
    41: "The tests are not effective enough to detect the bug"
}

# Participant code for post row (8305C treated as 8305J — same student, typo)
def normalize_code(code):
    code = code.strip().upper()
    return "8305J" if code == "8305C" else code

# ---------------------------------------------------------------------------
# Load questionnaire data
# ---------------------------------------------------------------------------
def load_scores(path, answer_cols, code_col):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    scores = {}
    for r in rows[1:]:
        code = normalize_code(r[code_col])
        if not code:
            continue
        scores[code] = sum(1 for col, ans in answer_cols.items() if r[col].strip() == ans)
    return scores

pre_scores  = load_scores(PRE_CSV,  CORRECT_PRE,  5)
post_scores = load_scores(POST_CSV, CORRECT_POST, 8)

matched = sorted(set(pre_scores) & set(post_scores))
N = len(matched)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mean(vals): return sum(vals) / len(vals) if vals else 0

def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Zero differences are discarded (Wilcoxon's original method), so n below is
    the number of non-zero pairs. Two p-values are reported: an exact one
    obtained by enumerating all 2^n sign assignments of the observed
    (tie-averaged) ranks, and the normal approximation with the standard
    correction for tied ranks. The effect size is r = |Z| / sqrt(N) with N the
    number of *pairs*; r_nonzero uses n instead and is given for transparency.
    """
    nz = np.array([d for d in diffs if d != 0], dtype=float)
    n = len(nz)
    if n < 3:
        return dict(n=n, p_exact=None, p_approx=None, Z=None, r=None)

    ranks = stats.rankdata(np.abs(nz))
    W_plus = float(ranks[nz > 0].sum())
    W_minus = float(ranks[nz < 0].sum())
    T = min(W_plus, W_minus)

    mean_W = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(nz), return_counts=True)
    tie_corr = float(((counts ** 3) - counts).sum())
    var_W = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr / 48.0
    Z = (W_plus - mean_W) / math.sqrt(var_W)
    p_approx = 2.0 * stats.norm.sf(abs(Z))

    # exact null distribution over all 2^n sign assignments of observed ranks
    dist = {0.0: 1}
    for r_ in ranks:
        nd = defaultdict(int)
        for s, c in dist.items():
            nd[s] += c
            nd[s + r_] += c
        dist = nd
    total = float(sum(dist.values()))
    hi = n * (n + 1) / 2.0 - T
    p_exact = sum(c for s, c in dist.items() if s <= T or s >= hi) / total

    return dict(n=n, n_pairs=len(diffs), W_plus=W_plus, W_minus=W_minus, T=T,
                Z=Z, p_exact=p_exact, p_approx=p_approx,
                r=abs(Z) / math.sqrt(len(diffs)), r_nonzero=abs(Z) / math.sqrt(n))

def norm_gain(pre, post, max_=11):
    return (post - pre) / (max_ - pre) if pre < max_ else 0.0

def spearman(x, y, n_perm=100000, seed=20260911):
    """Spearman rank correlation with tie handling, permutation p and 95% CI.

    scipy.stats.spearmanr computes rho as Pearson's r on tie-averaged ranks,
    which is the correct estimator when ranks are tied. (The textbook shortcut
    1 - 6*sum(d^2)/(n(n^2-1)) is only valid without ties and overestimates rho
    here.) The asymptotic p-value comes from the t-approximation; a permutation
    test is also run because n is small. The CI uses the Fisher z transform.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    res = stats.spearmanr(x, y)
    rho, p_t = float(res.statistic), float(res.pvalue)

    def stat(a, b):
        return stats.spearmanr(a, b).statistic

    perm = stats.permutation_test((x, y), stat, permutation_type="pairings",
                                  alternative="two-sided", n_resamples=n_perm,
                                  random_state=seed)
    z = np.arctanh(rho)
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - 1.959964 * se), math.tanh(z + 1.959964 * se)
    return dict(rho=rho, n=n, p_t=p_t, p_perm=float(perm.pvalue), ci=(lo, hi))

def cronbach_alpha(item_vectors):
    """Cronbach's alpha for k items measured on the same respondents."""
    k = len(item_vectors)
    arr = np.asarray(item_vectors, dtype=float)
    var_items = arr.var(axis=1, ddof=1).sum()
    var_total = arr.sum(axis=0).var(ddof=1)
    if var_total == 0:
        return float("nan")
    return (k / (k - 1)) * (1 - var_items / var_total)

def describe(vals):
    """Mean, median and interquartile range for an ordinal/interval vector."""
    a = np.asarray(vals, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return dict(mean=a.mean(), median=float(np.median(a)), q1=float(q1), q3=float(q3))

# ---------------------------------------------------------------------------
# MCQ Analysis (Section IV.A / RQ1)
# ---------------------------------------------------------------------------
print("=" * 60)
print("MCQ KNOWLEDGE ASSESSMENT")
print("=" * 60)

pre_vals  = [pre_scores[c]  for c in matched]
post_vals = [post_scores[c] for c in matched]
deltas    = [post_vals[i] - pre_vals[i] for i in range(N)]
ng        = [norm_gain(pre_vals[i], post_vals[i]) for i in range(N)]
w         = wilcoxon(deltas)

d_pre, d_post = describe(pre_vals), describe(post_vals)
print(f"N matched pairs: {N}")
print(f"Pre  mean={d_pre['mean']:.2f}  median={d_pre['median']:.1f}  "
      f"IQR=[{d_pre['q1']:.1f}, {d_pre['q3']:.1f}]")
print(f"Post mean={d_post['mean']:.2f}  median={d_post['median']:.1f}  "
      f"IQR=[{d_post['q1']:.1f}, {d_post['q3']:.1f}]")
print(f"Delta mean: {mean(deltas):+.2f}")
print(f"Improved / Same / Declined: {sum(1 for d in deltas if d>0)} / "
      f"{sum(1 for d in deltas if d==0)} / {sum(1 for d in deltas if d<0)}")
print()
print("Wilcoxon signed-rank, two-sided (scipy + exact enumeration):")
print(f"  pairs N={w['n_pairs']}, zero differences dropped -> n={w['n']}")
print(f"  W+ = {w['W_plus']:.1f}   W- = {w['W_minus']:.1f}   T = min = {w['T']:.1f}")
print(f"  Z = {w['Z']:.3f} (tie-corrected variance)")
print(f"  p (exact, 2^{w['n']} sign assignments) = {w['p_exact']:.4f}   <-- report this")
print(f"  p (normal approximation)              = {w['p_approx']:.4f}")
print(f"  effect size r = |Z|/sqrt(N={w['n_pairs']}) = {w['r']:.3f}"
      f"   [r using n={w['n']}: {w['r_nonzero']:.3f}]")
print(f"Normalized gain (mean): {mean(ng):.3f}")

# Sub-groups
print("\nSub-group analysis:")
for label, cond in [("Low (<8)", lambda p: p<8), ("Mid (8-9)", lambda p: 8<=p<=9), ("High (>=10)", lambda p: p>=10)]:
    grp = [c for c in matched if cond(pre_scores[c])]
    if grp:
        gd  = [post_scores[c]-pre_scores[c] for c in grp]
        gng = [norm_gain(pre_scores[c], post_scores[c]) for c in grp]
        print(f"  {label}: N={len(grp)}, pre={mean([pre_scores[c] for c in grp]):.2f}, "
              f"post={mean([post_scores[c] for c in grp]):.2f}, "
              f"gain={mean(gd):+.2f}, norm_gain={mean(gng):.3f}")

# Per-item accuracy
print("\nPer-item accuracy (%):")
pre_rows_all  = list(csv.reader(open(PRE_CSV,  encoding="utf-8")))
post_rows_all = list(csv.reader(open(POST_CSV, encoding="utf-8")))
n_pre  = len(pre_rows_all) - 1
n_post = len(post_rows_all) - 1
for i, (col, ans) in enumerate(CORRECT_PRE.items()):
    pre_pct  = sum(1 for r in pre_rows_all[1:]  if r[col].strip() == ans) / n_pre  * 100
    post_col = list(CORRECT_POST.keys())[i]
    post_ans = list(CORRECT_POST.values())[i]
    post_pct = sum(1 for r in post_rows_all[1:] if r[post_col].strip() == post_ans) / n_post * 100
    print(f"  Q{i+1:2d}: pre={pre_pct:5.1f}%  post={post_pct:5.1f}%  delta={post_pct-pre_pct:+.1f}%")

# ---------------------------------------------------------------------------
# Self-Confidence (Section IV.A / RQ1)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SELF-CONFIDENCE")
print("=" * 60)

SC_PRE_COLS  = [56, 58, 60, 62]
SC_POST_COLS = [44, 46, 48, 50]
SC_LABELS    = ["Understand concepts", "Write test case", "Identify/debug errors", "Analyze test failures"]

def parse_likert(v):
    for i in range(1, 6):
        if v.strip().startswith(str(i)):
            return i
    return None

pre_sc  = {}
for r in pre_rows_all[1:]:
    code = normalize_code(r[5])
    pre_sc[code] = [parse_likert(r[c]) for c in SC_PRE_COLS]
post_sc = {}
for r in post_rows_all[1:]:
    code = normalize_code(r[8])
    post_sc[code] = [parse_likert(r[c]) for c in SC_POST_COLS]

sc_matched = [c for c in matched if all(pre_sc.get(c,[])) and all(post_sc.get(c,[]))]
for i, label in enumerate(SC_LABELS):
    pv = [pre_sc[c][i]  for c in sc_matched]
    qv = [post_sc[c][i] for c in sc_matched]
    dv = [qv[j]-pv[j] for j in range(len(pv))]
    up = sum(1 for d in dv if d>0)
    same = sum(1 for d in dv if d==0)
    down = sum(1 for d in dv if d<0)
    print(f"  {label:30s}: pre={mean(pv):.2f} post={mean(qv):.2f} delta={mean(dv):+.2f}  {up}/{same}/{down}")

sc_pre_vecs  = [[pre_sc[c][i]  for c in sc_matched] for i in range(4)]
sc_post_vecs = [[post_sc[c][i] for c in sc_matched] for i in range(4)]
print(f"\n  Cronbach alpha: pre={cronbach_alpha(sc_pre_vecs):.3f}  "
      f"post={cronbach_alpha(sc_post_vecs):.3f}   (n={len(sc_matched)})")

# ---------------------------------------------------------------------------
# Telemetry Analysis (Section IV.C / RQ3)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TELEMETRY ANALYSIS")
print("=" * 60)

# The published telemetry export is already pseudonymized: every account appears
# as its questionnaire participant code, accounts that registered under a real
# name were rewritten to their code, and instructor, admin and test accounts were
# removed along with all name and credential fields. Nothing is left to resolve.
def resolve_user(username):
    return username.upper()

tele = json.loads(TELE_JSON.read_text())

user_events     = defaultdict(list)
user_max_room   = defaultdict(int)
user_test_execs = defaultdict(int)

for e in tele:
    code = resolve_user(e["user"]["username"])
    if code is None:
        continue
    user_events[code].append(e)
    if e["eventType"] == "GameProgressionChangedEvent":
        prog = json.loads(e["json"]).get("progression", {})
        user_max_room[code] = max(user_max_room[code], prog.get("room", 0))
    elif e["eventType"] in ("test-executed", "TestExecutedEvent"):
        user_test_execs[code] += 1

# Spearman: progression vs MCQ gain (N=21 matched after mapping)
tele_matched = sorted(set(pre_scores) & set(post_scores) & set(user_max_room))
rooms   = [user_max_room[c]              for c in tele_matched]
mcq_dlt = [post_scores[c]-pre_scores[c] for c in tele_matched]
tests   = [user_test_execs.get(c, 0)    for c in tele_matched]

s1 = spearman(rooms, mcq_dlt)
s2 = spearman(rooms, tests)
rho1, p1 = s1["rho"], s1["p_perm"]
rho2, p2 = s2["rho"], s2["p_perm"]

print(f"N (pre+post+tele matched): {len(tele_matched)}")
for label, s in [("progression vs MCQ gain", s1), ("progression vs test executions", s2)]:
    print(f"\nSpearman: {label}  (n={s['n']})")
    print(f"  rho (tie-corrected)      = {s['rho']:.3f}")
    print(f"  p (t-approximation)      = {s['p_t']:.4f}")
    print(f"  p (permutation, 100k)    = {s['p_perm']:.4f}   <-- report this")
    print(f"  95% CI (Fisher z)        = [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]")

# ---------------------------------------------------------------------------
# Perception constructs (Section IV.B / RQ2, Table IV)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PERCEPTION CONSTRUCTS")
print("=" * 60)
print("'% agree' = share of responses of 4 or 5 on the 5-point scale.")
print("Construct % agree = mean of its items' individual % agree.\n")

CONSTRUCTS = {
    "Perceived Learning": [(52, "helped understand testing vs debugging"),
                           (54, "improved understanding of unit testing"),
                           (56, "better understand how tests detect bugs"),
                           (58, "improved overall understanding")],
    "Engagement":         [(60, "the activity was engaging"),
                           (62, "increased my motivation to learn"),
                           (64, "prefer this over traditional lectures")],
    "Usability":          [(66, "the game was easy to use"),
                           (68, "interface was clear and understandable"),
                           (70, "instructions were easy to follow")],
    "Cognitive Load":     [(72, "required significant mental effort"),
                           (74, "some parts of the activity were confusing"),
                           (76, "the difficulty level was appropriate")],
}

for cname, items in CONSTRUCTS.items():
    print(f"{cname}")
    vecs, item_pcts = [], []
    for col, label in items:
        vals = [parse_likert(r[col]) for r in post_rows_all[1:]]
        vals = [v for v in vals if v is not None]
        d = describe(vals)
        pct = 100.0 * sum(1 for v in vals if v >= 4) / len(vals)
        item_pcts.append(pct)
        vecs.append(vals)
        print(f"   {label:45s} mean={d['mean']:.2f}  med={d['median']:.1f}  "
              f"IQR=[{d['q1']:.1f},{d['q3']:.1f}]  agree={pct:.0f}%")
    per_person = [mean([v[i] for v in vecs]) for i in range(len(vecs[0]))]
    a = cronbach_alpha(vecs)
    print(f"   {'>> CONSTRUCT':45s} mean={mean(per_person):.2f}  "
          f"agree={mean(item_pcts):.0f}%  Cronbach alpha={a:.3f}"
          f"{'   <-- NOT a reliable scale; report items separately' if a < 0.7 else ''}\n")

# ---------------------------------------------------------------------------
# Generate Figures
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    # matplotlib defaults to Type 3 fonts in PDF output, which IEEE PDF eXpress
    # rejects. 42 emits TrueType instead.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    import numpy as np

    # Figure: Per-item MCQ accuracy
    pre_pcts  = []
    post_pcts = []
    for i, (col, ans) in enumerate(CORRECT_PRE.items()):
        pre_pcts.append(sum(1 for r in pre_rows_all[1:] if r[col].strip() == ans) / n_pre * 100)
        pc = list(CORRECT_POST.keys())[i]
        pa = list(CORRECT_POST.values())[i]
        post_pcts.append(sum(1 for r in post_rows_all[1:] if r[pc].strip() == pa) / n_post * 100)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = range(11)
    ax.bar([i-0.2 for i in x], pre_pcts,  0.35, label="Pre",  color="#4472C4", alpha=0.85)
    ax.bar([i+0.2 for i in x], post_pcts, 0.35, label="Post", color="#ED7D31", alpha=0.85)
    ax.set_ylabel("% Correct"); ax.set_xlabel("Question")
    ax.set_title("Per-item MCQ Accuracy")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"Q{i+1}" for i in range(11)], fontsize=7)
    ax.set_ylim(0, 110); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_mcq_peritem.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: Correlation scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, y, rho, p, ylabel, color, title in [
        (ax1, mcq_dlt, rho1, p1, "MCQ Score Change (post−pre)", "#4472C4", "(a) Progression vs. Knowledge Gain"),
        (ax2, tests,   rho2, p2, "Number of Test Executions",   "#ED7D31", "(b) Progression vs. Testing Activity"),
    ]:
        ax.scatter(rooms, y, c=color, s=60, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
        z = np.polyfit(rooms, y, 1)
        xl = np.linspace(min(rooms)-0.3, max(rooms)+0.3, 50)
        ax.plot(xl, np.polyval(z, xl), "--", color="#C0392B", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Max Room Reached", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        lbl = f"ρ = {rho:.3f}, p = {p:.3f}" if p >= 0.001 else f"ρ = {rho:.3f}, p < 0.001"
        ax.text(0.05, 0.95, lbl, transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F0FE" if color=="#4472C4" else "#FFF2CC",
                          edgecolor=color, alpha=0.9))
        ax.set_xlim(0, 8); ax.grid(alpha=0.2)
    if mcq_dlt: ax1.set_ylim(-4, 9); ax1.axhline(y=0, color="gray", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_correlation.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nFigures saved to figures/")
except ImportError:
    print("\nmatplotlib not installed — skipping figures. Run: pip install matplotlib numpy")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
