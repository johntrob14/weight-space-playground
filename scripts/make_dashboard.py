"""Build results/dashboard.html from the result files (static SVG charts, no JavaScript).

    python scripts/make_dashboard.py [--check]     --check also writes results/dashboard_check.svg with literal
                                                   colours so the layout can be rasterised and inspected
"""
import html
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: os.path.join(ROOT, "results", p)  # noqa: E731
LIGHT = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#6250d6", "ink": "#0b0b0b", "ink2": "#52514e",
         "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb", "page": "#f9f9f7"}
DARK = {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#9085e9", "ink": "#f0efec", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "surface": "#1a1a19", "page": "#0d0d0d"}
LITERAL = "--check" in sys.argv


def col(k):
    return LIGHT[k] if LITERAL else f"var(--{k})"


def bars(rows, series, xmax, unit="", width=980, label_w=330, row_h=None, whisk=None):
    """Horizontal grouped bars. rows: [(label, [v1, v2, ...], [tip1, ...])]; series: [(name, colour key)]."""
    ns = len(series)
    bar_h = 20
    row_h = row_h or ns * (bar_h + 2) + 20
    top, bottom, right = 44 if ns > 1 else 12, 40, 60
    H = top + row_h * len(rows) + bottom
    pw = width - label_w - right
    x = lambda v: label_w + pw * v / xmax  # noqa: E731
    o = [f'<svg viewBox="0 0 {width} {H}" role="img" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui, -apple-system, Segoe UI, sans-serif">']
    if ns > 1:                                                     # legend
        lx = label_w
        for name, c in series:
            o.append(f'<rect x="{lx}" y="10" width="16" height="16" rx="3" fill="{col(c)}"/>'
                     f'<text x="{lx + 24}" y="24" font-size="18" fill="{col("ink2")}">{html.escape(name)}</text>')
            lx += 44 + 11 * len(name)
    step = xmax / 4
    for i in range(5):                                             # grid and ticks
        gx = x(i * step)
        o.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{H - bottom}" stroke="{col("grid")}" '
                 f'stroke-width="1"/>')
        o.append(f'<text x="{gx:.1f}" y="{H - bottom + 26}" font-size="17" text-anchor="middle" '
                 f'fill="{col("muted")}">{i * step:g}{unit}</text>')
    o.append(f'<line x1="{label_w}" y1="{top}" x2="{label_w}" y2="{H - bottom}" stroke="{col("axis")}" '
             f'stroke-width="1.5"/>')
    for r, (label, vals, tips) in enumerate(rows):
        y0 = top + r * row_h + 10
        o.append(f'<text x="{label_w - 14}" y="{y0 + (row_h - 20) / 2 + 6:.1f}" font-size="18" text-anchor="end" '
                 f'fill="{col("ink")}">{html.escape(label)}</text>')
        for s, (v, tip) in enumerate(zip(vals, tips)):
            y = y0 + s * (bar_h + 2)
            w = max(pw * v / xmax, 0)
            if w > 0:
                o.append(f'<path d="M{label_w},{y} h{max(w - 4, 0):.1f} a4,4 0 0 1 4,4 v{bar_h - 8} a4,4 0 0 1 -4,4 '
                         f'h{-max(w - 4, 0):.1f} z" fill="{col(series[s][1])}"><title>{html.escape(tip)}</title></path>')
            lab_x = label_w + w + 8
            if whisk and whisk[r] is not None and s == 0:
                lab_x = max(lab_x, x(whisk[r][1]) + 8)          # keep the value label clear of the interval line
            o.append(f'<text x="{lab_x:.1f}" y="{y + 15}" font-size="16" fill="{col("ink2")}">'
                     f'{v:g}{unit}</text>')
            if whisk and whisk[r] is not None and s == 0:
                lo, hi = whisk[r]
                o.append(f'<line x1="{x(lo):.1f}" y1="{y + bar_h / 2}" x2="{x(hi):.1f}" y2="{y + bar_h / 2}" '
                         f'stroke="{col("ink")}" stroke-width="2"/>')
    o.append("</svg>")
    return "".join(o)


def table(df):
    return df.to_html(index=False, border=0, classes="t")


# ---- data -------------------------------------------------------------------------------------------------------
m5 = pd.DataFrame([json.loads(l) for l in open(R("gen_eval_m500.jsonl"))])
m5t = m5.groupby("variant", sort=False).agg(n=("id", "size"), correct=("correct", "sum"), finished=("finished", "sum"),
                                            looped=("looped", "sum")).astype(int)
NAME5 = {"steer_incorrect_L15_c-1": "Steering vector (incorrect vs correct)", "published": "Published weight edit",
         "none": "Unedited model"}
order5 = ["steer_incorrect_L15_c-1", "published", "none"]
piv = m5.pivot(index="id", columns="variant", values="correct").dropna().astype(bool)
so = int((piv[order5[0]] & ~piv["published"]).sum())
po = int((~piv[order5[0]] & piv["published"]).sum())
from scipy.stats import binomtest  # noqa: E402
p_sign = binomtest(so, so + po, 0.5).pvalue

g = pd.read_csv(R("gen_eval_table_regraded.csv")).set_index("variant")
NAME36 = [("steer_incorrect_L15_c-1", "Steering, incorrect vs correct, strength −1"),
          ("steer_incorrect_L15_c-2", "Steering, incorrect vs correct, strength −2"),
          ("published", "Published weight edit (64 neurons × 2)"),
          ("proj_incorrect_a6", "Weight edit, incorrect-vs-correct direction"),
          ("proj_cut_a6", "Weight edit, hit-limit-vs-finished direction"),
          ("writecols_pubdir_a6", "Our 64 neurons, published direction"),
          ("pubcols_owndir_a6", "Published neurons, our loop direction"),
          ("act_top64_a6", "Our 64 neurons (activation only), loop direction"),
          ("write_top64_a6", "Our 64 neurons, loop direction"),
          ("full_owndir_a6", "Weight edit, loop direction, all neurons"),
          ("steer_loop_L15_c-2", "Steering, loop direction, strength −2"),
          ("random64_0_a6", "64 random neurons, loop direction"),
          ("steer_random_L15_c-2", "Steering, random direction, strength −2"),
          ("none", "Unedited model")]
raw36 = pd.DataFrame([json.loads(l) for l in open(R("gen_eval.jsonl"))])
for v, _ in NAME36:                                   # conditions finished after the regraded table was written
    if v not in g.index:
        d = raw36[raw36.variant == v]
        g.loc[v] = [len(d), int(d.looped.sum()), int(d.finished.sum()), int(d.correct.sum())]

s2 = pd.read_csv(R("step2_table.csv"))
s2 = s2[s2.fit == "all"]
M2 = [("write|published_u_signed", "Activation difference × write along published direction"),
      ("write|own_dir_signed", "Activation difference × write along our direction"),
      ("baseline|weights_only_abs_u_dot_W", "Weights only (uses published direction)"),
      ("within|absdiff", "Activation difference, loop vs productive tokens"),
      ("baseline|abs_mean_activation", "Baseline: largest mean activation"),
      ("between|absdiff", "Activation difference, looped vs non-looped prompts"),
      ("baseline|low_index_first", "Baseline: lowest neuron index"),
      ("baseline|random (mean of 200)", "Baseline: random")]
hits = lambda m, l: round(float(s2[(s2.method == m) & (s2.layer == l)]["P@64"].iloc[0]) * 64)  # noqa: E731

s3 = json.load(open(R("step3/summary.json")))["both layers"]["table"]
K3 = {"edited": "Edited neurons", "matched": "Matched neurons", "random": "Random neurons",
      "direction": "Random directions"}

px = pd.read_csv(R("step4_proxy_table.csv")).set_index("variant")
P4 = [("published", "Published edit"), ("write_top64_a6", "Our 64 neurons (activation × write)"),
      ("act_top64_a6", "Our 64 neurons (activation only)"), ("full_owndir_a6", "Loop direction, all 3,584 neurons"),
      ("pubcols_zero", "Published neurons zeroed"), ("random64_0_a6", "64 random neurons")]

# ---- charts -----------------------------------------------------------------------------------------------------
c_head = bars([(NAME5[v], [int(m5t.loc[v, "correct"]), int(m5t.loc[v, "looped"])],
                [f"{NAME5[v]}: {m5t.loc[v, 'correct']} of 60 correct", f"{NAME5[v]}: {m5t.loc[v, 'looped']} of 60 looped"])
               for v in order5], [("Correct", "s1"), ("Looped", "s2")], 60, label_w=400)
c_36 = bars([(n, [int(g.loc[v, "correct_regraded"]), int(g.loc[v, "looped"])],
             [f"{n}: correct of 36", f"{n}: looped of 36"]) for v, n in NAME36],
            [("Correct", "s1"), ("Looped", "s2")], 36, label_w=470, width=1100)
c_s2 = bars([(n, [hits(m, 15), hits(m, 19)], [f"{n}: layer 15, of 64", f"{n}: layer 19, of 64"]) for m, n in M2],
            [("Layer 15", "s3"), ("Layer 19", "s4")], 64, label_w=520, width=1100)
c_s3 = bars([(K3[r["kind"]], [round(100 * r["rate"])], [f"{K3[r['kind']]}: {r['one_job_any']} of {r['n']}"]) for r in s3],
            [("One job", "s1")], 100, unit="%", whisk=[(100 * r["ci_lo"], 100 * r["ci_hi"]) for r in s3], row_h=46)
t_px = pd.DataFrame([{"Edit": n, "Weights changed": f"{int(px.loc[v, 'weights']):,}",
                      "Loop text": f"{px.loc[v, 'loop']:.3f}", "Productive text": f"{px.loc[v, 'prod']:.3f}"}
                     for v, n in P4])
t_head = pd.DataFrame([{"Condition": NAME5[v], "Correct": f"{m5t.loc[v, 'correct']} / 60",
                        "Finished": f"{m5t.loc[v, 'finished']} / 60", "Looped": f"{m5t.loc[v, 'looped']} / 60"}
                       for v in order5])
t_s3 = pd.DataFrame([{"Unit type": K3[r["kind"]], "One job (with or without offshoots)": f"{r['one_job_any']} / {r['n']}",
                      "95% CI": f"{100 * r['ci_lo']:.0f}–{100 * r['ci_hi']:.0f}%",
                      "Labelled repetition-related": r["repetition_related"]} for r in s3])

if LITERAL:
    open(R("dashboard_check.svg"), "w").write(c_36)
    open(R("dashboard_check2.svg"), "w").write(c_head)
    open(R("dashboard_check3.svg"), "w").write(c_s2)
    open(R("dashboard_check4.svg"), "w").write(c_s3)
    sys.exit()

tile = lambda big, small: f'<div class="tile"><div class="big">{big}</div><div class="small">{small}</div></div>'  # noqa: E731
css_vars = lambda d: "".join(f"--{k}:{v};" for k, v in d.items())  # noqa: E731
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neuron interpretability vs a steering vector on a known weight edit</title>
<style>
:root{{{css_vars(LIGHT)}}} @media (prefers-color-scheme: dark){{:root{{{css_vars(DARK)}}}}}
body{{margin:0;background:var(--page);color:var(--ink);font:17px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:32px 24px 80px}}
h1{{font-size:34px;line-height:1.2;margin:0 0 10px}} h2{{font-size:24px;margin:52px 0 6px}}
.lede{{font-size:20px;color:var(--ink2);margin:0 0 24px;max-width:860px}}
.tiles{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:8px 0 20px}}
.tile,.card{{background:var(--surface);border:1px solid var(--grid);border-radius:12px}}
.tile{{padding:18px 22px}} .big{{font-size:44px;font-weight:650;line-height:1.1}} .small{{color:var(--ink2);margin-top:6px}}
.card{{padding:18px 18px 8px;margin:12px 0}} svg{{width:100%;height:auto;display:block}}
.cap{{color:var(--ink2);margin:10px 4px 14px;max-width:900px}} .note{{color:var(--ink2);max-width:900px}}
details{{margin:6px 4px 10px;color:var(--ink2)}} summary{{cursor:pointer}}
table.t{{border-collapse:collapse;margin:10px 0;font-size:16px}} table.t th,table.t td{{padding:6px 14px;border-bottom:1px solid var(--grid);text-align:left}}
table.t th{{color:var(--ink2);font-weight:600}}
@media (max-width:760px){{.tiles{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>A steering vector from incorrect vs correct rollouts beats the published weight edit</h1>
<p class="lede">Qwen3.5-0.8B loops on most math problems. On the same 60 held-out MATH-500 problems, adding one
difference-of-means vector after layer 15 gets {m5t.loc[order5[0], 'correct']} right; the published 64-neuron weight
edit gets {m5t.loc['published', 'correct']}; the unedited model gets {m5t.loc['none', 'correct']}. The vector was built
without knowing the failure was looping.</p>
<div class="tiles">
{tile(f"{m5t.loc[order5[0], 'correct']} / 60", "correct with the steering vector")}
{tile(f"{m5t.loc['published', 'correct']} / 60", "correct with the published weight edit")}
{tile(f"{so} vs {po}", f"problems only one of the two gets right (sign test p = {p_sign:.3f})")}
</div>
<div class="card">{c_head}</div>
<p class="cap">Held-out MATH-500 problems (not used to fit the vector), one sample each, 16,384-token cap, the post's
sampling settings. "Looped" means a loop onset was detected before the cap.</p>
<details><summary>Table</summary>{table(t_head)}</details>
<p class="note"><b>Limits.</b> The post's protocol uses a 32,768-token cap and four samples per problem; the shorter cap
favours whichever method finishes sooner. The 60 problems skew toward short statements. The vector rests on 15 correct
rollouts.</p>

<h2>All interventions on 36 practice problems</h2>
<div class="card">{c_36}</div>
<p class="cap">MATH train practice problems, one sample each, 8,192-token cap. Directions built from correctness or from
hitting the token limit both remove loops and reach answers. Directions built from loop labels remove loops but rarely
reach an answer. Random directions and random neurons are the controls. With 36 problems, gaps of a few problems are
noise.</p>

<h2>Can activation differences find the edited neurons?</h2>
<div class="card">{c_s2}</div>
<p class="cap">Number of the 64 edited neurons that appear in each ranking's top 64 (3,584 neurons per layer). Activation
difference alone finds 28 and 18. The edited neurons are high-variance, low-index neurons, so simple baselines already
find 9 to 20.</p>

<h2>Do the edited neurons have one clear job?</h2>
<div class="card">{c_s3}</div>
<p class="cap">Share of units that blinded labellers called one job, with or without related offshoots (128 units per
type; lines are 95% intervals). Edited neurons are far cleaner than random directions but no cleaner than other
neurons.</p>
<details open><summary>Table</summary>{table(t_s3)}</details>

<h2>Weight edits compared on a cheap proxy</h2>
<p class="note">Change in log-probability per token on 24 held-out looping generations: loop text should fall, productive
text should not. This proxy matched the published edit's size but did not predict which edits make the model finish, and
it ranked steering below weight edits, which generations reversed.</p>
{table(t_px)}
<p class="note">Source: <code>~/code/wsp-neurons/results</code>; notebook <code>notes/hypotheses.md</code>. Built by
<code>scripts/make_dashboard.py</code>.</p>
</main></body></html>"""
open(R("dashboard.html"), "w").write(page)
print("wrote", R("dashboard.html"), len(page), "bytes")
