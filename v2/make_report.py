"""Build a progress-report PDF from results/v2 + a hand-written notes fragment.

    cd v2 && python make_report.py NN notes.html
writes reports/progress/progress_NN_<date>_<time>.{html,pdf} (+ the chart PNG), rendered with headless Chrome.
The notes fragment holds the narrative sections (HTML); this script adds the results tables, chart and spend.
"""
import datetime
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import OUT, ROOT, TASK_ORDER  # noqa: E402

REP = os.path.join(ROOT, "reports", "progress")
os.makedirs(REP, exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
ORDER = ["jev", "qwen3.7-flash", "deepseek-v4-flash", "gpt-4o-mini", "llama-3.3-70b",
         "gpt-4o-mini-verbal", "deepseek-v4-flash-verbal", "gemini-3.8-flash", "claude-haiku-4.5"]
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
SHORT = {"nyc311": "NYC 311", "crimes": "crimes", "shelter": "shelter", "flights": "flights",
         "stackoverflow": "Stack Overflow", "hn": "HN", "github_prs": "GitHub PRs", "inspections": "inspections"}


def forest(T, family, path, title, xlabel):
    g = T[T.family == family]
    models = [m for m in ORDER if m in set(g.model)]
    n = len(models)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.05 * rows + 0.3), sharex=True, squeeze=False)
    lo, hi = g.lo.min(), g.hi.max()
    pad = 0.05 * (hi - lo)
    for ax, m in zip(axes.flat, models):
        h = g[g.model == m].set_index("task").reindex([t for t in TASK_ORDER if t in set(g[g.model == m].task)])
        ys = np.arange(len(h))[::-1]
        for y, (_, r) in zip(ys, h.iterrows()):
            c = BLUE if (r.holm_model and r.est > 0) else ORANGE if (r.holm_model and r.est < 0) else GRAY
            ax.plot([r.lo, r.hi], [y, y], color=c, lw=1.6, solid_capstyle="round")
            ax.plot(r.est, y, "o", ms=4.2, color=c, mec="white", mew=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([SHORT[t] for t in h.index], fontsize=6.5, color=MUTED)
        ax.set_title(m, fontsize=8, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=6.5, colors=MUTED)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#d0cfca")
        ax.grid(axis="x", color="#ecebe6", lw=0.6)
        ax.set_xlim(lo - pad, hi + pad)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=7, color=MUTED)
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pooled_chart(path, prefixes=("e1", "e1r"), fams=(("told-const", "told − constant"), ("H1 told-shift", "told − code shift"))):
    """Per model: pooled mean Brier difference over tasks with 95% CI, for each dataset (E1 filled, E1-R open)."""
    frames = []
    for pre in prefixes:
        f = os.path.join(OUT, f"{pre}_posthoc_pooled.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["data"] = pre
            frames.append(d)
    P = pd.concat(frames)
    models = [m for m in ORDER if m in set(P.model)]
    fig, axes = plt.subplots(1, len(fams), figsize=(7.2, 0.33 * len(models) + 1.0), sharey=True)
    for ax, (fam, lab) in zip(np.atleast_1d(axes), fams):
        for j, pre in enumerate(prefixes):
            g = P[(P.family == fam) & (P.data == pre)].set_index("model")
            for i, m in enumerate(models):
                if m not in g.index:
                    continue
                r = g.loc[m]
                y = len(models) - 1 - i + (0.17 if j == 0 else -0.17)
                c = BLUE if r.lo > 0 else ORANGE if r.hi < 0 else GRAY
                ax.plot([r.lo, r.hi], [y, y], color=c, lw=1.6)
                ax.plot(r.est, y, "o" if j == 0 else "s", ms=4.5, mfc=c if j == 0 else "white", mec=c, mew=1.2)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(lab, fontsize=8.5, color=INK, loc="left")
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_xlabel("Δ Brier (> 0: told is worse)", fontsize=7, color=MUTED)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color="#ecebe6", lw=0.6)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(models)))
    ax0.set_yticklabels(models[::-1], fontsize=7.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=MUTED, label="E1: 8 tasks, June–Aug 2026"),
         Line2D([], [], marker="s", ls="", mfc="white", mec=MUTED, label="E1-R: 5 fresh tasks, Sept 2026")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def verdict_table(T, family, a_name, b_name):
    g = T[T.family == family]
    rows = []
    for m in [m for m in ORDER if m in set(g.model)]:
        h = g[g.model == m]
        rows.append(f"<tr><td>{m}</td><td>{h.provider.iloc[0]}</td><td class=n>{len(h)}</td>"
                    f"<td class=n>{int((h.holm_model & (h.est > 0)).sum())}</td>"
                    f"<td class=n>{int((h.holm_model & (h.est < 0)).sum())}</td>"
                    f"<td class=n>{h.est.mean():+.4f}</td></tr>")
    return (f"<table><tr><th>Model</th><th>Provider</th><th>Tasks</th><th>{b_name} better (Holm)</th>"
            f"<th>{a_name} better (Holm)</th><th>Mean Δ Brier</th></tr>" + "".join(rows) + "</table>")


CSS = """
@page { size: Letter; margin: 16mm 16mm 16mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #0b0b0b; font-size: 10pt;
       line-height: 1.42; background: #fff; }
h1 { font-size: 16pt; margin: 0 0 2pt 0; } h2 { font-size: 11.5pt; margin: 12pt 0 4pt 0; border-bottom: 1px solid #e3e2dd; padding-bottom: 2pt; }
h3 { font-size: 10pt; margin: 8pt 0 3pt 0; }
.sub { color: #52514e; font-size: 9pt; margin-bottom: 8pt; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; font-size: 8.6pt; width: 100%; }
th, td { border-bottom: 1px solid #e3e2dd; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #f4f3ef; font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.box { background: #f4f7fc; border-left: 3px solid #2a78d6; padding: 6pt 9pt; margin: 6pt 0; }
.warn { background: #fdf3ee; border-left: 3px solid #eb6834; padding: 6pt 9pt; margin: 6pt 0; }
img { width: 100%; } ul { margin: 3pt 0 3pt 16pt; padding: 0; } li { margin: 1.5pt 0; }
.small { font-size: 8.3pt; color: #52514e; } .pb { page-break-before: always; }
"""


def spend_html():
    L = pd.read_csv(os.path.join(OUT, "spend_ledger.csv"))
    g = L.groupby("model").agg(usd=("usd", "sum"), n=("n_cases", "sum")).sort_values("usd", ascending=False)
    rows = "".join(f"<tr><td>{m}</td><td class=n>{int(r.n):,}</td><td class=n>${r.usd:.3f}</td></tr>" for m, r in g.iterrows())
    return (f"<table><tr><th>Model</th><th class=n>Decisions</th><th class=n>Cost</th></tr>{rows}"
            f"<tr><th>Total v2</th><th></th><th class=n>${L.usd.sum():.2f}</th></tr></table>")


def main(nn, notes_path):
    now = datetime.datetime.now()
    stem = f"progress_{int(nn):02d}_{now:%Y-%m-%d_%H%M}"
    T = pd.read_csv(os.path.join(OUT, "e1_tests.csv"))
    png = os.path.join(REP, f"{stem}_h1.png")
    forest(T, "H1 told-shift", png, "Brier(told) − Brier(code shift), per task, 95% paired bootstrap CI (mean of 3 seeds)",
           "Δ Brier (> 0: shifting in code is better)")
    notes = open(notes_path).read()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Progress report {nn}</title><style>{CSS}</style></head><body>
{notes.replace('{{H1_TABLE}}', verdict_table(T, 'H1 told-shift', 'told', 'shift'))
      .replace('{{CHART}}', f'<img src="{os.path.basename(png)}">')
      .replace('{{SPEND}}', spend_html())
      .replace('{{P5_TABLE}}', verdict_table(T, 'P5 shift-const', 'shift', 'const'))
      .replace('{{P6_TABLE}}', verdict_table(T, 'P6 platt-shift', 'Platt', 'shift'))
      .replace('{{LR_TABLE}}', verdict_table(T, 'LR lr-shift', 'LR', 'shift'))
      .replace('{{TOLDZS_TABLE}}', verdict_table(T, 'told-zs', 'told', 'zero-shot'))
      .replace('{{TOLDCONST_TABLE}}', verdict_table(T, 'told-const', 'told', 'const'))}
</body></html>"""
    hp = os.path.join(REP, f"{stem}.html")
    open(hp, "w").write(html)
    pdf = os.path.join(REP, f"{stem}.pdf")
    subprocess.run([CHROME, "--headless=new", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", hp],
                   check=True, capture_output=True)
    print(pdf)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
