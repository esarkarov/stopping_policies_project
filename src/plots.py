import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
FIXED_COLOR = "#8a8984"

ADAPTIVE_STYLE = {
    "P1": {"color": "#2a78d6", "marker": "s", "label": "P1 self-decided"},
    "P2": {"color": "#eb6834", "marker": "^", "label": "P2 answer stability"},
    "P3": {"color": "#1baf7a", "marker": "D", "label": "P3 sub-question coverage"},
}
FIXED = [f"P0_k{k}" for k in range(1, 6)]
METRIC_NAMES = {"f1": "F1", "em": "Exact match"}
COST_LABELS = {"searches": "Average number of searches", "tokens": "Average tokens per question"}


def plot_quality_vs_cost(summary, metric: str, title: str, path, cost: str = "searches") -> None:
    rows = summary.set_index("policy")
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [rows.loc[p, cost] for p in FIXED]
    ys = [rows.loc[p, metric] for p in FIXED]
    lo = [rows.loc[p, f"{metric}_lo"] for p in FIXED]
    hi = [rows.loc[p, f"{metric}_hi"] for p in FIXED]
    ax.fill_between(xs, lo, hi, color=FIXED_COLOR, alpha=0.15, linewidth=0)
    ax.plot(xs, ys, color=FIXED_COLOR, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="P0 fixed budget (k = 1 to 5)", zorder=3)

    present = [p for p in ADAPTIVE_STYLE if p in rows.index]
    sizes = {"P1": 9, "P2": 10.5, "P3": 12}
    for policy in sorted(present, key=lambda p: -sizes[p]):
        style, r = ADAPTIVE_STYLE[policy], rows.loc[policy]
        ax.plot(r[cost], r[metric], linestyle="none", marker=style["marker"], markersize=sizes[policy],
                color=style["color"], markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=style["label"], zorder=4 + (12 - sizes[policy]) / 10)

    if present:
        y_low, y_high = ax.get_ylim()
        gap = 0.06 * (y_high - y_low)
        x_low, x_high = min(xs), max(xs)
        span = x_high - x_low
        label_x = max(rows.loc[p, cost] for p in present) + 0.1125 * span
        ordered = sorted(present, key=lambda p: -rows.loc[p, metric])
        top = max(rows.loc[p, metric] for p in present) + gap * (len(ordered) - 1) / 2
        for i, policy in enumerate(ordered):
            r = rows.loc[policy]
            ax.annotate(policy, (r[cost], r[metric]), xytext=(label_x, top - i * gap),
                        fontsize=9, color=INK, fontweight="bold", va="center",
                        arrowprops={"arrowstyle": "-", "color": ADAPTIVE_STYLE[policy]["color"], "lw": 0.9,
                                    "shrinkA": 1, "shrinkB": 6})

    if "Oracle" in rows.index:
        r = rows.loc["Oracle"]
        ax.plot(r[cost], r[metric], linestyle="none", marker="*", markersize=13, color=INK,
                markeredgecolor=SURFACE, markeredgewidth=1, label="Oracle (reference only)", zorder=4)
        ax.annotate("Oracle", (r[cost], r[metric]), textcoords="offset points", xytext=(9, -3),
                    fontsize=9, color=INK)

    ax.set_xlabel(COST_LABELS[cost], color=INK_SECONDARY)
    ax.set_ylabel(METRIC_NAMES[metric], color=INK_SECONDARY)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    if cost == "searches":
        ax.set_xlim(0.7, 5.3)
    else:
        xmax = max(rows[cost].max(), max(xs))
        ax.set_xlim(-0.04 * xmax, xmax * 1.2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (labels[i][:2] == "Or", labels[i]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", frameon=False,
              fontsize=8, labelcolor=INK_SECONDARY)
    fig.text(0.01, 0.01, "Band: 95% bootstrap confidence interval of the fixed-budget F1" if metric == "f1" else "Band: 95% bootstrap confidence interval of the fixed-budget exact match", fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
