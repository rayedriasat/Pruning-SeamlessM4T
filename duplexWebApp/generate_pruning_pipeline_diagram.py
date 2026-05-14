import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as path_effects
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(28, 20))
ax.set_xlim(0, 28)
ax.set_ylim(0, 20)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color Palette ─────────────────────────────────────────────
C_BASELINE  = "#E8EAF6"
C_VOCAB     = "#C5CAE9"
C_ENCODER   = "#9FA8DA"
C_T2U       = "#7986CB"
C_DECODER   = "#5C6BC0"
C_RECOVERY  = "#66BB6A"
C_FINAL     = "#43A047"
C_METRIC    = "#FFF9C4"
C_ARROW     = "#424242"

BOX_W = 4.8
BOX_H = 2.8

def phase_box(ax, x, y, w, h, phase_num, title, details, metrics, color, highlight=False):
    lw = 3.5 if highlight else 2.5
    edge = "#D32F2F" if highlight else "#37474F"

    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.12",
                         linewidth=lw, edgecolor=edge,
                         facecolor=color, zorder=3)
    ax.add_patch(box)

    badge_y = y + h/2 - 0.42
    badge = plt.Circle((x - w/2 + 0.5, badge_y), 0.32,
                       color=edge, zorder=4)
    ax.add_patch(badge)
    ax.text(x - w/2 + 0.5, badge_y, f"P{phase_num}",
            ha='center', va='center', fontsize=20,
            fontweight='bold', color='white', zorder=5)

    title_y = y + h/2 - 0.42
    ax.text(x + 0.2, title_y, title, ha='center', va='center',
            fontsize=23, fontweight='bold', zorder=4)

    detail_y = y + 0.22
    for i, line in enumerate(details):
        ax.text(x, detail_y - i*0.38, line, ha='center', va='center',
                fontsize=17, color="#263238", zorder=4, style='italic')

    metric_box = FancyBboxPatch((x - w/2 + 0.18, y - h/2 + 0.15),
                                w - 0.36, 0.95,
                                boxstyle="round,pad=0.07",
                                linewidth=1.5, edgecolor="#F57C00",
                                facecolor=C_METRIC, alpha=0.9, zorder=4)
    ax.add_patch(metric_box)

    metric_y = y - h/2 + 0.78
    for i, (key, val) in enumerate(metrics.items()):
        if i < 2:
            metric_x = x - w/4 + (i * w/2)
            ax.text(metric_x, metric_y, f"{key}: {val}",
                   ha='center', va='center', fontsize=17,
                   fontweight='bold', color="#E65100", zorder=5)
        else:
            metric_x = x - w/4 + ((i-2) * w/2)
            ax.text(metric_x, metric_y - 0.38, f"{key}: {val}",
                   ha='center', va='center', fontsize=17,
                   fontweight='bold', color="#E65100", zorder=5)

def straight_arrow(ax, x1, y1, x2, y2, label=None, color=C_ARROW, lw=3.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.4",
                               color=color, lw=lw),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset_x = 0.0
        offset_y = 0.25 if abs(y2 - y1) < 0.1 else 0.0
        if abs(y2 - y1) > 0.5 and abs(x2 - x1) < 0.5:
            offset_x = 0.55
        ax.text(mx + offset_x, my + offset_y, label,
                fontsize=16, color="#0D47A1",
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='#1565C0', linewidth=2, alpha=0.95),
                zorder=6)

def elbow_arrow(ax, x1, y1, x2, y2, label=None, color=C_ARROW, lw=3.0):
    """L-shaped arrow: go down then left (or right)"""
    # draw two line segments + arrowhead at end
    mid_y = y2  # go vertical first, then horizontal
    ax.plot([x1, x1], [y1, mid_y], color=color, lw=lw, zorder=2)
    ax.annotate("", xy=(x2, y2), xytext=(x1, mid_y),
                arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.4",
                               color=color, lw=lw),
                zorder=2)
    if label:
        mx = (x1 + x2) / 2
        ax.text(mx, y2 + 0.25, label, fontsize=12, color="#0D47A1",
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='#1565C0', linewidth=2, alpha=0.95),
                zorder=6)

# ══════════════════════════════════════════════════════════════
#  LAYOUT POSITIONS
#  Row 1 (y=15.5): P0, P1, P2
#  Row 2 (y=10.5): P3, P4, P5
#  Row 3 (y=5.5):  P6, P7
#  Side panels: x=22..27
# ══════════════════════════════════════════════════════════════

# Column x-positions for main boxes
C1, C2, C3 = 3.2, 10.2, 17.2

# Row y-positions
R1, R2, R3 = 15.5, 10.5, 5.5

# ── Title ─────────────────────────────────────────────────────
title = ax.text(13.5, 19.2,
                "SeamlessM4Tv2 Compression Pipeline: Pruning & Recovery",
                ha='center', va='center', fontsize=28, fontweight='bold')
title.set_path_effects([path_effects.withStroke(linewidth=4, foreground='white')])

ax.text(13.5, 18.5, "Structured Pruning with Knowledge Distillation Recovery",
        ha='center', va='center', fontsize=20, style='italic', color="#37474F")

# ── Row 1 boxes ───────────────────────────────────────────────
phase_box(ax, C1, R1, BOX_W, BOX_H, 0, "Baseline",
          ["facebook/seamless-m4t-v2-large",
           "Full S2ST model"],
          {"Params": "1805.5M", "ChrF": "46.49", "BLEU": "15.88", "RTF": "0.2455"},
          C_BASELINE)

phase_box(ax, C2, R1, BOX_W, BOX_H, 1, "Vocabulary Pruning",
          ["5-language vocab trimming",
           "Keep: eng, ben, cmn, arb, hin"],
          {"Params": "1566.6M", "ChrF": "41.74", "BLEU": "13.65", "RTF": "0.2435"},
          C_VOCAB)

phase_box(ax, C3, R1, BOX_W, BOX_H, 2, "Speech Encoder Pruning",
          ["24 → 16 layers (BI-guided)",
           "Iterative removal"],
          {"Params": "1373.1M", "ChrF": "38.97", "BLEU": "11.13", "RTF": "0.1617"},
          C_ENCODER)

# ── Row 2 boxes ───────────────────────────────────────────────
phase_box(ax, C1, R2, BOX_W, BOX_H, 3, "T2U Merge",
          ["LaCo/RDSC layer merging",
           "Conservative threshold"],
          {"Params": "1331.2M", "ChrF": "38.47", "BLEU": "11.21", "RTF": "0.1646"},
          C_T2U)

phase_box(ax, C2, R2, BOX_W, BOX_H, 4, "Encoder Pruning (2nd)",
          ["16 → 14 layers",
           "Further BI-guided removal"],
          {"Params": "1282.8M", "ChrF": "35.74", "BLEU": "9.67", "RTF": "0.1635"},
          C_ENCODER)

phase_box(ax, C3, R2, BOX_W, BOX_H, 5, "Text Decoder Pruning",
          ["24 → 14 layers (aggressive)",
           "Largest quality drop"],
          {"Params": "1030.9M", "ChrF": "25.32", "BLEU": "5.83", "RTF": "0.1881"},
          C_DECODER)

# ── Row 3 boxes (centered between C1..C3 range) ───────────────
RC1 = 6.5
RC2 = 13.5

phase_box(ax, RC1, R3, BOX_W, BOX_H, 6, "KD Recovery",
          ["Teacher-student distillation",
           "Sparse top-k logit matching"],
          {"Params": "~1030M", "ChrF": "33.07", "BLEU": "7.95", "RTF": "0.1484"},
          C_RECOVERY)

phase_box(ax, RC2, R3, BOX_W, BOX_H, 7, "Final Hybrid Recovery",
          ["LoRA/DoRA adapters",
           "Merged & deployed"],
          {"Params": "1087.9M", "ChrF": "33.73", "BLEU": "8.16", "RTF": "0.1336"},
          C_FINAL, highlight=True)

# ══════════════════════════════════════════════════════════════
#  ARROWS
# ══════════════════════════════════════════════════════════════

# Row 1: P0 → P1 → P2 (horizontal)
straight_arrow(ax, C1 + BOX_W/2, R1, C2 - BOX_W/2, R1, "−239M")
straight_arrow(ax, C2 + BOX_W/2, R1, C3 - BOX_W/2, R1, "−193M\n1.51× faster")

# P2 → P3: direct diagonal from bottom-left of P2 to top-right of P3
p2_x = C3 - BOX_W/2  # left edge of P2 box
p2_y = R1 - BOX_H/2  # bottom edge of P2 box
p3_x = C1 + BOX_W/2  # right edge of P3 box
p3_y = R2 + BOX_H/2  # top edge of P3 box

ax.annotate("", xy=(p3_x, p3_y), xytext=(p2_x, p2_y),
            arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.4",
                           color=C_ARROW, lw=3.0),
            zorder=2)

# Row 2: P3 → P4 → P5 (horizontal)
straight_arrow(ax, C1 + BOX_W/2, R2, C2 - BOX_W/2, R2, "−42M")
straight_arrow(ax, C2 + BOX_W/2, R2, C3 - BOX_W/2, R2, "−48M")

# P5 → P6: direct diagonal from bottom-left of P5 to top-right of P6
p5_x = C3 - BOX_W/2
p5_y = R2 - BOX_H/2
p6_x = RC1 + BOX_W/2
p6_y = R3 + BOX_H/2

ax.annotate("", xy=(p6_x, p6_y), xytext=(p5_x, p5_y),
            arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.4",
                           color=C_ARROW, lw=3.0),
            zorder=2)
mx = (p5_x + p6_x) / 2
my = (p5_y + p6_y) / 2
ax.text(mx - 0.8, my, "Quality cliff\n−10 ChrF", fontsize=12, color="#0D47A1",
        ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                 edgecolor='#1565C0', linewidth=2, alpha=0.95),
        zorder=6)

# P6 → P7 (horizontal)
straight_arrow(ax, RC1 + BOX_W/2, R3, RC2 - BOX_W/2, R3, "+0.66 ChrF\n+0.21 BLEU")

# ══════════════════════════════════════════════════════════════
#  SIDE PANELS (x = 21.5 to 27.5)
# ══════════════════════════════════════════════════════════════
PANEL_X = 21.5
PANEL_W = 5.8

def side_panel(ax, x, y, w, h, title, items, bg, edge, title_color, item_color):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         linewidth=2.5, edgecolor=edge,
                         facecolor=bg, alpha=0.95, zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.45, title, ha='center', va='center',
            fontsize=18, fontweight='bold', color=title_color, zorder=4)
    for i, item in enumerate(items):
        ax.text(x + w/2, y + h - 1.05 - i*0.52, item,
                ha='center', va='center', fontsize=15, color=item_color, zorder=4)

side_panel(ax, PANEL_X, 13.5, PANEL_W, 4.5,
           "Compression Summary",
           ["• 39.7% param reduction",
            "• 1.84× RTF speedup",
            "• 72.6% ChrF retention",
            "• Runs on 4GB GPU"],
           "#E3F2FD", "#1976D2", "#0D47A1", "#1565C0")

side_panel(ax, PANEL_X, 8.0, PANEL_W, 4.8,
           "Key Techniques",
           ["• Block Influence (BI)",
            "• Iterative layer removal",
            "• Conservative T2U merge",
            "• Sparse KD recovery",
            "• LoRA/DoRA adapters"],
           "#E8F5E9", "#388E3C", "#1B5E20", "#2E7D32")

side_panel(ax, PANEL_X, 3.0, PANEL_W, 4.5,
           "Failed Attempts",
           ["✗ FLAP width pruning",
            "  (collapsed generation)",
            "✗ Sub-500M target",
            "  (destroyed quality)",
            "→ Defined safe boundary"],
           "#FFEBEE", "#D32F2F", "#B71C1C", "#C62828")

# ══════════════════════════════════════════════════════════════
#  SECTION LABELS
# ══════════════════════════════════════════════════════════════
pruning_label = ax.text(0.7, 13.0, "STRUCTURAL\nPRUNING",
                        ha='center', va='center',
                        fontsize=21, fontweight='bold', color="#37474F",
                        rotation=90, zorder=4)
pruning_label.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

recovery_label = ax.text(0.7, 5.5, "RECOVERY",
                         ha='center', va='center',
                         fontsize=22, fontweight='bold', color="#2E7D32",
                         rotation=90, zorder=4)
recovery_label.set_path_effects([path_effects.withStroke(linewidth=4, foreground='white')])

# Recovery dashed outline
recovery_highlight = Rectangle((4.8, 4.1), 10.6, 2.8,
                               linewidth=3, edgecolor="#43A047",
                               facecolor='none', linestyle='--',
                               alpha=0.7, zorder=1)
ax.add_patch(recovery_highlight)
ax.text(RC1 + (RC2 - RC1)/2 + 1.5, 3.85,
        "Knowledge Distillation Recovery Phase",
        ha='center', va='center',
        fontsize=16, fontweight='bold', color="#2E7D32", style='italic', zorder=4)

# ══════════════════════════════════════════════════════════════
#  QUALITY TREND ARROWS
# ══════════════════════════════════════════════════════════════
ax.annotate("", xy=(20.5, 10.8), xytext=(20.5, 14.5),
            arrowprops=dict(arrowstyle="->,head_width=0.6,head_length=0.5",
                           color="#D32F2F", lw=4.5, linestyle='--'),
            zorder=1)
qd = ax.text(21.1, 12.7, "QUALITY\nDROP", ha='center', va='center',
             fontsize=18, color="#D32F2F", fontweight='bold', rotation=-90, zorder=4)
qd.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

ax.annotate("", xy=(20.5, 6.9), xytext=(20.5, 4.2),
            arrowprops=dict(arrowstyle="->,head_width=0.6,head_length=0.5",
                           color="#43A047", lw=5.0),
            zorder=1)
qr = ax.text(21.1, 5.5, "QUALITY\nRECOVERY", ha='center', va='center',
             fontsize=18, color="#2E7D32", fontweight='bold', rotation=-90, zorder=4)
qr.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# ══════════════════════════════════════════════════════════════
#  LEGEND
# ══════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=C_BASELINE, edgecolor='#37474F', linewidth=2, label='Baseline'),
    mpatches.Patch(facecolor=C_VOCAB,    edgecolor='#37474F', linewidth=2, label='Vocabulary Pruning'),
    mpatches.Patch(facecolor=C_ENCODER,  edgecolor='#37474F', linewidth=2, label='Encoder Pruning'),
    mpatches.Patch(facecolor=C_T2U,      edgecolor='#37474F', linewidth=2, label='T2U Merge'),
    mpatches.Patch(facecolor=C_DECODER,  edgecolor='#37474F', linewidth=2, label='Decoder Pruning'),
    mpatches.Patch(facecolor=C_RECOVERY, edgecolor='#37474F', linewidth=2, label='KD Recovery'),
    mpatches.Patch(facecolor=C_FINAL,    edgecolor='#D32F2F', linewidth=2.5, label='Final Model'),
]
ax.legend(handles=legend_items, loc='lower left',
          fontsize=12, framealpha=0.95, ncol=4,
          title="Pipeline Phases", title_fontsize=13,
          bbox_to_anchor=(0.02, 0.01))

ax.text(13.5, 0.8,
        "8 language pairs: eng↔ben, eng↔cmn, eng↔arb, eng↔hin  |  Evaluation: 25 samples/pair on FLEURS",
        ha='center', va='center', fontsize=11, color="#546E7A", style='italic')

plt.tight_layout(pad=0.5)
plt.savefig("./pruning_pipeline_fixed.png", dpi=200, bbox_inches='tight')
plt.savefig("./pruning_pipeline_fixed.pdf", dpi=200, bbox_inches='tight')
print("Done.")