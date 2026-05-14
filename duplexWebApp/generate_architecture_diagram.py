import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects

fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('white')  # Pure white background

C_SPEECH  = "#AED6F1"
C_TEXTDEC = "#A9DFBF"
C_T2U     = "#F9E79F"
C_VOCODER = "#F5CBA7"
C_IO      = "#D7DBDD"
C_SHARED  = "#D2B4DE"
C_ADAPTER = "#FF6B6B"
C_WRAPPER = "#FFE66D"

boxes = {}

def box(ax, key, x, y, w, h, label, sublabel=None,
        color="#FFFFFF", fontsize=16, bold=False, highlight=False):
    boxes[key] = (x, y, w, h)
    lw      = 3.0 if highlight else 1.8
    edgecol = "#C0392B" if highlight else "#2C3E50"
    if highlight:
        glow = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.1",
                              linewidth=7, edgecolor=edgecol,
                              facecolor='none', alpha=0.25, zorder=3)
        ax.add_patch(glow)
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle="round,pad=0.1",
                          linewidth=lw, edgecolor=edgecol,
                          facecolor=color, zorder=4)
    ax.add_patch(rect)
    dy = 0.22 if sublabel else 0
    weight = 'bold' if (bold or highlight) else 'normal'
    txt = ax.text(x, y+dy, label, ha='center', va='center',
                  fontsize=fontsize, fontweight=weight, zorder=6)
    if highlight:
        txt.set_path_effects(
            [path_effects.withStroke(linewidth=2, foreground='white')])
    if sublabel:
        ax.text(x, y-dy, sublabel, ha='center', va='center',
                fontsize=fontsize-3, color="#333333",
                zorder=6, style='italic')


def edge_point(key, side, offset=0.10):
    x, y, w, h = boxes[key]
    if side == 'top':    return x,             y + h/2 + offset
    if side == 'bottom': return x,             y - h/2 - offset
    if side == 'left':   return x - w/2 - offset, y
    if side == 'right':  return x + w/2 + offset, y
    return x, y


def arrow(ax, x1, y1, x2, y2, label=None,
          color="#2C3E50", lw=2.2, dashed=False):
    ls = (0, (5, 4)) if dashed else 'solid'
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.32,head_length=0.22",
                    color=color, lw=lw, linestyle=ls,
                    shrinkA=0, shrinkB=0),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, fontsize=12, color="#333333",
                ha='left', va='center', style='italic', zorder=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='none', alpha=0.88))


def arrow_between(ax, from_key, from_side, to_key, to_side,
                  label=None, color="#2C3E50", lw=2.2,
                  dashed=False, gap=0.10):
    x1, y1 = edge_point(from_key, from_side, offset=gap)
    x2, y2 = edge_point(to_key,   to_side,   offset=gap)
    arrow(ax, x1, y1, x2, y2,
          label=label, color=color, lw=lw, dashed=dashed)


# ══════════════════════════════════════════════════════════════
#  LAYOUT  — y positions spread across full canvas height
#  Each row given enough vertical room so bigger text fits
# ══════════════════════════════════════════════════════════════
#
#  y=13.4  title
#  y=12.5  input
#  y=11.2  speech encoder
#  y=9.5   wrapper (left) | CIF detector (right)
#  y=7.7   chunk (center) | boundary probs (right)
#  y=6.0   shared (left)  | text decoder  | lm head
#  y=4.3   t2u enc        | t2u dec       | units
#  y=2.5   vocoder
#  y=1.0   output

# Title
ax.text(9, 13.4,
        "SeamlessM4Tv2 — Architecture Overview with CIF Streaming Extension",
        ha='center', va='center', fontsize=17, fontweight='bold', zorder=6,
        path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])

# ── Row 1: Input ──────────────────────────────────────────────
box(ax, 'input', 9, 12.5, 4.0, 0.75,
    "Live Audio Stream", "(16kHz, continuous)", C_IO, fontsize=15)

# ── Row 2: Speech Encoder ─────────────────────────────────────
box(ax, 'enc', 9, 11.2, 6.0, 1.0,
    "Speech Encoder  (14 Layers)",
    "Feature Proj → Conformer Blocks → Adapter",
    C_SPEECH, fontsize=16)

# ── Row 3: Wrapper (left) | CIF (right) ──────────────────────
box(ax, 'wrap', 3.6, 9.5, 4.8, 1.1,
    "SeamlessChunkedWrapper",
    "Integrates base model + CIF adapter",
    C_WRAPPER, bold=True, highlight=True, fontsize=15)

# --- after you create the CIF box ---
box(ax, 'cif', 14.0, 9.5, 5.0, 1.3,
    "CIF Boundary Detector",
    "MLP: 1024→512→256→1  (Sigmoid)\nDetects speech / silence boundaries",
    C_ADAPTER, bold=True, highlight=True, fontsize=15)

# Move the INNOVATION badge to the top-right of the CIF box (no overlap)
cx, cy, cw, ch = boxes['cif']
star_x = cx + cw/2 - 0.7      # shift right (tweak this number if you want)
star_y = cy + ch/2 + 0.45     # above the box

ax.scatter([star_x], [star_y], s=500, marker='*',
           color='#FFD700', edgecolors='#C0392B', linewidths=2, zorder=7)
ax.text(star_x, star_y + 0.45, "INNOVATION", ha='center', va='center',
        fontsize=12, fontweight='bold', color='#C0392B', zorder=7)

# ── Row 4: Chunking (center) | Boundary probs (right) ─────────
box(ax, 'chunk', 8.2, 7.7, 5.2, 0.85,
    "Adaptive Chunking",
    "silence_patience=0.8 s,  threshold=0.2",
    C_WRAPPER, bold=True, fontsize=15)

box(ax, 'bprob', 14.0, 7.7, 4.4, 0.85,
    "Boundary Probabilities",
    "(per-frame silence scores)", C_IO, fontsize=14)

# ── Row 5: Shared Emb | Text Decoder | LM Head ────────────────
box(ax, 'shared', 1.8, 6.0, 2.8, 0.9,
    "Shared\nEmbedding", "(vocab = 22 767)", C_SHARED, fontsize=14)

box(ax, 'tdec', 7.0, 6.0, 5.2, 1.1,
    "Text Decoder  (14 Layers)",
    "Self-Attn → Cross-Attn → FFN",
    C_TEXTDEC, fontsize=16)

box(ax, 'lmh', 12.8, 6.0, 3.2, 0.9,
    "LM Head", "(→ text tokens)", C_IO, fontsize=14)

# ── Row 6: T2U Enc | T2U Dec | Units ──────────────────────────
box(ax, 't2uenc', 4.2, 4.3, 4.0, 1.0,
    "T2U Encoder  (4L)",
    "Self-Attn → FFN", C_T2U, fontsize=14)

box(ax, 't2udec', 9.4, 4.3, 4.4, 1.0,
    "T2U Decoder  (6L)",
    "Duration Pred + Self-Attn + Conv",
    C_T2U, fontsize=14)

box(ax, 'units', 14.8, 4.3, 3.6, 0.9,
    "Unit Tokens", "(vocab = 10 082)", C_IO, fontsize=14)

# ── Row 7: Vocoder ────────────────────────────────────────────
box(ax, 'voc', 9, 2.5, 6.2, 1.0,
    "Vocoder  (HiFi-GAN)",
    "Unit/Spk/Lang Emb → Upsample → 15 ResBlocks",
    C_VOCODER, fontsize=16)

# ── Row 8: Output ─────────────────────────────────────────────
box(ax, 'output', 9, 1.0, 4.8, 0.78,
    "Translated Speech Output",
    "(streaming waveform chunks)", C_IO, fontsize=15)

# ── Live Streaming annotation (top-left) ──────────────────────
ann = FancyBboxPatch((0.2, 10.8), 2.6, 2.3,
                     boxstyle="round,pad=0.15",
                     linewidth=1.4, edgecolor="#34495E",
                     facecolor="#ECF0F1", alpha=0.93, zorder=4)
ax.add_patch(ann)
for i, (txt, fw, fs) in enumerate([
        ("Live Streaming",        'bold',   13),
        ("• Continuous input",    'normal', 11.5),
        ("• Adaptive chunking",   'normal', 11.5),
        ("• Real-time inference", 'normal', 11.5),
        ("• Queue-based output",  'normal', 11.5)]):
    ax.text(1.5, 12.7 - i*0.44, txt,
            ha='center', va='center',
            fontsize=fs, fontweight=fw, color="#2C3E50", zorder=6)

# ══════════════════════════════════════════════════════════════
#  ARROWS
# ══════════════════════════════════════════════════════════════
GAP = 0.10

arrow_between(ax, 'input',  'bottom', 'enc',    'top',   gap=GAP)
arrow_between(ax, 'enc',    'left',   'wrap',   'top',
              color="#E67E22", lw=2.6, gap=GAP)
arrow_between(ax, 'enc',    'right',  'cif',    'top',
              label="hidden states", color="#C0392B", lw=3.2, gap=GAP)
arrow_between(ax, 'cif',    'bottom', 'bprob',  'top',
              color="#C0392B", lw=2.6, gap=GAP)
arrow_between(ax, 'bprob',  'left',   'chunk',  'right',
              color="#C0392B", lw=2.6, gap=GAP)
arrow_between(ax, 'wrap',   'bottom', 'chunk',  'left',
              dashed=True, color="#7F8C8D", lw=2.0, gap=GAP)
arrow_between(ax, 'chunk',  'bottom', 'tdec',   'top',   gap=GAP)
arrow_between(ax, 'shared', 'right',  'tdec',   'left',
              color="#8E44AD", lw=2.4, gap=GAP)
arrow_between(ax, 'tdec',   'right',  'lmh',    'left',  gap=GAP)
arrow_between(ax, 'lmh',    'bottom', 't2uenc', 'top',   gap=GAP)
arrow_between(ax, 'lmh',    'bottom', 't2udec', 'top',   gap=GAP)
arrow_between(ax, 't2uenc', 'right',  't2udec', 'left',
              label="cross-attn", color="#B7950B", lw=2.4, gap=GAP)
arrow_between(ax, 't2udec', 'right',  'units',  'left',  gap=GAP)
arrow_between(ax, 'units',  'bottom', 'voc',    'right', gap=GAP)
arrow_between(ax, 't2uenc', 'bottom', 'voc',    'left',  gap=GAP)
arrow_between(ax, 'voc',    'bottom', 'output', 'top',   gap=GAP)

# ══════════════════════════════════════════════════════════════
#  LEGEND
# ══════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=C_ADAPTER, edgecolor='#C0392B', lw=2,
                   label='CIF Boundary Detector  (Innovation)'),
    mpatches.Patch(facecolor=C_WRAPPER, edgecolor='#2C3E50',
                   label='Chunked Wrapper  (Integration)'),
    mpatches.Patch(facecolor=C_SPEECH,  edgecolor='#2C3E50',
                   label='Speech Encoder'),
    mpatches.Patch(facecolor=C_TEXTDEC, edgecolor='#2C3E50',
                   label='Text Decoder'),
    mpatches.Patch(facecolor=C_T2U,     edgecolor='#2C3E50',
                   label='T2U Model'),
    mpatches.Patch(facecolor=C_VOCODER, edgecolor='#2C3E50',
                   label='Vocoder'),
    mpatches.Patch(facecolor=C_SHARED,  edgecolor='#2C3E50',
                   label='Shared Embedding'),
    mpatches.Patch(facecolor=C_IO,      edgecolor='#2C3E50',
                   label='I / O'),
]
ax.legend(handles=legend_items, loc='lower right',
          fontsize=12, framealpha=0.96, ncol=2,
          title="Components", title_fontsize=13,
          bbox_to_anchor=(0.99, 0.01))

plt.tight_layout()
plt.savefig("seamless_cif_architecture.pdf", dpi=300, bbox_inches='tight')
plt.savefig("seamless_cif_architecture.png", dpi=300, bbox_inches='tight')
print("✓ Saved: seamless_cif_architecture.pdf / .png")
plt.show()