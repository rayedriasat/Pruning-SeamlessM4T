import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

fig, ax = plt.subplots(1, 1, figsize=(20, 16))
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.axis('off')
fig.patch.set_facecolor('white')

# Color Palette
C_DATA = "#AED6F1"          # Data processing
C_MODEL = "#A9DFBF"         # Model components
C_TRAINING = "#F9E79F"      # Training process
C_INFERENCE = "#F5CBA7"     # Inference
C_OUTPUT = "#D7DBDD"        # Output/Results
C_ADAPTER = "#FF6B6B"       # CIF Adapter (highlight)
C_LOSS = "#E8DAEF"          # Loss/Optimization
C_DATASET = "#D5F4E6"       # Dataset

boxes = {}

def box(ax, key, x, y, w, h, label, sublabel=None,
        color="#FFFFFF", fontsize=15, bold=False, highlight=False):
    boxes[key] = (x, y, w, h)
    lw = 3.0 if highlight else 1.8
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
    
    dy = 0.25 if sublabel else 0
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
    if side == 'top':    return x, y + h/2 + offset
    if side == 'bottom': return x, y - h/2 - offset
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
        ax.text(mx+0.15, my, label, fontsize=11, color="#333333",
                ha='left', va='center', style='italic', zorder=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='none', alpha=0.88))

def arrow_between(ax, from_key, from_side, to_key, to_side,
                  label=None, color="#2C3E50", lw=2.2,
                  dashed=False, gap=0.10):
    x1, y1 = edge_point(from_key, from_side, offset=gap)
    x2, y2 = edge_point(to_key, to_side, offset=gap)
    arrow(ax, x1, y1, x2, y2, label=label, color=color, lw=lw, dashed=dashed)

# ══════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════
ax.text(10, 15.4,
        "CIF Boundary Detector Adapter — Training Pipeline",
        ha='center', va='center', fontsize=18, fontweight='bold', zorder=6,
        path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])

# ══════════════════════════════════════════════════════════════
#  STEP 1: DATA EXTRACTION (step1_extract.py)
# ══════════════════════════════════════════════════════════════
ax.text(2.5, 14.2, "STEP 1: Data Extraction", ha='left', va='center',
        fontsize=14, fontweight='bold', color="#2C3E50", zorder=6)

# Row 1: Input Dataset
box(ax, 'libritts', 3.5, 13.2, 4.5, 0.8,
    "LibriTTS Dataset", "train-clean-100 (24kHz WAV files)", C_DATASET, fontsize=14)

# Row 2: Audio Processing
box(ax, 'audio1', 1.8, 11.8, 2.2, 0.7,
    "Audio 1", "Random WAV", C_DATA, fontsize=13)

box(ax, 'silence', 3.5, 11.8, 2.2, 0.7,
    "Silence", "1-3 seconds", C_DATA, fontsize=13)

box(ax, 'audio2', 5.2, 11.8, 2.2, 0.7,
    "Audio 2", "Random WAV", C_DATA, fontsize=13)

# Row 3: Concatenation
box(ax, 'concat', 3.5, 10.5, 4.0, 0.75,
    "Concatenate Audio", "Resample to 16kHz", C_DATA, fontsize=13)

# Row 4: Label Generation
box(ax, 'rawlabels', 3.5, 9.3, 4.0, 0.75,
    "Raw Labels", "[1,1,1...] + [0,0,0...] + [1,1,1...]", C_DATA, fontsize=13)

# Row 5: SeamlessM4T Encoder
box(ax, 'encoder_extract', 3.5, 7.8, 5.0, 1.0,
    "SeamlessM4T Speech Encoder", "Extract hidden states [T, 1024]", C_MODEL, fontsize=14, bold=True)

# Row 6: Label Alignment
box(ax, 'align', 3.5, 6.3, 4.5, 0.85,
    "F.interpolate (Linear)", "Align labels to encoder seq length T", C_DATA, fontsize=13)

# Row 7: Save Features
box(ax, 'save_features', 1.5, 4.8, 2.5, 0.75,
    "sample_i_features.pt", "[T, 1024]", C_OUTPUT, fontsize=12)

box(ax, 'save_labels', 5.5, 4.8, 2.5, 0.75,
    "sample_i_labels.pt", "[T]", C_OUTPUT, fontsize=12)

# ══════════════════════════════════════════════════════════════
#  STEP 2: TRAINING (step2_train.py)
# ══════════════════════════════════════════════════════════════
ax.text(11.5, 14.2, "STEP 2: Adapter Training", ha='left', va='center',
        fontsize=14, fontweight='bold', color="#2C3E50", zorder=6)

# Row 1: Dataset Splits
box(ax, 'train_split', 10.5, 13.2, 2.5, 0.7,
    "Train Split", "5000 pairs", C_DATASET, fontsize=13)

box(ax, 'val_split', 13.5, 13.2, 2.5, 0.7,
    "Val Split", "500 pairs", C_DATASET, fontsize=13)

box(ax, 'test_split', 16.5, 13.2, 2.5, 0.7,
    "Test Split", "500 pairs", C_DATASET, fontsize=13)

# Row 2: DataLoader
box(ax, 'dataloader', 13.5, 11.8, 5.0, 0.85,
    "DataLoader + pad_collate_fn", "Batch=16, Pad variable sequences", C_TRAINING, fontsize=13)

# Row 3: CIF Adapter Architecture
box(ax, 'adapter_arch', 13.5, 10.2, 6.0, 1.3,
    "CIF Boundary Detector (MLP)", 
    "1024 → 512 (LN+ReLU+Drop) → 256 (LN+ReLU+Drop) → 1 (Sigmoid)",
    C_ADAPTER, fontsize=14, bold=True, highlight=True)

# Row 4: Forward Pass
box(ax, 'forward', 13.5, 8.5, 4.5, 0.8,
    "Forward Pass", "outputs = model(features)", C_TRAINING, fontsize=13)

# Row 5: Loss Computation
box(ax, 'loss', 13.5, 7.2, 4.0, 0.8,
    "Binary Cross-Entropy Loss", "BCELoss(outputs, labels)", C_LOSS, fontsize=13)

# Row 6: Optimization
box(ax, 'optimizer', 13.5, 5.9, 4.5, 0.8,
    "AdamW Optimizer", "lr=1e-3, weight_decay=1e-4", C_TRAINING, fontsize=13)

# Row 7: Training Loop
box(ax, 'train_loop', 13.5, 4.6, 4.0, 0.75,
    "Training Loop", "10 epochs", C_TRAINING, fontsize=13)

# Row 8: Saved Model
box(ax, 'saved_model', 13.5, 3.3, 4.0, 0.75,
    "boundary_adapter.pt", "Trained weights", C_OUTPUT, fontsize=13, bold=True)

# ══════════════════════════════════════════════════════════════
#  STEP 3: INFERENCE (infer_chunked.py)
# ══════════════════════════════════════════════════════════════
ax.text(2.5, 3.0, "STEP 3: Chunked Inference", ha='left', va='center',
        fontsize=14, fontweight='bold', color="#2C3E50", zorder=6)

# Row 1: Input Audio
box(ax, 'input_audio', 3.5, 2.0, 3.5, 0.7,
    "Input Audio", "test.wav (16kHz)", C_DATA, fontsize=13)

# Row 2: Wrapper
box(ax, 'wrapper', 3.5, 0.8, 4.5, 0.9,
    "SeamlessChunkedWrapper", "Base Model + Loaded Adapter", C_INFERENCE, fontsize=13, bold=True)

# ══════════════════════════════════════════════════════════════
#  ARROWS - STEP 1
# ══════════════════════════════════════════════════════════════
GAP = 0.10

arrow_between(ax, 'libritts', 'bottom', 'audio1', 'top', gap=GAP)
arrow_between(ax, 'libritts', 'bottom', 'silence', 'top', gap=GAP)
arrow_between(ax, 'libritts', 'bottom', 'audio2', 'top', gap=GAP)

arrow_between(ax, 'audio1', 'bottom', 'concat', 'left', gap=GAP)
arrow_between(ax, 'silence', 'bottom', 'concat', 'top', gap=GAP)
arrow_between(ax, 'audio2', 'bottom', 'concat', 'right', gap=GAP)

arrow_between(ax, 'concat', 'bottom', 'rawlabels', 'top', 
              label="Generate labels", gap=GAP)

arrow_between(ax, 'concat', 'bottom', 'encoder_extract', 'top',
              label="Extract features", color="#27AE60", lw=2.5, gap=GAP)

arrow_between(ax, 'rawlabels', 'bottom', 'align', 'top', gap=GAP)
arrow_between(ax, 'encoder_extract', 'bottom', 'align', 'top',
              label="Seq len T", color="#E67E22", lw=2.2, gap=GAP)

arrow_between(ax, 'align', 'bottom', 'save_features', 'top', gap=GAP)
arrow_between(ax, 'align', 'bottom', 'save_labels', 'top', gap=GAP)

# ══════════════════════════════════════════════════════════════
#  ARROWS - STEP 2
# ══════════════════════════════════════════════════════════════
arrow_between(ax, 'train_split', 'bottom', 'dataloader', 'top', gap=GAP)
arrow_between(ax, 'val_split', 'bottom', 'dataloader', 'top', gap=GAP)

arrow_between(ax, 'dataloader', 'bottom', 'adapter_arch', 'top',
              label="Batched features", color="#3498DB", lw=2.5, gap=GAP)

arrow_between(ax, 'adapter_arch', 'bottom', 'forward', 'top', gap=GAP)

arrow_between(ax, 'forward', 'bottom', 'loss', 'top',
              label="predictions", gap=GAP)

arrow_between(ax, 'loss', 'bottom', 'optimizer', 'top',
              label="loss.backward()", color="#E74C3C", lw=2.5, gap=GAP)

arrow_between(ax, 'optimizer', 'bottom', 'train_loop', 'top',
              label="optimizer.step()", gap=GAP)

arrow_between(ax, 'train_loop', 'bottom', 'saved_model', 'top', gap=GAP)

# Connection from saved features to training
x1, y1 = edge_point('save_features', 'right', offset=GAP)
x2, y2 = edge_point('train_split', 'left', offset=GAP)
arrow(ax, x1, y1, x2, y2, label="Load dataset", 
      color="#8E44AD", lw=2.5, dashed=True)

# ══════════════════════════════════════════════════════════════
#  ARROWS - STEP 3
# ══════════════════════════════════════════════════════════════
arrow_between(ax, 'input_audio', 'bottom', 'wrapper', 'top', gap=GAP)

x1, y1 = edge_point('saved_model', 'left', offset=GAP)
x2, y2 = edge_point('wrapper', 'right', offset=GAP)
arrow(ax, x1, y1, x2, y2, label="Load weights",
      color="#C0392B", lw=3.0, dashed=True)

# ══════════════════════════════════════════════════════════════
#  ANNOTATIONS
# ══════════════════════════════════════════════════════════════

# Step 1 Details Box
ann1 = FancyBboxPatch((7.2, 11.5), 3.0, 2.8,
                      boxstyle="round,pad=0.15",
                      linewidth=1.4, edgecolor="#34495E",
                      facecolor="#ECF0F1", alpha=0.93, zorder=4)
ax.add_patch(ann1)
details1 = [
    ("Data Generation", 'bold', 12),
    ("• 5000 train pairs", 'normal', 10.5),
    ("• 500 val pairs", 'normal', 10.5),
    ("• 500 test pairs", 'normal', 10.5),
    ("• Random silence: 1-3s", 'normal', 10.5),
    ("• Resample: 16kHz", 'normal', 10.5),
]
for i, (txt, fw, fs) in enumerate(details1):
    ax.text(8.7, 13.8 - i*0.42, txt,
            ha='center', va='center',
            fontsize=fs, fontweight=fw, color="#2C3E50", zorder=6)

# Step 2 Details Box
ann2 = FancyBboxPatch((17.2, 10.5), 2.6, 2.5,
                      boxstyle="round,pad=0.15",
                      linewidth=1.4, edgecolor="#34495E",
                      facecolor="#ECF0F1", alpha=0.93, zorder=4)
ax.add_patch(ann2)
details2 = [
    ("Training Config", 'bold', 12),
    ("• Batch: 16", 'normal', 10.5),
    ("• Epochs: 10", 'normal', 10.5),
    ("• LR: 1e-3", 'normal', 10.5),
    ("• Dropout: 0.3", 'normal', 10.5),
    ("• Loss: BCE", 'normal', 10.5),
]
for i, (txt, fw, fs) in enumerate(details2):
    ax.text(18.5, 12.5 - i*0.38, txt,
            ha='center', va='center',
            fontsize=fs, fontweight=fw, color="#2C3E50", zorder=6)

# Inference Details Box
ann3 = FancyBboxPatch((7.2, 0.2), 3.2, 2.2,
                      boxstyle="round,pad=0.15",
                      linewidth=1.4, edgecolor="#34495E",
                      facecolor="#ECF0F1", alpha=0.93, zorder=4)
ax.add_patch(ann3)
details3 = [
    ("Inference Mode", 'bold', 12),
    ("• Load adapter weights", 'normal', 10.5),
    ("• Detect boundaries", 'normal', 10.5),
    ("• Threshold: 0.2", 'normal', 10.5),
    ("• Chunk on silence", 'normal', 10.5),
    ("• Stream output", 'normal', 10.5),
]
for i, (txt, fw, fs) in enumerate(details3):
    ax.text(8.8, 2.1 - i*0.35, txt,
            ha='center', va='center',
            fontsize=fs, fontweight=fw, color="#2C3E50", zorder=6)

# Innovation Badge
star_x, star_y = 13.5, 12.0
ax.scatter([star_x], [star_y], s=600, marker='*',
           color='#FFD700', edgecolors='#C0392B', linewidths=2, zorder=7)
ax.text(star_x, star_y + 0.5, "TRAINED ADAPTER", ha='center', va='center',
        fontsize=11, fontweight='bold', color='#C0392B', zorder=7)

# ══════════════════════════════════════════════════════════════
#  LEGEND
# ══════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=C_ADAPTER, edgecolor='#C0392B', lw=2,
                   label='CIF Adapter (Trainable)'),
    mpatches.Patch(facecolor=C_DATASET, edgecolor='#2C3E50',
                   label='Dataset / Data Split'),
    mpatches.Patch(facecolor=C_DATA, edgecolor='#2C3E50',
                   label='Data Processing'),
    mpatches.Patch(facecolor=C_MODEL, edgecolor='#2C3E50',
                   label='Frozen Model (Feature Extraction)'),
    mpatches.Patch(facecolor=C_TRAINING, edgecolor='#2C3E50',
                   label='Training Components'),
    mpatches.Patch(facecolor=C_LOSS, edgecolor='#2C3E50',
                   label='Loss / Optimization'),
    mpatches.Patch(facecolor=C_INFERENCE, edgecolor='#2C3E50',
                   label='Inference Wrapper'),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor='#2C3E50',
                   label='Saved Outputs'),
]
ax.legend(handles=legend_items, loc='lower right',
          fontsize=11, framealpha=0.96, ncol=2,
          title="Pipeline Components", title_fontsize=12,
          bbox_to_anchor=(0.99, 0.01))

plt.tight_layout()
plt.savefig("cif_adapter_training_pipeline.pdf", dpi=300, bbox_inches='tight')
plt.savefig("cif_adapter_training_pipeline.png", dpi=300, bbox_inches='tight')
print("✓ Saved: cif_adapter_training_pipeline.pdf / .png")
plt.show()
