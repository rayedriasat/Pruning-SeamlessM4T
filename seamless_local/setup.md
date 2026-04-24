# SeamlessM4T v2 Compressed — Local Inference & Benchmark
Adapted from the Kaggle Phase 7/8 notebook for **local RTX 3050 4 GB** use.

**Same folder structure as Kaggle** — only `ON_KAGGLE=False` and the root path is `./local_working` instead of `/kaggle/working`.

---
## 🛠️ One-time Setup (run in terminal before opening this notebook)

### 1. Install uv
```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment
```bash
uv venv --python 3.12
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

### 3. Install PyTorch CUDA if available (for RTX 3050)
```bash
uv pip install torch --torch-backend=auto
```

### 4. Install all other packages
```bash
uv pip install transformers datasets accelerate peft librosa soundfile sounddevice requests pandas sacrebleu evaluate sentencepiece safetensors matplotlib seaborn notebook huggingface_hub
```

### 5. Download your model from Google Drive
Your model is at `cse465v5/models/phase7_dora_merged_v1` on Drive.

**Option A — rclone (same rclone.conf from Kaggle secret):**
```bash
# Place your rclone.conf at ~/.config/rclone/rclone.conf
mkdir -p local_working/models/phase7_dora_merged_v1
rclone copy gdrive:cse465v5/models/phase7_dora_merged_v1 ./local_working/models/phase7_dora_merged_v1 --progress
```

**Option B — Google Drive browser download:**
- Open drive.google.com → `cse465v5/models/phase7_dora_merged_v1/`
- Download all files and place them in `./local_working/models/phase7_dora_merged_v1/`

### 6. FLEURS parquet files (optional — Cell 7 auto-downloads if missing)
If you already synced them on Kaggle:
```bash
mkdir -p local_working/fleurs_parquet
rclone copy gdrive:cse465v5/fleurs_parquet ./local_working/fleurs_parquet --progress
```
Otherwise Cell 7 will download them fresh from HuggingFace (no `trust_remote_code` needed).

### 7. Launch Jupyter
```bash
jupyter notebook seamless_local_inference.ipynb
```

> **VRAM note:** 1095.9M params × float16 ≈ 2.2 GB VRAM. Your RTX 3050 4 GB handles it with ~1.8 GB to spare.