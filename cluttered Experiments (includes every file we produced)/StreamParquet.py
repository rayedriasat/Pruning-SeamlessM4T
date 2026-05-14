import torchaudio
from IPython.display import Audio as IPAudio, display

def play(audio, sr, label=''):
    if hasattr(audio, 'numpy'): audio = audio.squeeze().numpy()
    print(f'  {label}  ({len(audio)/sr:.1f}s | sr={sr})')
    display(IPAudio(audio, rate=int(sr)))

def save_audio(audio, sr, filename):
    path = f'{AUDIO_DIR}/{filename}'
    if not isinstance(audio, torch.Tensor): audio = torch.tensor(audio)
    torchaudio.save(path, audio.squeeze().unsqueeze(0).float().cpu(), sr)
    print(f'[audio] Saved {filename}')

print('Core utilities ready.')

# ══════════════════════════════════════════════════════════════════════════════
# RAM-Efficient Parquet Streaming Dataset
# Loads audio on-demand, not during initialization
# RAM: ~4MB for 4000 samples (vs ~20GB with old approach)
# ══════════════════════════════════════════════════════════════════════════════

import pyarrow.parquet as pq

class ParquetStreamingDataset:
    """Memory-efficient dataset that streams from parquet files."""
    
    def __init__(self, parquet_cache_dir, src_lang, tgt_lang, split='train', 
                 max_samples_per_pair=500):
        self.cache_dir = pathlib.Path(parquet_cache_dir)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.split = split
        self.max_samples = max_samples_per_pair
        self.samples = []
        self._build_index()
    
    def _build_index(self):
        """Build lightweight index (metadata only, no audio)."""
        src_files = sorted(self.cache_dir.glob(f'{M4T_FLEURS_MAP.get(self.src_lang)}/{self.split}_*.parquet'))
        tgt_files = sorted(self.cache_dir.glob(f'{M4T_FLEURS_MAP.get(self.tgt_lang)}/{self.split}_*.parquet'))
        
        if not src_files or not tgt_files:
            print(f'  WARNING: No parquet files for {self.src_lang}/{self.tgt_lang}')
            return
        
        # Read only ID columns (fast, <1MB RAM)
        src_ids = []
        for f in src_files:
            df = pd.read_parquet(f, columns=['id'])
            src_ids.extend([(str(f), idx, row_id) for idx, row_id in enumerate(df['id'])])
        
        tgt_ids = []
        for f in tgt_files:
            df = pd.read_parquet(f, columns=['id', 'transcription'])
            df = df[df['transcription'].str.strip().str.len() > 0]
            tgt_ids.extend([(str(f), idx, row_id, trans) 
                           for idx, (row_id, trans) in enumerate(zip(df['id'], df['transcription']))])
        
        # Create lookup dicts
        src_lookup = {row_id: (f, idx) for f, idx, row_id in src_ids}
        tgt_lookup = {row_id: (f, idx, trans) for f, idx, row_id, trans in tgt_ids}
        
        # Find matching IDs
        common_ids = set(src_lookup.keys()) & set(tgt_lookup.keys())
        
        # Build sample index (metadata only)
        for sample_id in list(common_ids)[:self.max_samples]:
            src_file, src_idx = src_lookup[sample_id]
            tgt_file, tgt_idx, tgt_text = tgt_lookup[sample_id]
            
            self.samples.append({
                'id': f"{self.src_lang}2{self.tgt_lang}_{sample_id}",
                'src_lang': self.src_lang,
                'tgt_lang': self.tgt_lang,
                'ref': tgt_text,
                '_src_file': src_file,
                '_src_idx': src_idx,
            })
        
        print(f'  Indexed {len(self.samples)} samples from {self.src_lang}→{self.tgt_lang}')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get sample with audio loaded on-demand."""
        sample = self.samples[idx].copy()
        
        # Load audio only when accessed
        if '_src_file' in sample:
            audio = self._load_audio_from_parquet(
                sample['_src_file'], 
                sample['_src_idx']
            )
            sample['wav'] = audio
            del sample['_src_file']
            del sample['_src_idx']
        
        return sample
    
    def _load_audio_from_parquet(self, parquet_file, row_idx):
        """Load single audio sample from parquet file."""
        table = pq.read_table(parquet_file, columns=['audio'])
        audio_cell = table.to_pandas().iloc[row_idx]['audio']
        return _load_wav(audio_cell)


class MultilingualStreamingDataset:
    """Combines multiple language pairs into a single streaming dataset."""
    
    def __init__(self, parquet_cache_dir, lang_pairs, split='train', 
                 max_samples_per_pair=25):
        self.datasets = []
        
        for src_lang, tgt_lang in lang_pairs:
            ds = ParquetStreamingDataset(
                parquet_cache_dir, src_lang, tgt_lang, split, max_samples_per_pair
            )
            if len(ds) > 0:
                self.datasets.append(ds)
        
        # Build flat index
        self.index = []
        for ds_idx, ds in enumerate(self.datasets):
            for sample_idx in range(len(ds)):
                self.index.append((ds_idx, sample_idx))
        
        print(f'\n✓ Multilingual dataset ready: {len(self.index)} total samples')
        print(f'  RAM usage: ~{len(self.index) * 0.001:.1f} MB (metadata only)')
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """Get sample from appropriate sub-dataset."""
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]
        ds_idx, sample_idx = self.index[idx]
        return self.datasets[ds_idx][sample_idx]
    
    def __iter__(self):
        """Allow iteration."""
        for i in range(len(self)):
            yield self[i]

print('✓ Streaming dataset classes ready.')



# ── Load Multilingual Eval Samples: En→X and X→En (all 5 languages) ─────────
# PLAN.md Section 5: 5 languages — EN, BN, ZH, AR, HI
N_EVAL_PER_PAIR = 25
EVAL_LANG_PAIRS = [
    ('eng', 'ben'), ('ben', 'eng'),  # English ↔ Bengali
    ('eng', 'cmn'), ('cmn', 'eng'),  # English ↔ Mandarin
    ('eng', 'arb'), ('arb', 'eng'),  # English ↔ Arabic
    ('eng', 'hin'), ('hin', 'eng'),  # English ↔ Hindi
]


# ── Load Multilingual Eval Samples: En→X and X→En (all 5 languages) ──────────
# STREAMING VERSION: Only loads audio when accessed
# RAM: ~200KB for 200 samples (vs ~1GB with old approach)

print('Loading evaluation samples (streaming mode)...')
eval_samples = MultilingualStreamingDataset(
    parquet_cache_dir=LOCAL_PARQUET_CACHE,
    lang_pairs=EVAL_LANG_PAIRS,
    split='test',
    max_samples_per_pair=N_EVAL_PER_PAIR
)

print(f'\n✓ Loaded {len(eval_samples)} multilingual eval samples')
print(f'  Language pairs: {len(EVAL_LANG_PAIRS)}')
print(f'  RAM usage: ~{len(eval_samples) * 0.001:.1f} MB (metadata only)')

# Test: Load one sample to verify it works
test_sample = eval_samples[26]
print(f'\n✓ Test sample loaded:')
print(f'  ID: {test_sample["id"]}')
print(f'  Audio shape: {test_sample["wav"].shape}')
print(f'  Reference: {test_sample["ref"][:50]}...')

play(test_sample["wav"], 16000, label='hello.wav')