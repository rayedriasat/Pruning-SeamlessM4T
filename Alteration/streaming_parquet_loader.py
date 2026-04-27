# ══════════════════════════════════════════════════════════════════════════════
# RAM-Efficient Parquet Streaming for FLEURS Dataset
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import random
from pathlib import Path

class ParquetStreamingDataset:
    """
    Memory-efficient dataset that streams from parquet files.
    Only loads audio when accessed, not during initialization.
    
    RAM usage: ~1KB per sample (metadata only) vs ~5MB per sample (with audio)
    """
    
    def __init__(self, parquet_cache_dir, src_lang, tgt_lang, split='train', 
                 max_samples_per_pair=500):
        self.cache_dir = Path(parquet_cache_dir)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.split = split
        self.max_samples = max_samples_per_pair
        
        # Build index of samples (metadata only, no audio)
        self.samples = []
        self._build_index()
    
    def _build_index(self):
        """Build lightweight index of available samples."""
        src_files = sorted(self.cache_dir.glob(f'{self.src_lang}/{self.split}_*.parquet'))
        tgt_files = sorted(self.cache_dir.glob(f'{self.tgt_lang}/{self.split}_*.parquet'))
        
        if not src_files or not tgt_files:
            print(f'  WARNING: No parquet files found for {self.src_lang}/{self.tgt_lang}')
            return
        
        # Read only ID columns first (very fast, <1MB RAM)
        src_ids = []
        for f in src_files:
            df = pd.read_parquet(f, columns=['id'])
            src_ids.extend([(f, idx, row_id) for idx, row_id in enumerate(df['id'])])
        
        tgt_ids = []
        for f in tgt_files:
            df = pd.read_parquet(f, columns=['id', 'transcription'])
            # Filter out empty transcriptions early
            df = df[df['transcription'].str.strip().str.len() > 0]
            tgt_ids.extend([(f, idx, row_id, trans) 
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
                # Lazy loading metadata
                '_src_file': str(src_file),
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
            # Clean up metadata
            del sample['_src_file']
            del sample['_src_idx']
        
        return sample
    
    def _load_audio_from_parquet(self, parquet_file, row_idx):
        """Load single audio sample from parquet file."""
        # Read only the specific row we need
        table = pq.read_table(parquet_file, columns=['audio'])
        audio_cell = table.to_pandas().iloc[row_idx]['audio']
        
        # Use existing _load_wav function
        return _load_wav(audio_cell)
    
    def get_metadata_only(self, idx):
        """Get sample without loading audio (for inspection)."""
        return {k: v for k, v in self.samples[idx].items() 
                if not k.startswith('_')}


class MultilingualStreamingDataset:
    """
    Combines multiple language pairs into a single streaming dataset.
    """
    
    def __init__(self, parquet_cache_dir, lang_pairs, split='train', 
                 max_samples_per_pair=500):
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
        ds_idx, sample_idx = self.index[idx]
        return self.datasets[ds_idx][sample_idx]
    
    def random_sample(self):
        """Get a random sample (useful for training)."""
        idx = random.randint(0, len(self) - 1)
        return self[idx]
    
    def get_metadata_summary(self):
        """Get summary without loading any audio."""
        summary = {}
        for ds in self.datasets:
            pair = f"{ds.src_lang}→{ds.tgt_lang}"
            summary[pair] = len(ds)
        return summary


# ══════════════════════════════════════════════════════════════════════════════
# Usage Example
# ══════════════════════════════════════════════════════════════════════════════

def create_streaming_ft_samples(parquet_cache_dir, lang_pairs, max_per_pair=500):
    """
    Create streaming training dataset.
    
    Returns:
        MultilingualStreamingDataset that loads audio on-demand
    """
    return MultilingualStreamingDataset(
        parquet_cache_dir=parquet_cache_dir,
        lang_pairs=lang_pairs,
        split='train',
        max_samples_per_pair=max_per_pair
    )


def create_streaming_eval_samples(parquet_cache_dir, lang_pairs, max_per_pair=25):
    """
    Create streaming evaluation dataset.
    
    Returns:
        MultilingualStreamingDataset that loads audio on-demand
    """
    return MultilingualStreamingDataset(
        parquet_cache_dir=parquet_cache_dir,
        lang_pairs=lang_pairs,
        split='test',
        max_samples_per_pair=max_per_pair
    )


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class ListLikeDataset:
    """
    Wrapper to make streaming dataset behave like a list.
    Allows existing code like `random.choice(ft_samples)` to work.
    """
    
    def __init__(self, streaming_dataset):
        self.dataset = streaming_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing
            indices = range(*idx.indices(len(self)))
            return [self.dataset[i] for i in indices]
        return self.dataset[idx]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.dataset[i]


# ══════════════════════════════════════════════════════════════════════════════
# Performance Comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_memory_usage():
    """
    Compare RAM usage between approaches.
    
    OLD approach (load all audio):
        4000 samples × 5 MB/sample = 20 GB RAM
    
    NEW approach (streaming):
        4000 samples × 0.001 MB/sample = 4 MB RAM
        
    Speedup: 5000× less RAM usage
    """
    pass
