"""
Phase 7a: DoRA Fine-Tuning for Text Recovery
============================================

This script implements DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tuning
for the speech encoder and text decoder to recover text translation quality.

Target: BLEU 35-40 (90-95% recovery from baseline 12.21)
Memory: 8-10 GB (fits in Kaggle T4)
Time: 2-3 hours (2000 steps)
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    SeamlessM4Tv2Model,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

# DoRA hyperparameters
DORA_CONFIG = {
    'r': 8,                    # Rank (lower = faster, higher = more capacity)
    'lora_alpha': 16,          # Scaling factor (typically 2×r)
    'lora_dropout': 0.05,      # Dropout for regularization
    'bias': 'none',            # Don't adapt bias terms
    'task_type': 'SEQ_2_SEQ_LM',
    'use_dora': True,          # Enable DoRA (magnitude + direction decomposition)
}

# Training hyperparameters
TRAINING_CONFIG = {
    'learning_rate': 5e-5,
    'num_train_steps': 2000,
    'warmup_steps': 200,
    'batch_size': 2,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'save_every': 200,
}

# Data config
SRC_LANG = 'eng'
TGT_LANG = 'ben'
SAMPLE_RATE = 16000

# ══════════════════════════════════════════════════════════════════════════════
# DoRA Target Modules
# ══════════════════════════════════════════════════════════════════════════════

def get_dora_target_modules():
    """
    Define which modules to apply DoRA to.
    
    Strategy:
    - Speech encoder: All attention + FFN layers
    - Text decoder: All attention + FFN layers
    - Skip: T2U model (will be trained in Phase 7b)
    """
    target_modules = []
    
    # Speech encoder (Conformer layers)
    # Pattern: speech_encoder.encoder.layers.{0-13}.{module}
    for layer_idx in range(14):  # Assuming 14 layers after pruning
        target_modules.extend([
            f"speech_encoder.encoder.layers.{layer_idx}.self_attn.q_proj",
            f"speech_encoder.encoder.layers.{layer_idx}.self_attn.k_proj",
            f"speech_encoder.encoder.layers.{layer_idx}.self_attn.v_proj",
            f"speech_encoder.encoder.layers.{layer_idx}.self_attn.out_proj",
            f"speech_encoder.encoder.layers.{layer_idx}.ffn.fc1",
            f"speech_encoder.encoder.layers.{layer_idx}.ffn.fc2",
        ])
    
    # Text decoder (Transformer layers)
    # Pattern: text_decoder.layers.{0-11}.{module}
    for layer_idx in range(12):  # Assuming 12 layers after pruning
        target_modules.extend([
            f"text_decoder.layers.{layer_idx}.self_attn.q_proj",
            f"text_decoder.layers.{layer_idx}.self_attn.k_proj",
            f"text_decoder.layers.{layer_idx}.self_attn.v_proj",
            f"text_decoder.layers.{layer_idx}.self_attn.out_proj",
            f"text_decoder.layers.{layer_idx}.encoder_attn.q_proj",
            f"text_decoder.layers.{layer_idx}.encoder_attn.k_proj",
            f"text_decoder.layers.{layer_idx}.encoder_attn.v_proj",
            f"text_decoder.layers.{layer_idx}.encoder_attn.out_proj",
            f"text_decoder.layers.{layer_idx}.ffn.fc1",
            f"text_decoder.layers.{layer_idx}.ffn.fc2",
        ])
    
    return target_modules


def inject_dora(model, config=None):
    """
    Inject DoRA adapters into the model.
    
    Args:
        model: Pruned SeamlessM4Tv2Model
        config: DoRA configuration dict (uses DORA_CONFIG if None)
    
    Returns:
        model_with_dora: Model with DoRA adapters
    """
    if config is None:
        config = DORA_CONFIG.copy()
    
    # Get target modules
    target_modules = get_dora_target_modules()
    config['target_modules'] = target_modules
    
    # Create LoRA config
    lora_config = LoraConfig(**config)
    
    # Apply PEFT
    model_with_dora = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(
        p.numel() for p in model_with_dora.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model_with_dora.parameters())
    
    print(f"DoRA injection complete:")
    print(f"  Trainable params: {trainable_params/1e6:.1f}M")
    print(f"  Total params: {total_params/1e6:.1f}M")
    print(f"  Trainable %: {100*trainable_params/total_params:.2f}%")
    
    return model_with_dora


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_fleurs_training_data(max_samples=2554):
    """
    Load FLEURS eng→ben training pairs.
    
    Returns:
        List of dicts: {audio_array, reference_text}
    """
    print(f"Loading FLEURS training data (max {max_samples} samples)...")
    
    # Load from HuggingFace
    en_ds = load_dataset('google/fleurs', 'en_us', split='train', streaming=True)
    bn_ds = load_dataset('google/fleurs', 'bn_in', split='train', streaming=True)
    
    # Build Bengali text map
    bn_text_map = {}
    for row in bn_ds:
        bn_text_map[row['id']] = row['transcription']
    
    # Pair English audio with Bengali text
    samples = []
    for row in en_ds:
        if len(samples) >= max_samples:
            break
        
        row_id = row['id']
        if row_id not in bn_text_map:
            continue
        
        audio_array = np.array(row['audio']['array'], dtype=np.float32)
        ref_text = bn_text_map[row_id]
        
        samples.append({
            'audio_array': audio_array,
            'reference_text': ref_text,
        })
    
    print(f"Loaded {len(samples)} training pairs.")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def prepare_batch(samples, processor, device):
    """
    Prepare a batch for S2TT training.
    
    Args:
        samples: List of sample dicts
        processor: SeamlessM4Tv2Processor
        device: torch device
    
    Returns:
        input_features: Audio features [B, T, 80]
        labels: Bengali text token IDs [B, L]
    """
    # Extract audio arrays
    audio_arrays = [s['audio_array'] for s in samples]
    ref_texts = [s['reference_text'] for s in samples]
    
    # Process audio
    inputs = processor(
        audios=audio_arrays,
        src_lang=SRC_LANG,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt',
        padding=True,
    )
    
    # Process text labels
    with processor.as_target_processor():
        labels = processor(
            text=ref_texts,
            return_tensors='pt',
            padding=True,
        ).input_ids
    
    # Move to device
    input_features = inputs['input_features'].to(device)
    labels = labels.to(device)
    
    # Replace padding token ID with -100 (ignore in loss)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    return input_features, labels


def train_phase7a(
    model,
    processor,
    train_samples,
    config=None,
    checkpoint_dir='./checkpoints',
):
    """
    Train Phase 7a: DoRA fine-tuning for text recovery.
    
    Args:
        model: Model with DoRA adapters
        processor: SeamlessM4Tv2Processor
        train_samples: List of training samples
        config: Training config dict
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        model: Trained model
        losses: List of training losses
    """
    if config is None:
        config = TRAINING_CONFIG.copy()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
    )
    
    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['num_train_steps'],
    )
    
    # Training loop
    model.train()
    losses = []
    
    print(f"\nStarting Phase 7a training:")
    print(f"  Steps: {config['num_train_steps']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Warmup steps: {config['warmup_steps']}")
    
    for step in tqdm(range(config['num_train_steps']), desc="Training"):
        # Sample random batch
        batch_indices = np.random.choice(
            len(train_samples),
            size=config['batch_size'],
            replace=False,
        )
        batch_samples = [train_samples[i] for i in batch_indices]
        
        # Prepare batch
        input_features, labels = prepare_batch(
            batch_samples,
            processor,
            DEVICE,
        )
        
        # Forward pass
        outputs = model(
            input_features=input_features,
            labels=labels,
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['max_grad_norm'],
        )
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Log
        losses.append(loss.item())
        
        if (step + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            lr = scheduler.get_last_lr()[0]
            print(f"\nStep {step+1}/{config['num_train_steps']}  "
                  f"Loss={avg_loss:.4f}  LR={lr:.2e}")
        
        # Save checkpoint
        if (step + 1) % config['save_every'] == 0:
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"phase7a_step{step+1:06d}.pt"
            )
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': losses,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    print("\nPhase 7a training complete!")
    return model, losses


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function for Phase 7a.
    """
    print("="*80)
    print("Phase 7a: DoRA Fine-Tuning for Text Recovery")
    print("="*80)
    
    # 1. Load pruned model from Phase 6
    print("\n[1/5] Loading Phase 6 pruned model...")
    model = SeamlessM4Tv2Model.from_pretrained(
        './models/phase6_pruned',  # Adjust path as needed
        torch_dtype=DTYPE,
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained('./models/phase6_pruned')
    
    # 2. Inject DoRA adapters
    print("\n[2/5] Injecting DoRA adapters...")
    model = inject_dora(model)
    
    # 3. Load training data
    print("\n[3/5] Loading training data...")
    train_samples = load_fleurs_training_data(max_samples=2554)
    
    # 4. Train
    print("\n[4/5] Training...")
    model, losses = train_phase7a(
        model,
        processor,
        train_samples,
        checkpoint_dir='./checkpoints/phase7a',
    )
    
    # 5. Save final model
    print("\n[5/5] Saving final model...")
    output_dir = './models/phase7a_dora'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    print("\n" + "="*80)
    print("Phase 7a complete! Next: Run Phase 7b (T2U distillation)")
    print("="*80)


if __name__ == '__main__':
    main()
