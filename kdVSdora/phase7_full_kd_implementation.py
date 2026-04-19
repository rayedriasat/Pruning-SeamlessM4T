"""
Phase 7: Full End-to-End Knowledge Distillation
================================================

Single-phase training with teacher + student both on GPU.
Uses memory optimization techniques to fit in T4 (16GB VRAM).

Target: 95-100% quality recovery
Memory: 13-15 GB VRAM (fits in T4!)
Time: 6-8 hours (3000 steps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    SeamlessM4Tv2Model,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os

# Try to import 8-bit optimizer (memory efficient)
try:
    from bitsandbytes.optim import Adam8bit
    USE_8BIT = True
    print("✅ Using 8-bit Adam optimizer (50% memory savings)")
except ImportError:
    print("⚠️  bitsandbytes not found. Install with: pip install bitsandbytes")
    print("⚠️  Falling back to standard Adam (may use more memory)")
    USE_8BIT = False

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

# Training hyperparameters
TRAINING_CONFIG = {
    'learning_rate': 5e-5,
    'num_train_steps': 3000,
    'warmup_steps': 300,
    'batch_size': 1,  # Keep small for memory
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'save_every': 500,
}

# Distillation hyperparameters
DISTILL_CONFIG = {
    'temperature': 2.0,
    'alpha_text': 0.3,   # Weight for text decoder loss
    'alpha_t2u': 0.5,    # Weight for T2U soft targets
    'alpha_hard': 0.2,   # Weight for T2U hard targets
}

# Data config
SRC_LANG = 'eng'
TGT_LANG = 'ben'
SAMPLE_RATE = 16000

# ══════════════════════════════════════════════════════════════════════════════
# Full KD Loss Function
# ══════════════════════════════════════════════════════════════════════════════

def compute_full_kd_loss(
    teacher_model,
    student_model,
    audio_features,
    text_labels,
    temperature=2.0,
    alpha_text=0.3,
    alpha_t2u=0.5,
    alpha_hard=0.2,
):
    """
    Full end-to-end knowledge distillation loss.
    
    Combines:
    1. Text decoder loss (S2TT cross-entropy)
    2. T2U distillation loss (soft targets from teacher)
    3. Hard target loss (stability)
    
    Args:
        teacher_model: Full SeamlessM4Tv2Model (frozen)
        student_model: Pruned SeamlessM4Tv2Model (trainable)
        audio_features: Input audio [B, T, 80]
        text_labels: Bengali text token IDs [B, L]
        temperature: Softmax temperature for distillation
        alpha_text: Weight for text loss
        alpha_t2u: Weight for T2U soft loss
        alpha_hard: Weight for T2U hard loss
    
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual losses
    """
    # ── Teacher forward (frozen) ────────────────────────────────────────────
    with torch.no_grad():
        teacher_model.eval()
        
        # Speech encoder
        teacher_enc_out = teacher_model.speech_encoder(
            input_features=audio_features
        )
        teacher_enc_hidden = teacher_enc_out.last_hidden_state
        
        # Create attention mask for encoder hidden states
        B, T_enc, H = teacher_enc_hidden.shape
        teacher_enc_mask = torch.ones(
            (B, T_enc),
            dtype=torch.long,
            device=audio_features.device,
        )
        
        # Text decoder
        teacher_dec_out = teacher_model.text_decoder(
            encoder_hidden_states=teacher_enc_hidden,
            encoder_attention_mask=teacher_enc_mask,
            labels=text_labels,
        )
        teacher_text_hidden = teacher_dec_out.last_hidden_state
        
        # T2U model
        teacher_t2u_out = teacher_model.t2u_model(
            inputs_embeds=teacher_text_hidden,
            return_dict=True,
        )
        teacher_unit_logits = teacher_t2u_out.logits  # [B, T_units, 10082]
    
    # ── Student forward (trainable) ─────────────────────────────────────────
    student_model.train()
    
    # Speech encoder
    student_enc_out = student_model.speech_encoder(
        input_features=audio_features
    )
    student_enc_hidden = student_enc_out.last_hidden_state
    
    # Create attention mask
    B, T_enc, H = student_enc_hidden.shape
    student_enc_mask = torch.ones(
        (B, T_enc),
        dtype=torch.long,
        device=audio_features.device,
    )
    
    # Text decoder
    student_dec_out = student_model.text_decoder(
        encoder_hidden_states=student_enc_hidden,
        encoder_attention_mask=student_enc_mask,
        labels=text_labels,
    )
    student_text_loss = student_dec_out.loss  # S2TT cross-entropy
    student_text_hidden = student_dec_out.last_hidden_state
    
    # T2U model
    student_t2u_out = student_model.t2u_model(
        inputs_embeds=student_text_hidden,
        return_dict=True,
    )
    student_unit_logits = student_t2u_out.logits  # [B, T_units, 10082]
    
    # ── Loss computation ────────────────────────────────────────────────────
    # 1. Text loss (S2TT cross-entropy)
    text_loss = student_text_loss
    
    # 2. T2U distillation loss (soft targets with temperature)
    soft_targets = F.softmax(teacher_unit_logits / temperature, dim=-1)
    soft_pred = F.log_softmax(student_unit_logits / temperature, dim=-1)
    
    t2u_soft_loss = F.kl_div(
        soft_pred,
        soft_targets,
        reduction='batchmean',
    ) * (temperature ** 2)
    
    # 3. Hard target loss (stability)
    hard_targets = teacher_unit_logits.argmax(dim=-1)
    t2u_hard_loss = F.cross_entropy(
        student_unit_logits.view(-1, student_unit_logits.size(-1)),
        hard_targets.view(-1),
        ignore_index=-100,
    )
    
    # Combined loss
    total_loss = (
        alpha_text * text_loss +
        alpha_t2u * t2u_soft_loss +
        alpha_hard * t2u_hard_loss
    )
    
    # Return loss dict for logging
    loss_dict = {
        'text': text_loss.item(),
        't2u_soft': t2u_soft_loss.item(),
        't2u_hard': t2u_hard_loss.item(),
        'total': total_loss.item(),
    }
    
    return total_loss, loss_dict


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


def prepare_batch(samples, processor, device):
    """
    Prepare a batch for training.
    
    Args:
        samples: List of sample dicts
        processor: SeamlessM4Tv2Processor
        device: torch device
    
    Returns:
        audio_features: Audio features [B, T, 80]
        text_labels: Bengali text token IDs [B, L]
    """
    # Extract audio arrays and text
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
    audio_features = inputs['input_features'].to(device)
    text_labels = labels.to(device)
    
    # Replace padding with -100 (ignore in loss)
    text_labels[text_labels == processor.tokenizer.pad_token_id] = -100
    
    return audio_features, text_labels


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_full_kd(
    teacher_model,
    student_model,
    processor,
    train_samples,
    training_config=None,
    distill_config=None,
    checkpoint_dir='./checkpoints',
):
    """
    Train with full end-to-end knowledge distillation.
    
    Args:
        teacher_model: Full model (frozen)
        student_model: Pruned model (trainable)
        processor: SeamlessM4Tv2Processor
        train_samples: List of training samples
        training_config: Training hyperparameters
        distill_config: Distillation hyperparameters
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        student_model: Trained model
        losses: List of loss dicts
    """
    if training_config is None:
        training_config = TRAINING_CONFIG.copy()
    if distill_config is None:
        distill_config = DISTILL_CONFIG.copy()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Enable gradient checkpointing (saves memory)
    student_model.gradient_checkpointing_enable()
    print("✅ Gradient checkpointing enabled (25% memory savings)")
    
    # Setup optimizer
    if USE_8BIT:
        optimizer = Adam8bit(
            student_model.parameters(),
            lr=training_config['learning_rate'],
            betas=(0.9, 0.999),
        )
    else:
        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=training_config['learning_rate'],
        )
    
    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config['warmup_steps'],
        num_training_steps=training_config['num_train_steps'],
    )
    
    # Print memory usage
    print(f"\n📊 Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"  Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
    
    # Training loop
    losses = []
    
    print(f"\n🚀 Starting Full KD Training:")
    print(f"  Steps: {training_config['num_train_steps']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Temperature: {distill_config['temperature']}")
    print(f"  Loss weights: text={distill_config['alpha_text']}, "
          f"t2u_soft={distill_config['alpha_t2u']}, "
          f"t2u_hard={distill_config['alpha_hard']}")
    
    for step in tqdm(range(training_config['num_train_steps']), desc="Training"):
        # Sample random batch
        batch_indices = np.random.choice(
            len(train_samples),
            size=training_config['batch_size'],
            replace=False,
        )
        batch_samples = [train_samples[i] for i in batch_indices]
        
        # Prepare batch
        audio_features, text_labels = prepare_batch(
            batch_samples,
            processor,
            DEVICE,
        )
        
        # Compute loss
        loss, loss_dict = compute_full_kd_loss(
            teacher_model,
            student_model,
            audio_features,
            text_labels,
            temperature=distill_config['temperature'],
            alpha_text=distill_config['alpha_text'],
            alpha_t2u=distill_config['alpha_t2u'],
            alpha_hard=distill_config['alpha_hard'],
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            student_model.parameters(),
            training_config['max_grad_norm'],
        )
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Log
        losses.append(loss_dict)
        
        if (step + 1) % 50 == 0:
            avg_losses = {
                k: np.mean([d[k] for d in losses[-50:]])
                for k in losses[-1].keys()
            }
            lr = scheduler.get_last_lr()[0]
            mem_gb = torch.cuda.memory_allocated() / 1e9
            
            print(f"\n📈 Step {step+1}/{training_config['num_train_steps']}")
            print(f"  Text: {avg_losses['text']:.4f}")
            print(f"  T2U Soft: {avg_losses['t2u_soft']:.4f}")
            print(f"  T2U Hard: {avg_losses['t2u_hard']:.4f}")
            print(f"  Total: {avg_losses['total']:.4f}")
            print(f"  LR: {lr:.2e}")
            print(f"  VRAM: {mem_gb:.2f} GB")
        
        # Save checkpoint
        if (step + 1) % training_config['save_every'] == 0:
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"full_kd_step{step+1:06d}.pt"
            )
            torch.save({
                'step': step + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': losses,
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")
    
    print("\n✅ Full KD training complete!")
    return student_model, losses


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function for full KD.
    """
    print("="*80)
    print("Phase 7: Full End-to-End Knowledge Distillation")
    print("="*80)
    
    # 1. Load teacher model (frozen)
    print("\n[1/5] Loading teacher model (frozen)...")
    teacher_model = SeamlessM4Tv2Model.from_pretrained(
        'facebook/seamless-m4t-v2-large',
        torch_dtype=DTYPE,
    ).to(DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    teacher_mem = torch.cuda.memory_allocated() / 1e9
    print(f"✅ Teacher loaded: {teacher_mem:.2f} GB VRAM")
    
    # 2. Load student model (trainable)
    print("\n[2/5] Loading student model (trainable)...")
    student_model = SeamlessM4Tv2Model.from_pretrained(
        './models/phase6_pruned',  # Adjust path as needed
        torch_dtype=DTYPE,
    ).to(DEVICE)
    student_model.train()
    
    total_mem = torch.cuda.memory_allocated() / 1e9
    student_mem = total_mem - teacher_mem
    print(f"✅ Student loaded: {student_mem:.2f} GB VRAM")
    print(f"✅ Total: {total_mem:.2f} GB VRAM")
    
    processor = AutoProcessor.from_pretrained('./models/phase6_pruned')
    
    # 3. Load training data
    print("\n[3/5] Loading training data...")
    train_samples = load_fleurs_training_data(max_samples=2554)
    
    # 4. Train
    print("\n[4/5] Training...")
    student_model, losses = train_full_kd(
        teacher_model,
        student_model,
        processor,
        train_samples,
        checkpoint_dir='./checkpoints/full_kd',
    )
    
    # 5. Save final model
    print("\n[5/5] Saving final model...")
    output_dir = './models/phase7_full_kd'
    os.makedirs(output_dir, exist_ok=True)
    student_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")
    
    print("\n" + "="*80)
    print("Phase 7 complete! Run final benchmark to verify results.")
    print("="*80)


if __name__ == '__main__':
    main()
