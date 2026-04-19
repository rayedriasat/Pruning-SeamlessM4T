"""
Phase 7b: T2U Knowledge Distillation for Audio Recovery
=======================================================

This script implements on-the-fly knowledge distillation from the full teacher
model to the pruned student model, specifically targeting the T2U (Text-to-Unit)
module to recover audio output quality.

Key Innovation: Teacher on CPU, student on GPU → memory efficient!

Target: ASR-BLEU 25-30 (80-85% recovery)
Memory: 6-7 GB VRAM (fits in Kaggle T4)
Time: 3-4 hours (2000 steps)
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

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

# Distillation hyperparameters
DISTILL_CONFIG = {
    'temperature': 2.0,        # Softmax temperature (higher = softer)
    'alpha': 0.7,              # Weight for soft targets (1-alpha for hard)
    'learning_rate': 1e-4,     # Higher LR for T2U only
    'num_train_steps': 2000,
    'warmup_steps': 100,
    'batch_size': 2,
    'max_grad_norm': 1.0,
    'save_every': 200,
}

# Data config
SRC_LANG = 'eng'
TGT_LANG = 'ben'
SAMPLE_RATE = 16000

# ══════════════════════════════════════════════════════════════════════════════
# Distillation Loss Function
# ══════════════════════════════════════════════════════════════════════════════

class T2UDistillationLoss(nn.Module):
    """
    Knowledge distillation loss for T2U model.
    
    Combines:
    1. Soft targets (KL divergence with temperature scaling)
    2. Hard targets (cross-entropy for stability)
    """
    
    def __init__(self, temperature=2.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, hard_targets=None):
        """
        Compute distillation loss.
        
        Args:
            student_logits: [B, T, vocab_size] from student T2U
            teacher_logits: [B, T, vocab_size] from teacher T2U
            hard_targets: [B, T] optional hard unit labels
        
        Returns:
            loss: Scalar distillation loss
        """
        # Soft targets (KL divergence with temperature)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_pred = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        soft_loss = F.kl_div(
            soft_pred,
            soft_targets,
            reduction='batchmean',
        ) * (self.temperature ** 2)
        
        # Hard targets (optional, for stability)
        if hard_targets is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                hard_targets.view(-1),
                ignore_index=-100,
            )
        else:
            # Use teacher's argmax as hard targets
            hard_targets = teacher_logits.argmax(dim=-1)
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                hard_targets.view(-1),
                ignore_index=-100,
            )
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return loss, soft_loss, hard_loss


# ══════════════════════════════════════════════════════════════════════════════
# On-the-Fly Teacher Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_teacher_t2u_logits(teacher_model, audio_features_cpu):
    """
    Run teacher model on CPU to get T2U logits.
    
    Args:
        teacher_model: Full SeamlessM4Tv2Model on CPU
        audio_features_cpu: Audio features [B, T, 80] on CPU
    
    Returns:
        teacher_logits: T2U logits [B, T_units, 10082] on GPU
    """
    teacher_model.eval()
    
    # Teacher forward pass (CPU)
    # Step 1: Speech encoder
    teacher_enc_out = teacher_model.speech_encoder(
        input_features=audio_features_cpu
    )
    teacher_enc_hidden = teacher_enc_out.last_hidden_state  # [B, T_enc, 1024]
    
    # Step 2: Text decoder (get hidden states for T2U)
    # Create dummy decoder input (start token)
    batch_size = audio_features_cpu.shape[0]
    decoder_input_ids = torch.zeros(
        (batch_size, 1),
        dtype=torch.long,
        device='cpu',
    )
    
    # Create attention mask for encoder hidden states
    B, T_enc, H = teacher_enc_hidden.shape
    encoder_attention_mask = torch.ones(
        (B, T_enc),
        dtype=torch.long,
        device='cpu',
    )
    
    teacher_dec_out = teacher_model.text_decoder(
        encoder_hidden_states=teacher_enc_hidden,
        encoder_attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    teacher_text_hidden = teacher_dec_out.last_hidden_state  # [B, 1, 1024]
    
    # Step 3: T2U model (generate unit logits)
    teacher_t2u_out = teacher_model.t2u_model(
        inputs_embeds=teacher_text_hidden,
        return_dict=True,
    )
    teacher_logits = teacher_t2u_out.logits  # [B, T_units, 10082]
    
    # Move to GPU
    teacher_logits_gpu = teacher_logits.to('cuda')
    
    return teacher_logits_gpu


def get_student_t2u_logits(student_model, audio_features_gpu):
    """
    Run student model on GPU to get T2U logits.
    
    Args:
        student_model: Pruned SeamlessM4Tv2Model on GPU
        audio_features_gpu: Audio features [B, T, 80] on GPU
    
    Returns:
        student_logits: T2U logits [B, T_units, 10082] on GPU
    """
    # Student forward pass (GPU)
    # Step 1: Speech encoder (frozen, from Phase 7a)
    student_enc_out = student_model.speech_encoder(
        input_features=audio_features_gpu
    )
    student_enc_hidden = student_enc_out.last_hidden_state  # [B, T_enc, 1024]
    
    # Step 2: Text decoder (frozen, from Phase 7a)
    batch_size = audio_features_gpu.shape[0]
    decoder_input_ids = torch.zeros(
        (batch_size, 1),
        dtype=torch.long,
        device='cuda',
    )
    
    # Create attention mask
    B, T_enc, H = student_enc_hidden.shape
    encoder_attention_mask = torch.ones(
        (B, T_enc),
        dtype=torch.long,
        device='cuda',
    )
    
    student_dec_out = student_model.text_decoder(
        encoder_hidden_states=student_enc_hidden,
        encoder_attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    student_text_hidden = student_dec_out.last_hidden_state  # [B, 1, 1024]
    
    # Step 3: T2U model (trainable)
    student_t2u_out = student_model.t2u_model(
        inputs_embeds=student_text_hidden,
        return_dict=True,
    )
    student_logits = student_t2u_out.logits  # [B, T_units, 10082]
    
    return student_logits


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_fleurs_training_data(max_samples=2554):
    """
    Load FLEURS eng→ben training pairs.
    
    Returns:
        List of dicts: {audio_array}
    """
    print(f"Loading FLEURS training data (max {max_samples} samples)...")
    
    # Load from HuggingFace
    en_ds = load_dataset('google/fleurs', 'en_us', split='train', streaming=True)
    
    samples = []
    for row in en_ds:
        if len(samples) >= max_samples:
            break
        
        audio_array = np.array(row['audio']['array'], dtype=np.float32)
        
        samples.append({
            'audio_array': audio_array,
        })
    
    print(f"Loaded {len(samples)} training samples.")
    return samples


def prepare_batch(samples, processor, device):
    """
    Prepare a batch for T2U distillation.
    
    Args:
        samples: List of sample dicts
        processor: SeamlessM4Tv2Processor
        device: torch device
    
    Returns:
        input_features: Audio features [B, T, 80]
    """
    # Extract audio arrays
    audio_arrays = [s['audio_array'] for s in samples]
    
    # Process audio
    inputs = processor(
        audios=audio_arrays,
        src_lang=SRC_LANG,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt',
        padding=True,
    )
    
    # Move to device
    input_features = inputs['input_features'].to(device)
    
    return input_features


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def freeze_encoder_decoder(model):
    """
    Freeze speech encoder and text decoder (already recovered in Phase 7a).
    Only T2U model is trainable.
    """
    # Freeze encoder
    for param in model.speech_encoder.parameters():
        param.requires_grad = False
    
    # Freeze decoder
    for param in model.text_decoder.parameters():
        param.requires_grad = False
    
    # Unfreeze T2U
    for param in model.t2u_model.parameters():
        param.requires_grad = True
    
    # Print trainable params
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters:")
    print(f"  T2U params: {trainable_params/1e6:.1f}M")
    print(f"  Total params: {total_params/1e6:.1f}M")
    print(f"  Trainable %: {100*trainable_params/total_params:.2f}%")


def train_phase7b(
    student_model,
    teacher_model,
    processor,
    train_samples,
    config=None,
    checkpoint_dir='./checkpoints',
):
    """
    Train Phase 7b: T2U knowledge distillation.
    
    Args:
        student_model: Pruned model (GPU) from Phase 7a
        teacher_model: Full model (CPU)
        processor: SeamlessM4Tv2Processor
        train_samples: List of training samples
        config: Training config dict
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        student_model: Trained model
        losses: Dict of training losses
    """
    if config is None:
        config = DISTILL_CONFIG.copy()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Freeze encoder + decoder
    freeze_encoder_decoder(student_model)
    
    # Setup loss function
    distill_loss_fn = T2UDistillationLoss(
        temperature=config['temperature'],
        alpha=config['alpha'],
    )
    
    # Setup optimizer (only T2U parameters)
    optimizer = torch.optim.AdamW(
        student_model.t2u_model.parameters(),
        lr=config['learning_rate'],
    )
    
    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['num_train_steps'],
    )
    
    # Training loop
    student_model.train()
    teacher_model.eval()
    
    losses = {
        'total': [],
        'soft': [],
        'hard': [],
    }
    
    print(f"\nStarting Phase 7b training:")
    print(f"  Steps: {config['num_train_steps']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Alpha: {config['alpha']}")
    
    for step in tqdm(range(config['num_train_steps']), desc="Training"):
        # Sample random batch
        batch_indices = np.random.choice(
            len(train_samples),
            size=config['batch_size'],
            replace=False,
        )
        batch_samples = [train_samples[i] for i in batch_indices]
        
        # Prepare batch
        audio_features_gpu = prepare_batch(
            batch_samples,
            processor,
            'cuda',
        )
        audio_features_cpu = audio_features_gpu.cpu()
        
        # Get teacher logits (CPU → GPU)
        teacher_logits = get_teacher_t2u_logits(
            teacher_model,
            audio_features_cpu,
        )
        
        # Get student logits (GPU)
        student_logits = get_student_t2u_logits(
            student_model,
            audio_features_gpu,
        )
        
        # Compute distillation loss
        loss, soft_loss, hard_loss = distill_loss_fn(
            student_logits,
            teacher_logits,
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            student_model.t2u_model.parameters(),
            config['max_grad_norm'],
        )
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Log
        losses['total'].append(loss.item())
        losses['soft'].append(soft_loss.item())
        losses['hard'].append(hard_loss.item())
        
        if (step + 1) % 50 == 0:
            avg_total = np.mean(losses['total'][-50:])
            avg_soft = np.mean(losses['soft'][-50:])
            avg_hard = np.mean(losses['hard'][-50:])
            lr = scheduler.get_last_lr()[0]
            print(f"\nStep {step+1}/{config['num_train_steps']}  "
                  f"Total={avg_total:.4f}  Soft={avg_soft:.4f}  "
                  f"Hard={avg_hard:.4f}  LR={lr:.2e}")
        
        # Save checkpoint
        if (step + 1) % config['save_every'] == 0:
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"phase7b_step{step+1:06d}.pt"
            )
            torch.save({
                'step': step + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': losses,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    print("\nPhase 7b training complete!")
    return student_model, losses


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function for Phase 7b.
    """
    print("="*80)
    print("Phase 7b: T2U Knowledge Distillation for Audio Recovery")
    print("="*80)
    
    # 1. Load teacher model (CPU)
    print("\n[1/5] Loading teacher model (CPU)...")
    teacher_model = SeamlessM4Tv2Model.from_pretrained(
        'facebook/seamless-m4t-v2-large',
        torch_dtype=torch.float32,  # CPU uses fp32
    ).to('cpu')
    teacher_model.eval()
    print("Teacher model loaded on CPU")
    
    # 2. Load student model from Phase 7a (GPU)
    print("\n[2/5] Loading Phase 7a student model (GPU)...")
    student_model = SeamlessM4Tv2Model.from_pretrained(
        './models/phase7a_dora',  # Adjust path as needed
        torch_dtype=DTYPE,
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained('./models/phase7a_dora')
    print("Student model loaded on GPU")
    
    # 3. Load training data
    print("\n[3/5] Loading training data...")
    train_samples = load_fleurs_training_data(max_samples=2554)
    
    # 4. Train
    print("\n[4/5] Training...")
    student_model, losses = train_phase7b(
        student_model,
        teacher_model,
        processor,
        train_samples,
        checkpoint_dir='./checkpoints/phase7b',
    )
    
    # 5. Save final model
    print("\n[5/5] Saving final model...")
    output_dir = './models/phase7b_t2u_distilled'
    os.makedirs(output_dir, exist_ok=True)
    student_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    print("\n" + "="*80)
    print("Phase 7b complete! Next: Run final benchmark")
    print("="*80)


if __name__ == '__main__':
    main()
