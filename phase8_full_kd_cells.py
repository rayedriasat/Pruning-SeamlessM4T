# ============================================================================
# Phase 8: FULL MODEL Knowledge Distillation Implementation
# ============================================================================
# This file contains the replacement cells for Phase 8 in full-kd.ipynb
# 
# APPROACH: Train the ENTIRE Phase 7 model (all parameters) using KD from teacher
# TARGET: Improve ASR-BLEU and ASR-ChrF (audio quality)
# BASE MODEL: phase7_dora_merged_v1 (DoRA fine-tuned model with good text quality)
# ============================================================================

# ============================================================================
# Phase 8 — Cell 1: Load Phase 7 Student Model (ALL Parameters Trainable)
# ============================================================================

"""
## Phase 8 — Cell 1: Load Phase 7 Student Model for Full KD

**Change from previous approach:** 
- Load Phase 7 model with ALL parameters trainable (no freezing)
- This is Full Model KD, not T2U-only KD
"""

import gc as _gc
import torch

print('='*80)
print('PHASE 8: FULL MODEL KNOWLEDGE DISTILLATION')
print('='*80)
print('Approach: Train ENTIRE model (not just T2U) using KD from teacher')
print('Base: phase7_dora_merged_v1 (DoRA fine-tuned)')
print('Target: Improve ASR-BLEU and ASR-ChrF (audio quality)')
print('='*80)

# Clear any existing models from memory
if 'model_p8_student' in globals():
    del model_p8_student
if 'model_p7_ref' in globals():
    del model_p7_ref
_gc.collect()
torch.cuda.empty_cache()

# Load Phase 7 model as student
print('\n[P8] Loading phase7_dora_merged_v1 as student model...')
model_p8_student = load_model_from_drive('phase7_dora_merged_v1', processor)
model_p8_student.train()  # Set to training mode

# Count trainable parameters
total_params = sum(p.numel() for p in model_p8_student.parameters())
trainable_params = sum(p.numel() for p in model_p8_student.parameters() if p.requires_grad)

print(f'[P8] Student model loaded:')
print(f'     Total params     : {total_params/1e6:.1f}M')
print(f'     Trainable params : {trainable_params/1e6:.1f}M')
print(f'     Training mode    : {model_p8_student.training}')
print(f'[P8] All parameters are trainable (Full Model KD)')

# Verify all major components are trainable
components_to_check = [
    ('speech_encoder', 'Speech Encoder'),
    ('text_decoder', 'Text Decoder'),
    ('t2u_model', 'T2U Model'),
    ('lm_head', 'LM Head'),
    ('vocoder', 'Vocoder'),
]

print('\n[P8] Component trainability check:')
for attr_name, display_name in components_to_check:
    if hasattr(model_p8_student, attr_name):
        component = getattr(model_p8_student, attr_name)
        if component is not None:
            comp_params = sum(p.numel() for p in component.parameters())
            comp_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
            trainable_pct = (comp_trainable / comp_params * 100) if comp_params > 0 else 0
            print(f'     {display_name:<20} : {comp_trainable/1e6:>6.1f}M / {comp_params/1e6:>6.1f}M ({trainable_pct:.0f}% trainable)')

print('\n[P8] Student model ready for Full KD training.')


# ============================================================================
# Phase 8 — Cell 2: Load Teacher Model for KD
# ============================================================================

"""
## Phase 8 — Cell 2: Load Teacher Model for KD

Load the full teacher model (SeamlessM4Tv2Large) for knowledge distillation.
Teacher is always in eval() mode and never updated.
"""

# Clear teacher if already loaded
if 'model_teacher' in globals():
    print('[P8] Teacher already in memory, reusing...')
else:
    print('[P8] Loading teacher model (SeamlessM4Tv2Large)...')
    model_teacher = load_base_model(processor)
    print('[P8] Teacher model loaded.')

model_teacher.eval()
for p in model_teacher.parameters():
    p.requires_grad = False

teacher_params = sum(p.numel() for p in model_teacher.parameters())
print(f'[P8] Teacher: {teacher_params/1e6:.1f}M params, eval mode, frozen')


# ============================================================================
# Phase 8 — Cell 3: Full Model KD Loss Function
# ============================================================================

"""
## Phase 8 — Cell 3: Full Model KD Loss & Training Utilities

Implements knowledge distillation loss combining:
1. Text sequence distillation (KL divergence on text decoder logits)
2. Audio waveform distillation (MSE on vocoder outputs)
"""

import torch.nn.functional as F
import logging

# Suppress verbose warnings during training
_hf_logger = logging.getLogger('transformers.generation.utils')
_hf_logger.setLevel(logging.ERROR)


def compute_full_kd_loss(teacher, student, wav_batch, tgt_lang='ben', 
                         temperature=2.0, alpha=0.7, device='cuda'):
    """
    Compute Full Model KD loss combining text and audio distillation.
    
    Args:
        teacher: Teacher model (frozen, eval mode)
        student: Student model (trainable, train mode)
        wav_batch: List of input audio tensors
        tgt_lang: Target language code
        temperature: Temperature for KL divergence softening
        alpha: Weight for KD loss (1-alpha for hard loss if using labels)
        device: Device to run on
    
    Returns:
        loss: Combined KD loss
        metrics: Dict with loss components for logging
    """
    
    # Prepare inputs
    inputs = processor(audios=wav_batch, return_tensors='pt', sampling_rate=16000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get target language ID
    tgt_lang_id = model_teacher.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
    if tgt_lang_id is None:
        raise ValueError(f"Target language '{tgt_lang}' not found in model config")
    
    try:
        # ========================================================================
        # 1. TEXT SEQUENCE DISTILLATION
        # ========================================================================
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_outputs = teacher.generate(
                **inputs,
                tgt_lang=tgt_lang,
                generate_speech=False,  # Text only for now
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
            )
            teacher_text_ids = teacher_outputs.sequences
            
            # Get teacher logits by running forward pass with teacher's generated IDs
            teacher_forward = teacher(
                **inputs,
                tgt_lang=tgt_lang,
                labels=teacher_text_ids,
                return_dict=True,
            )
            teacher_text_logits = teacher_forward.logits  # [batch, seq_len, vocab]
        
        # Student forward pass (with grad)
        student_outputs = student(
            **inputs,
            tgt_lang=tgt_lang,
            labels=teacher_text_ids,  # Use teacher's text as target
            return_dict=True,
        )
        student_text_logits = student_outputs.logits  # [batch, seq_len, vocab]
        
        # Compute KL divergence loss on text logits
        # Align sequence lengths (teacher might generate different length)
        min_len = min(teacher_text_logits.size(1), student_text_logits.size(1))
        teacher_text_logits = teacher_text_logits[:, :min_len, :]
        student_text_logits = student_text_logits[:, :min_len, :]
        
        # Apply temperature and compute KL divergence
        teacher_probs = F.softmax(teacher_text_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_text_logits / temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)  # Scale by T^2 as per Hinton et al.
        
        # ========================================================================
        # 2. AUDIO WAVEFORM DISTILLATION
        # ========================================================================
        
        # Generate audio from both models
        with torch.no_grad():
            teacher_audio_outputs = teacher.generate(
                **inputs,
                tgt_lang=tgt_lang,
                generate_speech=True,
                return_dict_in_generate=True,
                max_new_tokens=256,
            )
            teacher_waveform = teacher_audio_outputs.waveform  # [batch, audio_len]
        
        student_audio_outputs = student.generate(
            **inputs,
            tgt_lang=tgt_lang,
            generate_speech=True,
            return_dict_in_generate=True,
            max_new_tokens=256,
        )
        student_waveform = student_audio_outputs.waveform  # [batch, audio_len]
        
        # Align waveform lengths
        min_audio_len = min(teacher_waveform.size(-1), student_waveform.size(-1))
        teacher_waveform = teacher_waveform[..., :min_audio_len]
        student_waveform = student_waveform[..., :min_audio_len]
        
        # MSE loss on waveforms
        audio_mse_loss = F.mse_loss(student_waveform, teacher_waveform)
        
        # ========================================================================
        # 3. COMBINE LOSSES
        # ========================================================================
        
        # Weighted combination: prioritize text quality, but include audio
        # alpha controls the balance (higher alpha = more weight on KD)
        total_loss = alpha * kl_loss + (1 - alpha) * audio_mse_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'kl_loss': kl_loss.item(),
            'audio_mse': audio_mse_loss.item(),
        }
        
        return total_loss, metrics
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'[P8] OOM in KD loss computation: {e}')
            torch.cuda.empty_cache()
            # Return a small dummy loss to allow training to continue
            dummy_loss = torch.tensor(0.01, device=device, requires_grad=True)
            return dummy_loss, {'total_loss': 0.01, 'kl_loss': 0.0, 'audio_mse': 0.0, 'oom': True}
        else:
            raise


print('[P8] Full KD loss function defined.')
print('     - Text sequence distillation: KL divergence on decoder logits')
print('     - Audio waveform distillation: MSE on vocoder outputs')
print('     - Combined loss: alpha * KL + (1-alpha) * MSE')


# ============================================================================
# Phase 8 — Cell 4: Optimizer Setup
# ============================================================================

"""
## Phase 8 — Cell 4: Optimizer Setup

Setup optimizer for Full Model KD training.
Lower learning rate (1e-5) since we're fine-tuning the entire model.
"""

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hyperparameters for Full Model KD
KD_MAX_STEPS = 1000
KD_BATCH_SIZE = 1
KD_GRAD_ACCUM = 8
KD_LR = 1e-5  # Lower LR for full model fine-tuning
KD_TEMPERATURE = 2.0
KD_ALPHA = 0.7  # 70% KD loss, 30% audio MSE

# All parameters are trainable
kd_optimizer = AdamW(
    [p for p in model_p8_student.parameters() if p.requires_grad],
    lr=KD_LR,
    weight_decay=0.01
)

kd_scheduler = CosineAnnealingLR(kd_optimizer, T_max=KD_MAX_STEPS)

# Resume from checkpoint if available
kd_ckpt = load_latest_checkpoint('phase8_full_kd')
kd_start_step = 0
kd_loss_log = []

if kd_ckpt:
    print('[P8] Resuming from checkpoint...')
    kd_start_step = kd_ckpt.get('step', 0)
    kd_loss_log = kd_ckpt.get('loss_log', [])
    
    if 'model_state' in kd_ckpt:
        model_p8_student.load_state_dict(kd_ckpt['model_state'])
        print(f'[P8] Loaded model state from step {kd_start_step}')
    
    if 'optimizer_state' in kd_ckpt:
        kd_optimizer.load_state_dict(kd_ckpt['optimizer_state'])
        print('[P8] Loaded optimizer state')
    
    if 'scheduler_state' in kd_ckpt:
        kd_scheduler.load_state_dict(kd_ckpt['scheduler_state'])
        print('[P8] Loaded scheduler state')
else:
    print('[P8] Starting Full KD from scratch.')

print(f'[P8] Optimizer: AdamW  LR={KD_LR}  EffectiveBatch={KD_BATCH_SIZE * KD_GRAD_ACCUM}')
print(f'[P8] Max steps: {KD_MAX_STEPS}  Temperature: {KD_TEMPERATURE}  Alpha: {KD_ALPHA}')
print(f'[P8] Training ALL model parameters ({trainable_params/1e6:.1f}M params)')


# ============================================================================
# Phase 8 — Cell 5: Full KD Training Loop
# ============================================================================

"""
## Phase 8 — Cell 5: Full Model KD Training Loop

Train the entire student model using knowledge distillation from teacher.
Combines text sequence and audio waveform distillation.
"""

import time
from tqdm.auto import tqdm

print('\n' + '='*80)
print('STARTING FULL MODEL KD TRAINING')
print('='*80)

# Training state
optim_steps = kd_start_step
micro_step = 0
epoch_start = time.time()

# Metrics tracking
kl_losses = []
audio_losses = []
total_losses = []

try:
    model_p8_student.train()
    kd_optimizer.zero_grad()
    
    pbar = tqdm(total=KD_MAX_STEPS - kd_start_step, desc='[P8] Full KD', 
                initial=kd_start_step, unit='step')
    
    while optim_steps < KD_MAX_STEPS:
        # Sample a batch from training data
        batch_wavs = []
        for _ in range(KD_BATCH_SIZE):
            idx = torch.randint(0, len(train_samples), (1,)).item()
            sample = train_samples[idx]
            wav = sample['audio']['array']
            batch_wavs.append(wav)
        
        # Compute KD loss
        try:
            loss, metrics = compute_full_kd_loss(
                teacher=model_teacher,
                student=model_p8_student,
                wav_batch=batch_wavs,
                tgt_lang=TARGET_LANG,
                temperature=KD_TEMPERATURE,
                alpha=KD_ALPHA,
                device=DEVICE
            )
            
            # Normalize by gradient accumulation steps
            loss = loss / KD_GRAD_ACCUM
            loss.backward()
            
            # Track metrics
            total_losses.append(metrics['total_loss'])
            kl_losses.append(metrics.get('kl_loss', 0.0))
            audio_losses.append(metrics.get('audio_mse', 0.0))
            
            micro_step += 1
            
            # Optimizer step after accumulation
            if micro_step % KD_GRAD_ACCUM == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model_p8_student.parameters(), max_norm=1.0)
                
                kd_optimizer.step()
                kd_scheduler.step()
                kd_optimizer.zero_grad()
                
                optim_steps += 1
                
                # Logging
                if optim_steps % 10 == 0:
                    avg_total = sum(total_losses[-10:]) / len(total_losses[-10:])
                    avg_kl = sum(kl_losses[-10:]) / len(kl_losses[-10:])
                    avg_audio = sum(audio_losses[-10:]) / len(audio_losses[-10:])
                    lr = kd_scheduler.get_last_lr()[0]
                    
                    pbar.set_postfix({
                        'loss': f'{avg_total:.4f}',
                        'kl': f'{avg_kl:.4f}',
                        'audio': f'{avg_audio:.4f}',
                        'lr': f'{lr:.2e}'
                    })
                
                pbar.update(1)
                
                # Save checkpoint every 250 steps
                if optim_steps % 250 == 0:
                    save_checkpoint(
                        dict(step=optim_steps,
                             loss_log=kd_loss_log + total_losses,
                             model_state=model_p8_student.state_dict(),
                             optimizer_state=kd_optimizer.state_dict(),
                             scheduler_state=kd_scheduler.state_dict()),
                        name='phase8_full_kd',
                        step=optim_steps
                    )
                
                # Clear cache periodically
                if optim_steps % 50 == 0:
                    torch.cuda.empty_cache()
        
        except RuntimeError as e:
            print(f'\n[ERR] Step {optim_steps}: {e}')
            torch.cuda.empty_cache()
            kd_optimizer.zero_grad()
            micro_step = 0
            continue
    
    pbar.close()
    
    # Final save
    kd_loss_log.extend(total_losses)
    save_checkpoint(
        dict(step=optim_steps,
             loss_log=kd_loss_log,
             model_state=model_p8_student.state_dict(),
             optimizer_state=kd_optimizer.state_dict(),
             scheduler_state=kd_scheduler.state_dict()),
        name='phase8_full_kd',
        step=optim_steps
    )
    
    elapsed = (time.time() - epoch_start) / 60
    print(f'\n[P8] Full KD complete. Final step: {optim_steps}  Time: {elapsed:.1f} min')
    print(f'[P8] Final losses - Total: {total_losses[-1]:.4f}, KL: {kl_losses[-1]:.4f}, Audio: {audio_losses[-1]:.4f}')

except KeyboardInterrupt:
    print('\n[P8] Training interrupted by user.')
    kd_loss_log.extend(total_losses)
    save_checkpoint(
        dict(step=optim_steps,
             loss_log=kd_loss_log,
             model_state=model_p8_student.state_dict(),
             optimizer_state=kd_optimizer.state_dict(),
             scheduler_state=kd_scheduler.state_dict()),
        name='phase8_full_kd',
        step=optim_steps
    )

finally:
    model_p8_student.eval()
    torch.cuda.empty_cache()
    print('[P8] Training loop exited. Model set to eval mode.')


# ============================================================================
# Phase 8 — Cell 6: Plot Full KD Training Curves
# ============================================================================

"""
## Phase 8 — Cell 6: Plot Full KD Training Curves

Visualize the training progress with separate plots for:
1. Total loss
2. KL divergence (text distillation)
3. Audio MSE (waveform distillation)
"""

import matplotlib.pyplot as plt
import numpy as np

if not total_losses:
    print('[P8] No training data to plot.')
else:
    # Smooth curves with moving average
    def smooth(data, window=10):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Phase 8 — Full Model Knowledge Distillation Training', 
                 fontsize=13, fontweight='bold')
    
    # Plot 1: Total Loss
    ax = axes[0]
    steps = np.arange(len(total_losses))
    ax.plot(steps, total_losses, alpha=0.3, color='#1976D2', linewidth=0.5)
    ax.plot(smooth(total_losses), color='#1976D2', linewidth=2, label='Total Loss')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total KD Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: KL Divergence (Text)
    ax = axes[1]
    ax.plot(steps, kl_losses, alpha=0.3, color='#388E3C', linewidth=0.5)
    ax.plot(smooth(kl_losses), color='#388E3C', linewidth=2, label='KL Divergence')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('KL Loss')
    ax.set_title('Text Sequence Distillation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Audio MSE
    ax = axes[2]
    ax.plot(steps, audio_losses, alpha=0.3, color='#D32F2F', linewidth=0.5)
    ax.plot(smooth(audio_losses), color='#D32F2F', linewidth=2, label='Audio MSE')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Audio Waveform Distillation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, 'phase8_full_kd_training_curves.png')
    plt.show()
    print('[P8] Training curves saved.')


# ============================================================================
# Phase 8 — Cell 7: Save phase8_full_kd Model to Drive
# ============================================================================

"""
## Phase 8 — Cell 7: Save phase8_full_kd Model to Drive

Save the trained student model after Full KD.
"""

model_p8_student.eval()
sync_model_config(model_p8_student)

save_model_to_drive(model_p8_student, processor, 'phase8_full_kd')
print_model_breakdown(model_p8_student, 'After Phase 8: Full Model KD')

print('\n[P8] Model saved to Drive as phase8_full_kd')
print('[P8] Ready for benchmarking.')


# ============================================================================
# NOTES FOR BENCHMARK CELLS
# ============================================================================

"""
The benchmark cells (Phase 8 Benchmark Cells 1-5) need minor updates:

1. Change all references from 'phase8_kd' to 'phase8_full_kd'
2. Update labels from 'P8 KD' to 'P8 Full KD'
3. Keep all 4 metrics: ASR-BLEU, ASR-ChrF, Text-BLEU, Text-ChrF
4. Compare: Teacher, Phase 6, Phase 7, Phase 8 Full KD

The benchmark cell structure remains the same, just update the model name.
"""
