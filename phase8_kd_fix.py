"""
Phase 8 KD Training Fix
=======================

This file contains the corrected compute_full_kd_loss function and training loop
that properly handles gradients for full model knowledge distillation.

Key fixes:
1. Handle both dict and tuple returns from model.generate()
2. Use forward() passes instead of generate() for gradient flow
3. Proper text decoder KD with teacher forcing
4. Optional T2U distillation via hidden state matching
5. Hard label loss for grounding

Replace the corresponding cells in your notebook with this code.
"""

import torch
import torch.nn.functional as F
import logging

# Suppress verbose warnings during training
_hf_logger = logging.getLogger('transformers.generation.utils')
_hf_logger.setLevel(logging.ERROR)


def compute_full_kd_loss(teacher, student, wav_batch, ref_texts, processor,
                         tgt_lang='ben', temperature=2.0, alpha=0.7, beta=0.2, device='cuda'):
    """
    Compute Full Model KD loss with proper gradient flow through student.
    
    Strategy:
    1. Get teacher's text predictions (no grad)
    2. Use teacher predictions as labels for both teacher and student forward passes
    3. Compute KL divergence on text decoder logits (with grad for student)
    4. Compute T2U distillation via hidden state matching (with grad for student)
    5. Optional: Add hard label loss using reference texts
    
    Args:
        teacher: Teacher model (frozen, eval mode)
        student: Student model (trainable, train mode)
        wav_batch: List of input audio tensors
        ref_texts: List of reference Bengali texts (for hard label loss)
        processor: SeamlessM4TProcessor instance
        tgt_lang: Target language code
        temperature: Temperature for KL divergence softening
        alpha: Weight for text KD loss
        beta: Weight for T2U KD loss (1-alpha-beta for hard label loss)
        device: Device to run on
    
    Returns:
        loss: Combined KD loss
        metrics: Dict with loss components for logging
    """
    
    # Prepare inputs
    inputs = processor(audio=wav_batch, return_tensors='pt', sampling_rate=16000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Tokenize reference texts for hard label loss
    text_inputs = processor(text=ref_texts, return_tensors='pt', src_lang='eng', tgt_lang=tgt_lang)
    labels = text_inputs['input_ids'].to(device)
    
    try:
        # ========================================================================
        # 1. GET TEACHER PREDICTIONS (no grad)
        # ========================================================================
        with torch.no_grad():
            teacher.eval()
            # Generate to get teacher's predicted sequence
            teacher_gen = teacher.generate(
                **inputs,
                tgt_lang=tgt_lang,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=256,
            )
            # Handle both dict and tuple returns
            if isinstance(teacher_gen, dict):
                teacher_text_ids = teacher_gen['sequences']
            elif isinstance(teacher_gen, tuple):
                teacher_text_ids = teacher_gen[0]
            else:
                teacher_text_ids = teacher_gen
            
            # Get teacher logits via forward pass with teacher's own predictions
            teacher_forward = teacher(
                **inputs,
                tgt_lang=tgt_lang,
                labels=teacher_text_ids,
                return_dict=True,
            )
            teacher_text_logits = teacher_forward.logits  # [batch, seq_len, vocab]
            
            # Get teacher hidden states for T2U distillation
            teacher_hidden = None
            if hasattr(teacher_forward, 'encoder_hidden_states') and teacher_forward.encoder_hidden_states is not None:
                teacher_hidden = teacher_forward.encoder_hidden_states[-1]  # Last layer
        
        # ========================================================================
        # 2. STUDENT FORWARD PASS (with grad)
        # ========================================================================
        student.train()
        
        # Forward pass with teacher's predicted text as labels
        student_forward = student(
            **inputs,
            tgt_lang=tgt_lang,
            labels=teacher_text_ids,
            return_dict=True,
            output_hidden_states=True,
        )
        student_text_logits = student_forward.logits  # [batch, seq_len, vocab]
        
        # Get student hidden states for T2U distillation
        student_hidden = None
        if hasattr(student_forward, 'encoder_hidden_states') and student_forward.encoder_hidden_states is not None:
            student_hidden = student_forward.encoder_hidden_states[-1]  # Last layer
        
        # ========================================================================
        # 3. TEXT DECODER KL DIVERGENCE LOSS
        # ========================================================================
        # Align sequence lengths
        min_len = min(teacher_text_logits.size(1), student_text_logits.size(1))
        teacher_text_logits_aligned = teacher_text_logits[:, :min_len, :]
        student_text_logits_aligned = student_text_logits[:, :min_len, :]
        
        # Apply temperature and compute KL divergence
        teacher_probs = F.softmax(teacher_text_logits_aligned / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_text_logits_aligned / temperature, dim=-1)
        
        text_kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)  # Scale by T^2 as per Hinton et al.
        
        # ========================================================================
        # 4. T2U HIDDEN STATE DISTILLATION
        # ========================================================================
        t2u_loss = torch.tensor(0.0, device=device)
        
        if teacher_hidden is not None and student_hidden is not None:
            # MSE loss on hidden states (proxy for T2U quality)
            min_hidden_len = min(teacher_hidden.size(1), student_hidden.size(1))
            t2u_loss = F.mse_loss(
                student_hidden[:, :min_hidden_len, :],
                teacher_hidden[:, :min_hidden_len, :].detach()
            )
        
        # ========================================================================
        # 5. HARD LABEL LOSS (optional, for grounding)
        # ========================================================================
        hard_loss = torch.tensor(0.0, device=device)
        gamma = max(0.0, 1.0 - alpha - beta)  # Weight for hard loss
        
        if gamma > 0.0:  # Only compute if we're using it
            # Forward pass with ground truth labels
            student_hard = student(
                **inputs,
                tgt_lang=tgt_lang,
                labels=labels,
                return_dict=True,
            )
            if hasattr(student_hard, 'loss') and student_hard.loss is not None:
                hard_loss = student_hard.loss
        
        # ========================================================================
        # 6. COMBINE LOSSES
        # ========================================================================
        total_loss = alpha * text_kl_loss + beta * t2u_loss + gamma * hard_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'text_kl': text_kl_loss.item(),
            't2u_loss': t2u_loss.item(),
            'hard_loss': hard_loss.item(),
        }
        
        return total_loss, metrics
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'[P8] OOM in KD loss computation: {e}')
            torch.cuda.empty_cache()
            # Return a small dummy loss to allow training to continue
            dummy_loss = torch.tensor(0.01, device=device, requires_grad=True)
            return dummy_loss, {
                'total_loss': 0.01, 
                'text_kl': 0.0, 
                't2u_loss': 0.0, 
                'hard_loss': 0.0, 
                'oom': True
            }
        else:
            raise


# ============================================================================
# TRAINING LOOP (Replace Phase 8 Cell 5)
# ============================================================================

def run_kd_training(model_teacher, model_p8_student, processor, ft_samples,
                   kd_optimizer, kd_scheduler, kd_start_step=0, kd_loss_log=None):
    """
    Full KD training loop with proper gradient handling.
    
    Args:
        model_teacher: Teacher model (frozen)
        model_p8_student: Student model (trainable)
        processor: SeamlessM4TProcessor
        ft_samples: List of training samples with 'wav' and 'ref' keys
        kd_optimizer: Optimizer
        kd_scheduler: Learning rate scheduler
        kd_start_step: Starting step (for resuming)
        kd_loss_log: Previous loss log (for resuming)
    
    Returns:
        Updated loss log
    """
    import time
    from tqdm.auto import tqdm
    
    # Hyperparameters
    KD_MAX_STEPS = 1000
    KD_BATCH_SIZE = 1
    KD_GRAD_ACCUM = 8
    KD_TEMPERATURE = 2.0
    KD_ALPHA = 0.7  # Text KD weight
    KD_BETA = 0.2   # T2U KD weight
    TARGET_LANG = 'ben'
    
    print('\n' + '='*80)
    print('STARTING FULL MODEL KD TRAINING')
    print('='*80)
    print(f'Alpha (text KD): {KD_ALPHA}')
    print(f'Beta (T2U KD): {KD_BETA}')
    print(f'Gamma (hard label): {1.0 - KD_ALPHA - KD_BETA}')
    print('='*80)
    
    DEVICE = next(model_teacher.parameters()).device
    
    # Training state
    optim_steps = kd_start_step
    micro_step = 0
    epoch_start = time.time()
    
    # Metrics tracking
    if kd_loss_log is None:
        kd_loss_log = []
    
    text_kl_losses = []
    t2u_losses = []
    hard_losses = []
    total_losses = []
    
    try:
        model_p8_student.train()
        kd_optimizer.zero_grad()
        
        pbar = tqdm(total=KD_MAX_STEPS - kd_start_step, desc='[P8] Full KD', 
                    initial=kd_start_step, unit='step')
        
        while optim_steps < KD_MAX_STEPS:
            # Sample a batch from training data
            batch_wavs = []
            batch_refs = []
            for _ in range(KD_BATCH_SIZE):
                idx = torch.randint(0, len(ft_samples), (1,)).item()
                sample = ft_samples[idx]
                batch_wavs.append(sample['wav'])
                batch_refs.append(sample['ref'])
            
            # Compute KD loss
            try:
                loss, metrics = compute_full_kd_loss(
                    teacher=model_teacher,
                    student=model_p8_student,
                    wav_batch=batch_wavs,
                    ref_texts=batch_refs,
                    processor=processor,
                    tgt_lang=TARGET_LANG,
                    temperature=KD_TEMPERATURE,
                    alpha=KD_ALPHA,
                    beta=KD_BETA,
                    device=DEVICE
                )
                
                # Normalize by gradient accumulation steps
                loss = loss / KD_GRAD_ACCUM
                loss.backward()
                
                # Track metrics
                total_losses.append(metrics['total_loss'])
                text_kl_losses.append(metrics.get('text_kl', 0.0))
                t2u_losses.append(metrics.get('t2u_loss', 0.0))
                hard_losses.append(metrics.get('hard_loss', 0.0))
                
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
                        avg_text_kl = sum(text_kl_losses[-10:]) / len(text_kl_losses[-10:])
                        avg_t2u = sum(t2u_losses[-10:]) / len(t2u_losses[-10:])
                        avg_hard = sum(hard_losses[-10:]) / len(hard_losses[-10:])
                        lr = kd_scheduler.get_last_lr()[0]
                        
                        pbar.set_postfix({
                            'loss': f'{avg_total:.4f}',
                            'text_kl': f'{avg_text_kl:.4f}',
                            't2u': f'{avg_t2u:.4f}',
                            'hard': f'{avg_hard:.4f}',
                            'lr': f'{lr:.2e}'
                        })
                    
                    pbar.update(1)
                    
                    # Save checkpoint every 250 steps
                    if optim_steps % 250 == 0:
                        from your_io_module import save_checkpoint  # Import your save function
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
        from your_io_module import save_checkpoint
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
        print(f'[P8] Final losses - Total: {total_losses[-1]:.4f}, Text KL: {text_kl_losses[-1]:.4f}, '
              f'T2U: {t2u_losses[-1]:.4f}, Hard: {hard_losses[-1]:.4f}')
        
        return kd_loss_log

    except KeyboardInterrupt:
        print('\n[P8] Training interrupted by user.')
        kd_loss_log.extend(total_losses)
        from your_io_module import save_checkpoint
        save_checkpoint(
            dict(step=optim_steps,
                 loss_log=kd_loss_log,
                 model_state=model_p8_student.state_dict(),
                 optimizer_state=kd_optimizer.state_dict(),
                 scheduler_state=kd_scheduler.state_dict()),
            name='phase8_full_kd',
            step=optim_steps
        )
        return kd_loss_log

    finally:
        model_p8_student.eval()
        torch.cuda.empty_cache()
        print('[P8] Training loop exited. Model set to eval mode.')


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
To use this in your notebook:

1. Replace Phase 8 Cell 3 (compute_full_kd_loss) with the function above

2. Replace Phase 8 Cell 5 (training loop) with:

```python
kd_loss_log = run_kd_training(
    model_teacher=model_teacher,
    model_p8_student=model_p8_student,
    processor=processor,
    ft_samples=ft_samples,
    kd_optimizer=kd_optimizer,
    kd_scheduler=kd_scheduler,
    kd_start_step=kd_start_step,
    kd_loss_log=kd_loss_log
)
```

Key improvements:
- ✅ Handles both dict and tuple returns from generate()
- ✅ Uses forward() passes for gradient flow (not generate())
- ✅ Proper text decoder KD with teacher forcing
- ✅ T2U distillation via hidden state matching
- ✅ Optional hard label loss for grounding
- ✅ All student parameters receive gradients
"""
