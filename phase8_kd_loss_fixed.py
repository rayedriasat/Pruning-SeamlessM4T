"""
Fixed compute_full_kd_loss function for Phase 8 KD training.

Key fixes:
1. Convert all token IDs to .long() to avoid embedding type errors
2. Use teacher-forced forward passes instead of model() which doesn't use T2U
3. Access text_decoder and T2U model directly for proper gradient flow
"""

import torch
import torch.nn.functional as F


def compute_full_kd_loss(teacher, student, wav_batch, ref_texts, processor,
                         tgt_lang='ben', temperature=2.0, alpha=0.7, beta=0.2, device='cuda'):
    """
    Compute Full Model KD loss with proper gradient flow through student.
    
    CRITICAL FIXES:
    - All token IDs converted to .long() to avoid embedding type errors
    - Direct access to text_decoder and t2u_model for teacher-forced training
    - Proper handling of SeamlessM4Tv2 architecture
    
    Args:
        teacher: Teacher model (frozen, eval mode)
        student: Student model (trainable, train mode)
        wav_batch: List of input audio tensors
        ref_texts: List of reference Bengali texts
        processor: SeamlessM4TProcessor instance
        tgt_lang: Target language code
        temperature: Temperature for KL divergence
        alpha: Weight for text CE loss
        beta: Weight for T2U loss
        device: Device to run on
    
    Returns:
        loss: Combined KD loss
        metrics: Dict with loss components
    """
    
    # Prepare inputs
    inputs = processor(audio=wav_batch, return_tensors='pt', sampling_rate=16000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Tokenize reference texts
    text_inputs = processor(text=ref_texts, return_tensors='pt', src_lang='eng', tgt_lang=tgt_lang)
    labels = text_inputs['input_ids'].to(device).long()  # CRITICAL: Convert to long
    
    try:
        # ====================================================================
        # 1. GET TEACHER PREDICTIONS (no grad)
        # ====================================================================
        with torch.no_grad():
            teacher.eval()
            
            # Generate to get teacher's predicted sequence
            teacher_gen = teacher.generate(
                **inputs,
                tgt_lang=tgt_lang,
                return_dict_in_generate=True,
                return_intermediate_token_ids=True,
                max_new_tokens=256,
            )
            
            # Extract teacher text IDs
            teacher_text_ids = teacher_gen.sequences.long()  # CRITICAL: Convert to long
            
            # Extract teacher unit IDs if available
            teacher_unit_ids = None
            if hasattr(teacher_gen, 'unit_sequences') and teacher_gen.unit_sequences is not None:
                teacher_unit_ids = teacher_gen.unit_sequences.long()  # CRITICAL: Convert to long
        
        # ====================================================================
        # 2. STUDENT TEXT DECODER (teacher-forced with teacher's text)
        # ====================================================================
        student.train()
        
        # Get speech encoder outputs
        speech_encoder_out = student.model.speech_encoder(**inputs)
        encoder_hidden = speech_encoder_out.last_hidden_state
        
        # Teacher-forced text decoder forward
        # Input: teacher_text_ids shifted right, Target: teacher_text_ids shifted left
        decoder_input_ids = teacher_text_ids[:, :-1]  # Remove last token
        decoder_targets = teacher_text_ids[:, 1:]     # Remove first token (BOS)
        
        text_decoder_out = student.model.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
            return_dict=True,
        )
        student_text_logits = text_decoder_out.logits
        
        # Text CE loss
        text_ce_loss = F.cross_entropy(
            student_text_logits.reshape(-1, student_text_logits.size(-1)),
            decoder_targets.reshape(-1),
            ignore_index=processor.tokenizer.pad_token_id,
        )
        
        # ====================================================================
        # 3. STUDENT T2U MODEL (teacher-forced with teacher's units)
        # ====================================================================
        t2u_loss = torch.tensor(0.0, device=device)
        
        if teacher_unit_ids is not None:
            # T2U encoder: encode the text sequence
            t2u_encoder_out = student.t2u_model.model.encoder(
                input_ids=teacher_text_ids,
                return_dict=True,
            )
            
            # T2U decoder: predict units (teacher-forced)
            t2u_decoder_input = teacher_unit_ids[:, :-1]
            t2u_decoder_target = teacher_unit_ids[:, 1:]
            
            t2u_decoder_out = student.t2u_model.model.decoder(
                input_ids=t2u_decoder_input,
                encoder_hidden_states=t2u_encoder_out.last_hidden_state,
                return_dict=True,
            )
            student_unit_logits = t2u_decoder_out.logits
            
            # T2U CE loss
            t2u_loss = F.cross_entropy(
                student_unit_logits.reshape(-1, student_unit_logits.size(-1)),
                t2u_decoder_target.reshape(-1),
                ignore_index=-100,
            )
        
        # ====================================================================
        # 4. HARD LABEL LOSS (optional grounding with ground truth)
        # ====================================================================
        hard_loss = torch.tensor(0.0, device=device)
        gamma = max(0.0, 1.0 - alpha - beta)
        
        if gamma > 0.0:
            # Teacher-forced with ground truth labels
            gt_input = labels[:, :-1]
            gt_target = labels[:, 1:]
            
            hard_decoder_out = student.model.text_decoder(
                input_ids=gt_input,
                encoder_hidden_states=encoder_hidden,
                return_dict=True,
            )
            
            hard_loss = F.cross_entropy(
                hard_decoder_out.logits.reshape(-1, hard_decoder_out.logits.size(-1)),
                gt_target.reshape(-1),
                ignore_index=processor.tokenizer.pad_token_id,
            )
        
        # ====================================================================
        # 5. COMBINE LOSSES
        # ====================================================================
        total_loss = alpha * text_ce_loss + beta * t2u_loss + gamma * hard_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'text_ce': text_ce_loss.item(),
            't2u_loss': t2u_loss.item(),
            'hard_loss': hard_loss.item(),
        }
        
        return total_loss, metrics
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'[P8] OOM: {e}')
            torch.cuda.empty_cache()
            dummy_loss = torch.tensor(0.01, device=device, requires_grad=True)
            return dummy_loss, {
                'total_loss': 0.01,
                'text_ce': 0.0,
                't2u_loss': 0.0,
                'hard_loss': 0.0,
                'oom': True
            }
        else:
            raise


print('[P8] Fixed KD loss function loaded.')
print('Key fixes:')
print('  1. All token IDs converted to .long()')
print('  2. Direct access to text_decoder and t2u_model')
print('  3. Teacher-forced training for proper gradients')
