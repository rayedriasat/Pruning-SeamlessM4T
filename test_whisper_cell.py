# ── Test Whisper ASR: Bengali → English Translation ──────────────────────────
# This cell tests the complete pipeline: BN audio → S2ST → EN audio → Whisper ASR

print('='*70)
print('  WHISPER ASR TEST: Bengali → English Translation')
print('='*70)

# Load a Bengali test sample
test_sample = eval_samples[0]  # First Bengali sample
print(f'\n📥 Input:')
print(f'  Language: Bengali → English')
print(f'  Duration: {len(test_sample["wav"])/16000:.1f}s')
print(f'  Reference (EN): {test_sample["ref"][:100]}...')

# Run S2ST translation (Bengali → English)
print(f'\n🔄 Running S2ST translation...')
try:
    _, wav_out = run_s2st(model_v1, test_sample['wav'], tgt_lang='eng')
    print(f'  ✓ Translation complete')
    print(f'  Output duration: {len(wav_out)/16000:.1f}s')
except Exception as e:
    print(f'  ✗ Translation failed: {e}')
    import traceback
    traceback.print_exc()

# Test Whisper ASR on English output
print(f'\n🎤 Testing Whisper ASR (English)...')
try:
    # Ensure Whisper is loaded
    _ensure_whisper_loaded()
    
    # Transcribe using Whisper
    hyp = asr_transcribe_whisper(wav_out, lang='en', sr=16000)
    
    print(f'  ✓ Whisper transcription complete')
    print(f'\n📝 Results:')
    print(f'  Reference: {test_sample["ref"][:150]}')
    print(f'  Whisper:   {hyp[:150]}')
    
    # Compute metrics
    from sacrebleu.metrics import BLEU, CHRF
    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF()
    
    bleu_score = bleu_metric.sentence_score(hyp, [test_sample['ref']]).score
    chrf_score = chrf_metric.sentence_score(hyp, [test_sample['ref']]).score
    
    print(f'\n📊 Metrics:')
    print(f'  ASR-BLEU: {bleu_score:.2f}')
    print(f'  ASR-ChrF: {chrf_score:.2f}')
    
    # Play audio (optional)
    print(f'\n🔊 Audio samples:')
    play(test_sample['wav'], 16000, 'Input (Bengali)')
    play(wav_out, 16000, 'Output (English, voice-cloned)')
    
    print(f'\n✅ Whisper ASR test PASSED')
    
except Exception as e:
    print(f'  ✗ Whisper ASR failed: {e}')
    import traceback
    traceback.print_exc()
    print(f'\n❌ Whisper ASR test FAILED')

print('='*70)
