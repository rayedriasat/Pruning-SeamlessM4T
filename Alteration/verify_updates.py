#!/usr/bin/env python3
"""
Verify that all ASR and multilingual updates are present in the notebook
"""

import json
import re

def read_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_pattern(nb, pattern, description):
    """Check if pattern exists anywhere in notebook"""
    for cell in nb['cells']:
        source_text = ''.join(cell.get('source', []))
        if re.search(pattern, source_text, re.IGNORECASE):
            return True
    return False

def main():
    nb_path = 'Alteration/seamless-final.ipynb'
    nb = read_notebook(nb_path)
    
    checks = [
        # Dataset loading
        ("EVAL_LANG_PAIRS.*=.*\\[.*eng.*ben.*cmn.*arb.*hin", 
         "✓ EVAL_LANG_PAIRS defined with all 5 languages"),
        
        ("eval_samples.*=.*\\[\\].*Unified multilingual eval set",
         "✓ eval_samples initialized as multilingual"),
        
        ("ft_samples.*=.*\\[\\].*Unified multilingual training set",
         "✓ ft_samples initialized as multilingual"),
        
        # ASR functions
        ("def run_benchmark_asr\\(.*ASR-based benchmark",
         "✓ run_benchmark_asr function defined"),
        
        ("ASR-BLEU.*ASR-ChrF",
         "✓ ASR metrics used in output"),
        
        ("asr_transcribe\\(wav_out.*tgt_lang",
         "✓ ASR transcription called on output audio"),
        
        # Multilingual support
        ("for src_m4t, tgt_m4t in EVAL_LANG_PAIRS",
         "✓ Loops over all language pairs"),
        
        ("src_lang.*tgt_lang.*in sample dict",
         "✓ Samples include src_lang and tgt_lang"),
        
        # Phase updates
        ("run_benchmark_asr\\(.*model_v1.*eval_samples",
         "✓ Phase 0 uses ASR benchmark"),
        
        ("quick_eval_chrf\\(mdl, samples.*max_eval",
         "✓ quick_eval_chrf updated for multilingual"),
        
        # KD extraction
        ("all_train_samples.*=.*\\{\\}.*for src_m4t, tgt_m4t in",
         "✓ KD extraction uses multilingual samples"),
        
        ("PAIRS.*=.*EVAL_LANG_PAIRS",
         "✓ KD extraction uses all language pairs"),
        
        # Phase 7
        ("samples_by_pair.*=.*defaultdict",
         "✓ Phase 7 groups samples by pair"),
        
        ("ASR-ChrF.*ASR-BLEU.*for all language pairs",
         "✓ Phase 7 reports ASR metrics"),
        
        # Visualization
        ("Translation Quality by Language Pair.*ASR",
         "✓ Visualization uses ASR metrics"),
    ]
    
    print("=" * 70)
    print("VERIFICATION: ASR Metrics + Multilingual Support")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for pattern, description in checks:
        if check_pattern(nb, pattern, description):
            print(f"✅ {description}")
            passed += 1
        else:
            print(f"❌ {description}")
            failed += 1
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All checks passed! Notebook is ready for execution.")
        print("\nNext steps:")
        print("  1. Run the notebook in Kaggle/Colab")
        print("  2. Verify eval_samples loads 200 samples (25 per pair × 8 pairs)")
        print("  3. Verify ft_samples loads 1600 samples (200 per pair × 8 pairs)")
        print("  4. Check Phase 0 benchmark reports ASR-ChrF/BLEU for all pairs")
        print("  5. Monitor Phase 2 encoder pruning with multilingual ASR-ChrF")
        print("  6. Verify Phase 7 shows results for all 8 language pairs")
    else:
        print(f"\n⚠️  {failed} check(s) failed. Review the notebook.")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
