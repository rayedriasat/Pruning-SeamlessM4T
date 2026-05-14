import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor
from transformers.modeling_outputs import BaseModelOutput

class CIFBoundaryDetector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class SeamlessChunkedWrapper(nn.Module):
    def __init__(self, base_model, adapter_path):
        super().__init__()
        self.model = base_model
        self.adapter = CIFBoundaryDetector()
        self.adapter.load_state_dict(torch.load(adapter_path, map_location='cpu', weights_only=True))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_features, attention_mask=None):
        encoder_outputs = self.model.speech_encoder(input_features, attention_mask=attention_mask, return_dict=True)
        hidden_states = encoder_outputs.last_hidden_state
        boundary_probs = self.adapter(hidden_states.float())
        return hidden_states, boundary_probs

def _remap_ids_for_decode(mdl, ids):
    if hasattr(mdl, '_vocab_remap_to_old'):
        remap = mdl._vocab_remap_to_old
        ids = ids.clone()
        mask = (ids >= 0) & (ids < len(remap))
        ids[mask] = remap[ids[mask]]
    return ids

def run_s2st_chunked(wrapper, processor, wav, tgt_lang='ben', silence_threshold=0.2):
    # 1. Run the full audio through the wrapper to get adapter predictions
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    device = next(wrapper.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        hidden_states, boundary_probs = wrapper(**inputs)
        
        # Find where probability drops below threshold (silence detected)
        is_silence = boundary_probs.squeeze(-1) < silence_threshold
        
        fire_indices = []
        in_silence = False
        for i in range(is_silence.size(1)):
            if is_silence[0, i] and not in_silence:
                fire_indices.append(i)
                in_silence = True
            elif not is_silence[0, i]:
                in_silence = False
                
        # Ensure we capture the final segment
        if len(fire_indices) == 0 or fire_indices[-1] != hidden_states.size(1) - 1:
            fire_indices.append(hidden_states.size(1) - 1)

        total_frames = hidden_states.size(1)
        audio_samples = len(wav)
        
        full_text = []
        audio_chunks = []
        start_sample = 0
        
        print("\n--- Streaming Inference Logs ---")
        
        for chunk_num, end_idx in enumerate(fire_indices):
            # Convert the encoder frame index back into raw audio sample timing
            end_sample = int(((end_idx + 1) / total_frames) * audio_samples)
            timestamp_sec = end_sample / 16000
            
            print(f"📦 Chunk {chunk_num + 1} | Slicing audio up to timestamp: {timestamp_sec:.2f}s")
            
            # Slice the raw waveform!
            wav_chunk = wav[start_sample:end_sample]
            start_sample = end_sample
            
            # Skip tiny micro-chunks of pure noise (less than 0.1 seconds)
            if len(wav_chunk) < 1600:
                continue
                
            # 2. Run standard generation on this sliced audio chunk
            chunk_inputs = processor(audio=wav_chunk, sampling_rate=16000, return_tensors='pt')
            chunk_inputs = {k: v.to(device) for k, v in chunk_inputs.items()}
            
            try:
                # Standard generate call (No more NoneType errors!)
                out = wrapper.model.generate(
                    **chunk_inputs,
                    tgt_lang=tgt_lang,
                    return_intermediate_token_ids=True
                )
                
                text_ids = _remap_ids_for_decode(wrapper.model, out.sequences.cpu())
                text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
                wav_out = out.waveform.cpu().float().numpy().squeeze() if out.waveform is not None else np.zeros(16000, dtype=np.float32)
                
                full_text.append(text)
                audio_chunks.append(wav_out)
                
                print(f"   -> Translated Text: {text}")
                
            except Exception as e:
                print(f"  [!] Decoder failed on Chunk {chunk_num + 1}: {e}")

        final_text = " ".join(full_text)
        final_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(16000, dtype=np.float32)
        
        return final_text, final_audio

if __name__ == "__main__":
    INPUT_AUDIO = '/home/nihal/CSE465/Pruning-SeamlessM4T/seamless_local/test.wav'
    OUTPUT_AUDIO = '/home/nihal/CSE465/Pruning-SeamlessM4T/seamless_local/output_chunked.wav'
    MODEL_DIR = '/home/nihal/CSE465/seamV5/models/phase7_dora_merged_v1'
    ADAPTER_PATH = '/home/nihal/CSE465/Duplex_making/boundary_adapter.pt'
    TGT_LANG = 'ben'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Processor...")
    processor = SeamlessM4TProcessor.from_pretrained(MODEL_DIR)
    
    print("Loading Base Model...")
    base_model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map=device)
    base_model.eval()

    print("Loading CIF Boundary Adapter...")
    wrapper = SeamlessChunkedWrapper(base_model, ADAPTER_PATH).to(device)
    wrapper.eval()

    print(f"Processing audio: {INPUT_AUDIO}")
    audio, sr = librosa.load(INPUT_AUDIO, sr=16000)
    
    pred_text, out_wav = run_s2st_chunked(wrapper, processor, audio, tgt_lang=TGT_LANG)

    print(f"\nFinal Translated Text: {pred_text}")
    sf.write(OUTPUT_AUDIO, out_wav, 16000)
    print(f"Output audio saved to: {OUTPUT_AUDIO}")