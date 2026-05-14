import numpy as np
from transformers.modeling_outputs import BaseModelOutput

class SeamlessChunkedWrapper(nn.Module):
    def __init__(self, base_model, adapter_path):
        super().__init__()
        self.model = base_model
        
        # Load your trained adapter
        self.adapter = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        self.adapter.load_state_dict(torch.load(adapter_path, map_location='cpu', weights_only=True))
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_features, attention_mask=None):
        encoder_outputs = self.model.speech_encoder(input_features, attention_mask=attention_mask, return_dict=True)
        hidden_states = encoder_outputs.last_hidden_state
        boundary_probs = self.adapter(hidden_states)
        return hidden_states, boundary_probs

def run_s2st_chunked(wrapper, processor, wav, tgt_lang='ben', silence_threshold=0.2):
    """
    Speech-to-speech with CIF boundary chunking and timestamp logging.
    silence_threshold: The probability below which the model considers it "silence" (0).
    """
    inputs = processor(audio=wav, sampling_rate=16000, return_tensors='pt')
    device = next(wrapper.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    audio_duration_sec = len(wav) / 16000
    
    with torch.no_grad():
        hidden_states, boundary_probs = wrapper(**inputs)
        
        # Find points where speech ends and silence begins (probability drops below threshold)
        is_silence = boundary_probs.squeeze(-1) < silence_threshold
        
        # We want to trigger a chunk when we hit a block of silence
        fire_indices = []
        in_silence = False
        for i in range(is_silence.size(1)):
            if is_silence[0, i] and not in_silence:
                fire_indices.append(i)
                in_silence = True
            elif not is_silence[0, i]:
                in_silence = False
                
        # Ensure the last chunk is processed
        if len(fire_indices) == 0 or fire_indices[-1] != hidden_states.size(1) - 1:
            fire_indices.append(hidden_states.size(1) - 1)

        total_frames = hidden_states.size(1)
        full_text = []
        audio_chunks = []
        
        start_idx = 0
        print("\n--- Streaming Inference Logs ---")
        
        for chunk_num, end_idx in enumerate(fire_indices):
            # Calculate timestamp ratio
            timestamp_sec = (end_idx / total_frames) * audio_duration_sec
            print(f"📦 Chunk {chunk_num + 1} sent to decoder at timestamp: {timestamp_sec:.2f}s")
            
            chunk_hidden = hidden_states[:, start_idx:end_idx+1, :]
            chunk_encoder_outputs = BaseModelOutput(last_hidden_state=chunk_hidden)
            
            try:
                out = wrapper.model.generate(
                    encoder_outputs=chunk_encoder_outputs,
                    tgt_lang=tgt_lang,
                    return_intermediate_token_ids=True
                )
                text_ids = _remap_ids_for_decode(wrapper.model, out.sequences.cpu())
                text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
                wav_out = out.waveform.cpu().float().numpy().squeeze() if out.waveform is not None else np.zeros(16000, dtype=np.float32)
                
                full_text.append(text)
                audio_chunks.append(wav_out)
            except Exception as e:
                print(f"  [!] Decoder failed on Chunk {chunk_num + 1}: {e}")
            
            start_idx = end_idx + 1

        final_text = " ".join(full_text)
        final_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(16000, dtype=np.float32)
        
        return final_text, final_audio

# ==== Example Execution Block for infer_clean.py ====
# wrapper = SeamlessChunkedWrapper(model, adapter_path="boundary_adapter.pt").to(_model_input_device(model))
# wrapper.eval()
# pred_text, out_wav = run_s2st_chunked(wrapper, processor, audio, tgt_lang=TGT_LANG)