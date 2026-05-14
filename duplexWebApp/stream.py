import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import threading
import queue
import time
from collections import deque
from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor

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

class LiveStreamingEngine:
    def __init__(
        self,
        wrapper,
        processor,
        tgt_lang='ben',
        sample_rate=16000,
        silence_threshold=0.2,
        buffer_duration=30.0,
        min_chunk_duration=0.5,
        silence_patience_sec=0.8,
        output_dir='./streaming_output'
    ):
        self.wrapper = wrapper
        self.processor = processor
        self.tgt_lang = tgt_lang
        self.sr = sample_rate
        self.silence_threshold = silence_threshold
        self.min_chunk_samples = int(min_chunk_duration * sample_rate)
        self.buffer_samples = int(buffer_duration * sample_rate)
        self.silence_patience_sec = silence_patience_sec
        self.output_dir = output_dir
        
        self.audio_buffer = deque(maxlen=self.buffer_samples)
        self.buffer_lock = threading.Lock()
        
        self.inference_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        self.running = False
        self.chunk_counter = 0
        
        os.makedirs(output_dir, exist_ok=True)
        self.device = next(wrapper.parameters()).device
        
    def _audio_capture_loop(self, stop_event):
        audio_chunks = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            audio_chunks.append(indata.copy())
        
        stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype='float32',
            callback=callback,
            blocksize=1024
        )
        
        with stream:
            print(f"Started streaming at {self.sr} Hz")
            while not stop_event.is_set():
                if audio_chunks:
                    chunk = np.concatenate(audio_chunks)
                    audio_chunks = []
                    with self.buffer_lock:
                        for sample in chunk:
                            self.audio_buffer.append(sample[0])
                time.sleep(0.01)
        
    def _inference_loop(self, stop_event):
        while not stop_event.is_set():
            try:
                chunk_data = self.inference_queue.get(timeout=0.1)
                wav_chunk, start_sample, end_sample, chunk_id = chunk_data
                
                chunk_inputs = self.processor(
                    audio=wav_chunk,
                    sampling_rate=self.sr,
                    return_tensors='pt'
                )
                chunk_inputs = {k: v.to(self.device) for k, v in chunk_inputs.items()}
                
                try:
                    out = self.wrapper.model.generate(
                        **chunk_inputs,
                        tgt_lang=self.tgt_lang,
                        return_intermediate_token_ids=True
                    )
                    
                    text_ids = _remap_ids_for_decode(self.wrapper.model, out.sequences.cpu())
                    text = self.processor.batch_decode(text_ids, skip_special_tokens=True)[0]
                    wav_out = out.waveform.cpu().float().numpy().squeeze() if out.waveform is not None else np.zeros(16000, dtype=np.float32)
                    
                    output_path = os.path.join(self.output_dir, f'chunk_{chunk_id:04d}_audio.wav')
                    text_path = os.path.join(self.output_dir, f'chunk_{chunk_id:04d}_text.txt')
                    
                    sf.write(output_path, wav_out, self.sr)
                    with open(text_path, 'w') as f:
                        f.write(text)
                        
                    self.output_queue.put((chunk_id, text, output_path))
                    
                except Exception as e:
                    print(f"Error on chunk {chunk_id}: {e}")
                
            except queue.Empty:
                continue
    
    def _boundary_detection_loop(self, stop_event):
        while not stop_event.is_set():
            with self.buffer_lock:
                buffer_len = len(self.audio_buffer)
            
            if buffer_len < self.min_chunk_samples:
                time.sleep(0.05)
                continue
            
            with self.buffer_lock:
                buffer_audio = np.array(list(self.audio_buffer))
            
            inputs = self.processor(
                audio=buffer_audio,
                sampling_rate=self.sr,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _, boundary_probs = self.wrapper(**inputs)
            
            is_silence = boundary_probs.squeeze(-1) < self.silence_threshold
            total_frames = is_silence.size(1)
            
            frames_per_sec = total_frames / (len(buffer_audio) / self.sr)
            patience_frames = int(self.silence_patience_sec * frames_per_sec)
            
            silence_streak = 0
            for i in range(total_frames - 1, -1, -1):
                if is_silence[0, i]:
                    silence_streak += 1
                else:
                    break
                    
            if silence_streak >= patience_frames:
                boundary_frame = total_frames - silence_streak
                boundary_sample = int((boundary_frame / total_frames) * len(buffer_audio))
                
                if boundary_sample > self.min_chunk_samples:
                    wav_chunk = buffer_audio[:boundary_sample]
                    
                    with self.buffer_lock:
                        for _ in range(boundary_sample):
                            self.audio_buffer.popleft()
                    
                    self.chunk_counter += 1
                    
                    self.inference_queue.put((
                        wav_chunk.copy(),
                        0,
                        boundary_sample,
                        self.chunk_counter
                    ))
            else:
                time.sleep(0.05)
    
    def start(self):
        self.running = True
        self.stop_event = threading.Event()
        
        self.capture_thread = threading.Thread(
            target=self._audio_capture_loop,
            args=(self.stop_event,)
        )
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            args=(self.stop_event,)
        )
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        self.boundary_thread = threading.Thread(
            target=self._boundary_detection_loop,
            args=(self.stop_event,)
        )
        self.boundary_thread.daemon = True
        self.boundary_thread.start()
    
    def stop(self):
        self.stop_event.set()
        
        self.capture_thread.join(timeout=2)
        self.inference_thread.join(timeout=2)
        self.boundary_thread.join(timeout=2)
        
        self.running = False
        
        while not self.output_queue.empty():
            chunk_id, text, path = self.output_queue.get()
    
    def get_latest_output(self, timeout=0.1):
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

def main():
    MODEL_DIR = '/home/nihal/CSE465/DuplexStreaming/STREAM/models2'
    ADAPTER_PATH = '/home/nihal/CSE465/Duplex_making/boundary_adapter.pt'
    TGT_LANG = 'ben'
    OUTPUT_DIR = '/home/nihal/CSE465/DuplexStreaming/STREAM/output'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = SeamlessM4TProcessor.from_pretrained(MODEL_DIR)
    
    base_model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map=device
    )
    base_model.eval()
    
    wrapper = SeamlessChunkedWrapper(base_model, ADAPTER_PATH).to(device)
    wrapper.eval()
    
    engine = LiveStreamingEngine(
        wrapper=wrapper,
        processor=processor,
        tgt_lang=TGT_LANG,
        silence_threshold=0.2,
        buffer_duration=30.0,
        min_chunk_duration=0.5,
        silence_patience_sec=0.8,
        output_dir=OUTPUT_DIR
    )
    
    try:
        engine.start()
        
        while engine.running:
            result = engine.get_latest_output(timeout=0.5)
            if result:
                chunk_id, text, path = result
                print(f"Chunk {chunk_id} Complete")
                print(f"Text: {text}")
                print(f"Audio: {path}")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()

if __name__ == "__main__":
    main()