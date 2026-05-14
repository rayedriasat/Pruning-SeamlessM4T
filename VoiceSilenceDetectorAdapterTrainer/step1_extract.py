import os
import glob
import random
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4TProcessor

# --- Configuration ---
LIBRITTS_DIR = "/home/nihal/CSE465/DatasetCollection/LibriTTS/train-clean-100"
MODEL_DIR = "/home/nihal/CSE465/seamV5/models/phase7_dora_merged_v1"
OUTPUT_DIR = "/home/nihal/CSE465/Duplex_making/working_dataset"

TARGET_SR = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_and_save_split(wav_list, split_name, model, processor, num_pairs=1000):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Processing {split_name} split ({num_pairs} pairs)...")
    
    for i in tqdm(range(num_pairs)):
        # 1. Select 2 random audio files
        wav1_path = random.choice(wav_list)
        wav2_path = random.choice(wav_list)
        
        # 2. Load and Resample (Checking Khz as requested)
        audio1, sr1 = torchaudio.load(wav1_path)
        audio2, sr2 = torchaudio.load(wav2_path)
        
        if sr1 != TARGET_SR: audio1 = torchaudio.transforms.Resample(sr1, TARGET_SR)(audio1)
        if sr2 != TARGET_SR: audio2 = torchaudio.transforms.Resample(sr2, TARGET_SR)(audio2)
            
        audio1 = audio1.squeeze(0)
        audio2 = audio2.squeeze(0)
        
        # 3. Generate random silence (1 to 3 seconds)
        silence_sec = random.uniform(1.0, 3.0)
        silence_samples = int(silence_sec * TARGET_SR)
        silence = torch.zeros(silence_samples)
        
        # 4. Concatenate Audio and Create Raw Labels
        merged_audio = torch.cat([audio1, silence, audio2])
        raw_labels = torch.cat([
            torch.ones(len(audio1)), 
            torch.zeros(len(silence)), 
            torch.ones(len(audio2))
        ])
        
        # 5. Extract Features using SeamlessM4T Encoder
        inputs = processor(audio=merged_audio.numpy(), sampling_rate=TARGET_SR, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            encoder_outputs = model.speech_encoder(**inputs, return_dict=True)
            hidden_states = encoder_outputs.last_hidden_state.cpu() # Shape: [1, T, 1024]
            
        # 6. Align Labels to Encoder Output Sequence Length (T)
        seq_len = hidden_states.size(1)
        aligned_labels = F.interpolate(
            raw_labels.unsqueeze(0).unsqueeze(0).float(), 
            size=seq_len, 
            mode='linear', 
            align_corners=False
        ).squeeze() # Shape: [T]
        
        # 7. Save to disk
        torch.save(hidden_states.squeeze(0), os.path.join(split_dir, f"sample_{i}_features.pt"))
        torch.save(aligned_labels, os.path.join(split_dir, f"sample_{i}_labels.pt"))

if __name__ == "__main__":
    # Load Model Once
    print("Loading Base Model for Feature Extraction...")
    processor = SeamlessM4TProcessor.from_pretrained(MODEL_DIR)
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map=device)
    model.eval()

    # Get all wavs and shuffle
    all_wavs = glob.glob(os.path.join(LIBRITTS_DIR, "*", "*", "*.wav"))
    random.shuffle(all_wavs)
    
    # Split Data (80% Train, 10% Val, 10% Test)
    total = len(all_wavs)
    train_wavs = all_wavs[:int(0.8 * total)]
    val_wavs = all_wavs[int(0.8 * total):int(0.9 * total)]
    test_wavs = all_wavs[int(0.9 * total):]
    
    # Process
    process_and_save_split(train_wavs, "train", model, processor, num_pairs=5000)
    process_and_save_split(val_wavs, "val", model, processor, num_pairs=500)
    process_and_save_split(test_wavs, "test", model, processor, num_pairs=500)
    print("Dataset Generation Complete.")