# QUICK FIX: Add this cell right before running the benchmark in Phase 0

# Fix the processor None issue
if processor is None:
    print("WARNING: processor is None, reloading...")
    from transformers import SeamlessM4TProcessor
    MODEL_NAME = 'facebook/seamless-m4t-v2-large'
    processor = SeamlessM4TProcessor.from_pretrained(MODEL_NAME)
    print("Processor reloaded successfully")
else:
    print("Processor is already loaded")

# Verify processor is working
print(f"Processor type: {type(processor)}")
print(f"Processor model: {getattr(processor, 'model_name', 'Unknown')}")