import os
import json
import glob

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
ARCHIVE_DIR = "processed_archive"
CLEANED_DIR = "cleaned_outputs"

OUTPUT_JSON = "qlora_dataset.json"   # Standard JSON array format
OUTPUT_JSONL = "qlora_dataset.jsonl" # Line-delimited JSON (Best for Hugging Face/Unsloth)

INSTRUCTION_PROMPT = (
    "You are a Sinhala ASR correction system. "
    "Fix the spelling and grammar of the noisy input text while preserving all numbers and context."
)

def create_qlora_dataset():
    dataset = []
    
    # Locate all raw .txt files in the archive folder
    raw_files = glob.glob(os.path.join(ARCHIVE_DIR, "*.txt"))
    
    if not raw_files:
        print(f"❌ Error: No .txt files found in '{ARCHIVE_DIR}/'")
        return

    print(f"🔍 Found {len(raw_files)} files in '{ARCHIVE_DIR}/'. Matching counterparts...")
    
    matched_count = 0
    skipped_count = 0

    for file_path in raw_files:
        # Extract filename (e.g., "call_001.txt")
        filename = os.path.basename(file_path)
        name_only, ext = os.path.splitext(filename)
        
        # Build the expected cleaned filename (e.g., "call_001_cleaned.txt")
        cleaned_filename = f"{name_only}_cleaned{ext}"
        cleaned_path = os.path.join(CLEANED_DIR, cleaned_filename)
        
        # Check if the cleaned counterpart exists
        if not os.path.exists(cleaned_path):
            print(f"⚠️ Warning: Missing cleaned file for '{filename}'. Skipping...")
            skipped_count += 1
            continue
            
        # Read noisy text
        with open(file_path, 'r', encoding='utf-8') as f:
            noisy_text = f.read().strip()
            
        # Read cleaned text
        with open(cleaned_path, 'r', encoding='utf-8') as f:
            clean_text = f.read().strip()
            
        # Skip empty files
        if not noisy_text or not clean_text:
            print(f"⚠️ Warning: Empty content in '{filename}' or its cleaned file. Skipping...")
            skipped_count += 1
            continue
            
        # Build the Alpaca dictionary structure
        data_pair = {
            "instruction": INSTRUCTION_PROMPT,
            "input": noisy_text,
            "output": clean_text
        }
        
        dataset.append(data_pair)
        matched_count += 1

    print(f"✅ Successfully matched and structured {matched_count} file pairs. ({skipped_count} skipped)")

    # 1. Save as formatted JSON Array (qlora_dataset.json)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"📁 Exported JSON file: {OUTPUT_JSON}")

    # 2. Save as JSONL (qlora_dataset.jsonl)
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"📁 Exported JSONL file: {OUTPUT_JSONL}")

if __name__ == "__main__":
    create_qlora_dataset()