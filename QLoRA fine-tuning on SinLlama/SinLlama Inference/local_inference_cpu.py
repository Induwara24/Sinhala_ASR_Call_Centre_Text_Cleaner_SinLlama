import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. Configuration & Loading
# ==========================================
LORA_BACKUP_PATH = "./lora_backup"
BASE_MODEL_ID = "SAWithanage/SinLlama-Llama-3-8B-Merged"

print("Loading Base Model onto CPU (This may take several minutes and 16GB+ RAM)...")
# Load the base model natively on CPU
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="cpu",
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True
)

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_BACKUP_PATH)

print("Injecting LoRA Adapters...")
# Merge the fine-tuned LoRA weights onto the base model
model = PeftModel.from_pretrained(base_model, LORA_BACKUP_PATH)
model.eval() # Set to inference mode

print("Model Ready for Inference on CPU!\n")

# ==========================================
# 2. The Inference Generator
# ==========================================
def clean_sinhala_text(noisy_text):
    # The exact Alpaca prompt used during training[cite: 1]
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a Sinhala ASR correction system. Fix the spelling and grammar of the noisy input text while preserving all numbers and context.

### Input:
{input}

### Response:
"""
    # Format the prompt with the noisy text[cite: 1]
    formatted_prompt = alpaca_prompt.format(input=noisy_text)
    
    # Tokenize and explicitly send to CPU (removing the .to("cuda") from Colab)[cite: 1]
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cpu")

    # Generate the output[cite: 1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, 
            use_cache=True, 
            temperature=0.1, # Low temperature for accurate correction[cite: 1]
            pad_token_id=tokenizer.eos_token_id 
        )

    # Decode and extract the response[cite: 1]
    full_output = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]

    try:
        # Split at the response trigger[cite: 1]
        generated_text = full_output.split("### Response:\n")[-1]
        
        # Clean up hidden tokens[cite: 1]
        generated_text = generated_text.replace(tokenizer.eos_token, "").strip() 
        generated_text = generated_text.replace("<|eot_id|>", "").strip() 
    except Exception:
        generated_text = "Error: Could not parse output." 

    return generated_text

# ==========================================
# 3. Execution & Testing
# ==========================================
if __name__ == "__main__":
    # Test text taken directly from the Colab evaluation cells[cite: 1]
    test_text = "ඕය ම පිනකයි මට පුවන් සයවන්නහෙලෝ න්ටඉන්න හෙලෝ පේඔව් කියන්න සර් හෙලෝ ඔව් මගේ රවුටර බිල් එක මට දැන් මාස දෙක් විතර ගෙවාගන්න බැරි වුනා මේ ම වෙනදට බිල් එක පේ කරන්නේ විදිහට ඒ කරගන්න විදිහන්නේනෑහැ ඔන්ලයින් පමේ පේමන්ට් එකට බිල් වීව් එක කියරලා තමයි යන්නේ ම මට දැන් ඒකේ රවුටර් බිල් එකේ මවුන්ට් එක හරියට දැනගන්නයි කොහොමද මේ ඔන්ලයින් පේමන්ට් එක කරගන්න පුළුවන් විදිහක් පලිය කරගන්න තමයි ගත්තේ හරි ඔය සර්ගේ අදාළ එස් එල්ටී කනෙක්ෂන් එකේ නම්බර් එක කියන්න බංදු දයි හරි යි දෙකයි හරි අයි ර කරුණාකර ඇතුමේ රැඳී ඉන්නඕකරැඳිටාට ස්තූතියි සර් මෙතන කනෙක්ෂන් හිමිකරුගේ නම කියන්න මිස්ටර් විතර් සිංහදඔව් එතකොට සර් කොව වෙනදා කොහොමද පේමන්ට් එක කරන්නේ සර් ලයින් එකට දැන ටරනම් ලයින් එක ස්පේන්ඩ් වෙලා තියෙන්නේ ම සාමාන්යෙන් කරගෙන ආවේ මගේ වීව් කරනවා ඔන්ලයින් බිල් වීව් එකෙන් බලා ඒක එවලේ ඒ කියන්නේ ඔන්ලයින් විදිහට ප්රොසීඩ් කරන එක තමයි පේමන්ට් එක කරේ ම එහෙම තමයි සෑහෙන කාලයක් රගෙන මේ මාස දෙක් තිස්සේ මට ඒක මේ වීව් වෙන්නේ නෑ කොච්චර කරත් බිල් එක වීව් වෙන්නේ නැහැ මේ ඒ ප්රශ්නෙ නිසා තමයි ඇත්තකට ම මේ ගෙවුනේ නැත්තේ හරි ම සර්ට ලින්ක් දෙක් එවන්න නම් බිල් වීව් ලින්ක් එකයි ඒ වගේම පේ ඔන්ලයින් ලින්ක් එකුයි සර්ගේ මොබයල් කන්ටැක්ට් නම්බර් එක කියන්න මේ කතා කරන අංකයට ගන්න පුළුවන්ද ඔහ් මේ නම්බර් එකට පුළුවන්රන ම්බ්යට හරි දැන් ඔන්වන් නම්බර් වීප කවුන්ට් නම්බර් එක යටතේ සර්ගේ තියෙන්නේ කනෙක්ෂන් එක ෆූජී කනෙක්ෂන් එක් නේද ඔව් ෂ ඉන්ටර්නෙට් එකට විතරක් භාවිතා කරන එක සර්විස් එක් හරි ඇමතුවට රැඳී න්නේ සර් ඔවමතුම රැඳී න්නේ සර් ලයින් එක ඉන්නකේ හරි සර් රැනටිසටාට ස්තියි ම දැන් ලින්ක් දෙක් සර්ට එවලා තියෙනවා එක් පේ ඉන්ස්ටන්ලි කියන ලින්ක් එක අනිත් එක ේඔමේවිව් කරගන්න ලින්ක් එක සර් මේ දෙකම භාිවිනවන් එකේ කරන්න පුළුවන්ද කියලා හරි සර් වෙනත් යමක් දැනගන්න අවශ්යද දැනට ම මේකේ ප්රොසීඩ් කරලා බලන්නම් කොහොමද බිල් එක පෙන්න්නේ මවුන්ට් එක මොක්ද සර් බිල්මවුන්ට් එක මේ වෙනකොට කීයද පෙන්න් බිල් මවුන්ට් එක දැනට් පෙන්නුම් කරනවා මේක අප්ඩේටඩ් වෙලා තියෙන්නේ නි මාසේ දක්වා් දැන් ජනවාරි මාසේ පළවෙනිදත් බිල් එක ඉෂූ වෙලා තියෙනවා ක් සවි අටක් කේ හරි සර් තැන්ක් යූ එස් එල්ටී මොබිටෙල් ඇමතුවාට ස්තූතියි සුභ දවක් ඇමතුමඇගයසකන්"

    print("Original (Noisy) :", test_text) 
    print("--------------------------------------------------") 
    print("Cleaned Output   :", clean_sinhala_text(test_text)) 