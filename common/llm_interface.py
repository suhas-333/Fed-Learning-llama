# Fedrated-Learning-flwr/common/llm_interface.py

from ctransformers import AutoModelForCausalLM
import json
import sys

# Global variable to hold the model, so we only load it once per script instance
llm_model = None

def load_local_llm():
    """Loads the quantized LLM into memory if it's not already loaded."""
    global llm_model
    if llm_model is None:
        print("[LLM] Loading local TinyLlama model... This may take a moment.")
        try:
            llm_model = AutoModelForCausalLM.from_pretrained(
                "../Models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
                model_type="llama",
                gpu_layers=1, # Set to 0 if you don't have a GPU or encounter errors
                context_length=2048
            )
            print("[LLM] Model loaded successfully.")
        except Exception as e:
            print(f"[LLM FATAL ERROR] Could not load the model. Error: {e}")
            sys.exit(1)
    return llm_model

def get_llm_response(prompt_text, system_message):
    """
    Gets a response from the LLM and tries to parse it as JSON.
    Crucially, it ALWAYS returns the raw text output as a fallback for chat.

    Returns:
        tuple: (parsed_json, raw_text_output)
               - parsed_json is a dictionary if successful, otherwise None.
               - raw_text_output is the full, unprocessed string from the LLM.
    """
    llm = load_local_llm()
    
    full_prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{prompt_text}</s>\n<|assistant|>\n"
    
    # Generate a response
    raw_output = llm(full_prompt, max_new_tokens=200, temperature=0.2)
    
    # The debug print statement has been removed for a cleaner UI.
    
    try:
        # Try to extract and parse JSON from the response
        json_str = raw_output[raw_output.find('{'):raw_output.rfind('}')+1]
        parsed_json = json.loads(json_str)
        # If parsing succeeds, return the JSON and the raw text
        return parsed_json, raw_output
    except json.JSONDecodeError:
        # If parsing fails, return None for the JSON part, but still return the raw text
        return None, raw_output