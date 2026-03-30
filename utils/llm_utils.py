import os
import streamlit as st
import google.generativeai as genai
from llama_cpp import Llama

# Unified Model Path (Gemma-2B)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_PATH = os.path.join(PROJECT_ROOT, "llm", "base_model", "gemma-2b-it.Q4_K_M.gguf")

class UniversalLLM:
    """A wrapper for Gemini or Local Llama to provide a consistent interface."""
    def __init__(self, mode="local", api_key=None, model_name="gemini-1.5-flash"):
        self.mode = mode
        self.api_key = api_key
        self.model_name = model_name
        self._llm = None

        if self.mode == "gemini" and self.api_key:
            genai.configure(api_key=self.api_key)
            self._llm = genai.GenerativeModel(model_name)
        else:
            # Fallback or choice of local
            self.mode = "local"
            self._llm = self._load_llama()

    def _load_llama(self):
        if not os.path.exists(GGUF_PATH):
            raise FileNotFoundError(f"GGUF model not found at {GGUF_PATH}")
        return Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            n_gpu_layers=0,
            verbose=False,
            chat_format="gemma",
        )

    def create_chat_completion(self, messages, max_tokens=500, temperature=0.7, stop=None):
        if self.mode == "gemini":
            # Format messages for Gemini (Gemini uses a different format)
            # system/user/assistant
            system_instruction = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
            
            # Recreate model with system instruction if present
            if system_instruction:
                self._llm = genai.GenerativeModel(self.model_name, system_instruction=system_instruction)

            # Build history
            history = []
            for msg in messages:
                if msg["role"] == "user":
                    history.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    history.append({"role": "model", "parts": [msg["content"]]})
            
            # Last message is the current one
            last_message = ""
            if history:
                last_msg_obj = history.pop()
                last_message = last_msg_obj["parts"][0]
            
            chat = self._llm.start_chat(history=history)
            response = chat.send_message(
                last_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop if stop else []
                )
            )
            
            # Wrap in a format compatible with llama-cpp-python output
            return {
                "choices": [{
                    "message": {
                        "content": response.text,
                        "role": "assistant"
                    }
                }]
            }
        else:
            # Local llama-cpp-python
            return self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )

    def __call__(self, prompt, **kwargs):
        """Standard __call__ for legacy/simple usage."""
        if self.mode == "gemini":
            response = self._llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=kwargs.get("max_tokens", 500),
                    temperature=kwargs.get("temperature", 0.7),
                    stop_sequences=kwargs.get("stop", [])
                )
            )
            # Wrap in a format compatible with llama-cpp-python output
            return {
                "choices": [{
                    "text": response.text
                }]
            }
        else:
            return self._llm(prompt, **kwargs)

def get_llm_mode():
    """Detect appropriate LLM mode and API key."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key and st.runtime.exists():
        api_key = st.secrets.get("GOOGLE_API_KEY")
    
    if api_key:
        return "gemini", api_key
    return "local", None

@st.cache_resource(show_spinner="⚙️  Initializing LLM Engine...")
def get_unified_llm():
    mode, api_key = get_llm_mode()
    return UniversalLLM(mode=mode, api_key=api_key)

# Legacy aliases to maintain compatibility
def get_llama_model():
    return get_unified_llm()
