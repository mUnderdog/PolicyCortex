import os
import streamlit as st
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# Unified Model Path (Gemma-2B)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_PATH = os.path.join(PROJECT_ROOT, "llm", "base_model", "gemma-2b-it.Q4_K_M.gguf")

class UniversalLLM:
    """A wrapper for Groq or Local Llama to provide a consistent interface."""
    def __init__(self, mode="local", api_key=None, model_name="llama-3.3-70b-versatile"):
        self.mode = mode
        self.api_key = api_key
        self.model_name = model_name
        self._llm = None

        if self.mode == "groq" and self.api_key:
            if not GROQ_AVAILABLE:
                raise ImportError("groq library is not installed. Please run 'pip install groq'.")
            self._llm = Groq(api_key=self.api_key)
        else:
            # Fallback or choice of local
            self.mode = "local"
            if not LLAMA_AVAILABLE:
                # If llama-cpp isn't here (cloud), we must have an API key or we fail
                if self.api_key:
                    self.mode = "groq"
                    self._llm = Groq(api_key=self.api_key)
                else:
                    raise ImportError("No LLM engine available. Please provide GROQ_API_KEY or install llama-cpp-python.")
            else:
                self._llm = self._load_llama()

    def _load_llama(self):
        if not os.path.exists(GGUF_PATH):
            # If model file is missing but library is present, we still can't run local
            return None
        return Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            n_gpu_layers=0,
            verbose=False,
            chat_format="gemma",
        )

    def create_chat_completion(self, messages, max_tokens=1024, temperature=0.7, stop=None):
        if self.mode == "groq":
            try:
                response = self._llm.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop
                )
                return {
                    "choices": [{
                        "message": {
                            "content": response.choices[0].message.content,
                            "role": "assistant"
                        }
                    }]
                }
            except Exception as e:
                return {"error": str(e), "choices": [{"message": {"content": f"Error from Groq: {e}", "role": "assistant"}}]}
        else:
            # Local llama-cpp-python
            if self._llm is None:
                return {"choices": [{"message": {"content": "Local model file (.gguf) not found. Please provide a GROQ_API_KEY for cloud inference.", "role": "assistant"}}]}
            
            return self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )

    def __call__(self, prompt, **kwargs):
        """Standard __call__ for legacy/simple usage (returns 'text' key for compatibility)."""
        if self.mode == "groq":
            messages = [{"role": "user", "content": prompt}]
            resp = self.create_chat_completion(messages, **kwargs)
            if "choices" in resp:
                content = resp["choices"][0]["message"]["content"]
                resp["choices"][0]["text"] = content
            return resp
        else:
            if self._llm is None:
                return {"choices": [{"text": "Local model file not found."}]}
            return self._llm(prompt, **kwargs)

def get_llm_mode():
    """Detect appropriate LLM mode and API key."""
    # Check Environment Variables
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Check Streamlit Secrets if in a runtime
    if not api_key:
        try:
            if st.runtime.exists():
                api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
            
    if api_key:
        return "groq", api_key
    
    # Check for legacy Gemini key as fallback if the user hasn't switched envs yet
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        # Note: We aren't supporting Gemini in the NEW code, 
        # but we returning local to trigger prompt for Groq
        pass

    return "local", None

@st.cache_resource(show_spinner="⚙️  Initializing LLM Engine...")
def get_unified_llm():
    mode, api_key = get_llm_mode()
    try:
        return UniversalLLM(mode=mode, api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

# Legacy aliases
def get_llama_model():
    return get_unified_llm()
