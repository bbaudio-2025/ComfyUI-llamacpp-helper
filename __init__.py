from .nodes import (LLamaCppModel, LLamaCppSkill, LLamaCppHelperLLM,
                   LLamaCppHelperStop)

NODE_CLASS_MAPPINGS = {
    "LLamaCppModel": LLamaCppModel,
    "LLamaCppSkill": LLamaCppSkill,
    "LLamaCppHelperLLM": LLamaCppHelperLLM,
    "LLamaCppHelperStop": LLamaCppHelperStop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLamaCppModel": "Load llama.cpp Model",
    "LLamaCppSkill": "Load llama.cpp Skill",
    "LLamaCppHelperLLM": "LLM (llama.cpp server)",
    "LLamaCppHelperStop": "Stop llama.cpp server",
}
