import base64
import io
import json
import os
import shlex
import urllib.error
import urllib.request
import wave

from . import server_manager as sm

MODEL_CATEGORY = "LLM/llama.cpp"

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

_DEFAULT_ROOTS = [
    r"C:\Users\bbaudio\MyApps\AItemp",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "llamacpp"),
]

_DEFAULT_SKILL_ROOTS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills"),
    r"C:\Users\bbaudio\.config\opencode\skill",
]

try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)
except Exception:
    _cfg = {}

MODEL_ROOTS = [p for p in _cfg.get("model_roots", _DEFAULT_ROOTS) if os.path.isdir(p)]
MMPROJ_ROOTS = [p for p in _cfg.get("mmproj_roots", _DEFAULT_ROOTS) if os.path.isdir(p)]
SKILL_ROOTS = [p for p in _cfg.get("skill_roots", _DEFAULT_SKILL_ROOTS) if os.path.isdir(p)]


def _resolve(value):
    if value and os.path.exists(value):
        return value
    return value


def _scan_gguf(roots):
    out = []
    for r in roots:
        for dp, _, fns in os.walk(r):
            for fn in fns:
                if fn.lower().endswith(".gguf"):
                    out.append(os.path.join(dp, fn))
    return sorted(out)


def _model_options():
    return _scan_gguf(MODEL_ROOTS) or [""]


def _mmproj_options():
    return [p for p in _scan_gguf(MMPROJ_ROOTS) if "mmproj" in os.path.basename(p).lower()] or [""]


_TEXT_EXTS = {
    ".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml",
    ".csv", ".html", ".css", ".sh", ".bat", ".ps1", ".r", ".ipynb",
}

_EXT_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
}


def _image_file_to_data_url(path):
    ext = os.path.splitext(path)[1].lower()
    mime = _EXT_MIME.get(ext, "image/png")
    with open(path, "rb") as f:
        return "data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode("ascii"))


def _scan_skills(roots):
    out = []
    for r in roots:
        for dp, _, fns in os.walk(r):
            if any(fn.lower() == "skill.md" for fn in fns):
                out.append(dp)
    return sorted(out)


def _skill_options():
    return _scan_skills(SKILL_ROOTS) or [""]


def _image_to_data_urls(image):
    import torch
    from PIL import Image
    arr = image.detach().cpu().float().numpy() if hasattr(image, "cpu") else image
    urls = []
    for i in range(arr.shape[0]):
        img = (arr[i] * 255.0).clip(0, 255).astype("uint8")
        im = Image.fromarray(img, "RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    return urls


def _audio_to_b64(audio):
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    arr = wf.detach().cpu().numpy() if hasattr(wf, "cpu") else wf
    if arr.ndim == 3:
        arr = arr[0]
    arr = (arr * 32767.0).clip(-32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(arr.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(arr.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _video_to_b64(video):
    path = None
    if isinstance(video, str):
        path = video
    elif isinstance(video, dict):
        path = video.get("path") or video.get("filename") or video.get("video")
    else:
        for attr in ("path", "filename", "video_path", "file"):
            v = getattr(video, attr, None)
            if v:
                path = v
                break
    if not path:
        raise ValueError("could not determine video file path")
    if not os.path.exists(path):
        raise FileNotFoundError("video file not found: %s" % path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


class LLamaCppModel:
    @classmethod
    def INPUT_TYPES(cls):
        models = _model_options()
        mmproj = _mmproj_options()
        return {
            "required": {
                "server_exe": ("STRING", {"default": r"C:\Users\bbaudio\MyApps\AItemp\llama-bin-win-cuda-13.3-x64\llama-server.exe"}),
                "model_path": (models, {"default": models[0]}),
                "mmproj_path": (mmproj, {"default": mmproj[0] if mmproj else ""}),
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8080, "min": 1, "max": 65535}),
                "n_gpu_layers": ("INT", {"default": 99, "min": 0, "max": 9999}),
                "context_size": ("INT", {"default": 4096, "min": 256, "max": 131072}),
            },
            "optional": {
                "auto_start": ("BOOLEAN", {"default": True}),
                "mmproj_gpu_offload": ("BOOLEAN", {"default": True}),
                "advanced_launch_args": ("STRING", {"default": "", "multiline": True}),
                "model_name": ("STRING", {"default": "llamacpp-helper"}),
            },
        }

    RETURN_TYPES = ("LLAMACPP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = MODEL_CATEGORY

    def load(self, server_exe, model_path, mmproj_path, host, port, n_gpu_layers, context_size,
             auto_start=True, mmproj_gpu_offload=True,
             advanced_launch_args="", model_name="llamacpp-helper"):
        model_path = _resolve(model_path)
        mmproj = _resolve(mmproj_path) if mmproj_path and mmproj_path != "" else ""
        extra = []
        if advanced_launch_args.strip():
            for line in advanced_launch_args.splitlines():
                line = line.strip()
                if line:
                    extra += shlex.split(line)
        if auto_start:
            sm.ensure_server(server_exe, model_path, host, port, n_gpu_layers, context_size,
                             extra_args=extra, timeout=180, mmproj=mmproj,
                             mmproj_gpu_offload=mmproj_gpu_offload, alias=model_name)
        return ({
            "model_path": model_path,
            "host": host,
            "port": port,
            "model_name": model_name,
            "server_exe": server_exe,
            "n_gpu_layers": n_gpu_layers,
            "context_size": context_size,
            "mmproj": mmproj,
            "mmproj_gpu_offload": mmproj_gpu_offload,
            "extra_args": extra,
            "managed": bool(auto_start),
        },)


class LLamaCppSkill:
    @classmethod
    def INPUT_TYPES(cls):
        skills = _skill_options()
        return {
            "required": {
                "skill": (skills, {"default": skills[0] if skills else ""}),
            },
            "optional": {
                "include_files": ("BOOLEAN", {"default": True}),
                "include_images": ("BOOLEAN", {"default": True}),
                "max_file_chars": ("INT", {"default": 8000, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("LLAMACPP_SKILL",)
    RETURN_NAMES = ("skill",)
    FUNCTION = "load"
    CATEGORY = MODEL_CATEGORY

    def load(self, skill, include_files=True, include_images=True, max_file_chars=8000):
        if not skill or not os.path.isdir(skill):
            raise ValueError("skill directory not found: %s" % skill)
        instruction = None
        for fn in os.listdir(skill):
            if fn.lower() == "skill.md":
                with open(os.path.join(skill, fn), "r", encoding="utf-8") as f:
                    instruction = f.read()
                break
        if instruction is None:
            raise ValueError("no SKILL.md found in %s" % skill)

        references = []
        images = []
        if include_files or include_images:
            for dp, _, fns in os.walk(skill):
                for fn in fns:
                    full = os.path.join(dp, fn)
                    rel = os.path.relpath(full, skill)
                    ext = os.path.splitext(fn)[1].lower()
                    if include_images and ext in _EXT_MIME:
                        images.append(full)
                        continue
                    if include_files and ext in _TEXT_EXTS:
                        try:
                            with open(full, "r", encoding="utf-8") as f:
                                content = f.read()
                        except Exception:
                            continue
                        if max_file_chars > 0 and len(content) > max_file_chars:
                            content = content[:max_file_chars] + "\n...[truncated]"
                        references.append("### %s\n%s" % (rel, content))

        return ({
            "name": os.path.basename(skill),
            "instruction": instruction,
            "references": "\n\n".join(references),
            "images": images,
        },)


class LLamaCppHelperLLM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llamacpp_model": ("LLAMACPP_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "video": ("VIDEO",),
                "skill": ("LLAMACPP_SKILL",),
                "history": ("STRING", {"default": "", "multiline": True}),
                "auto_start": ("BOOLEAN", {"default": True}),
                "release_after_use": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 131072}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning")
    FUNCTION = "generate"
    CATEGORY = MODEL_CATEGORY

    def generate(self, llamacpp_model, prompt, system_prompt="", image=None, audio=None, video=None,
                 skill=None, history="", auto_start=True, release_after_use=True,
                 temperature=0.8, top_p=0.95, top_k=40, max_tokens=1024, repeat_penalty=1.0, seed=-1):
        if not llamacpp_model.get("model_path"):
            raise ValueError("llamacpp_model is not loaded")

        host = llamacpp_model["host"]
        port = llamacpp_model["port"]
        model_name = llamacpp_model.get("model_name", "llamacpp-helper")

        if auto_start and llamacpp_model.get("managed"):
            exe = llamacpp_model.get("server_exe")
            if exe:
                sm.ensure_server(
                    exe, llamacpp_model["model_path"], host, port,
                    llamacpp_model.get("n_gpu_layers", 99),
                    llamacpp_model.get("context_size", 4096),
                    extra_args=llamacpp_model.get("extra_args"),
                    timeout=180,
                    mmproj=llamacpp_model.get("mmproj", ""),
                    mmproj_gpu_offload=llamacpp_model.get("mmproj_gpu_offload", True),
                    alias=model_name,
                )

        try:
            media = []
            if skill:
                for p in skill.get("images", []):
                    media.append({"type": "image_url", "image_url": {"url": _image_file_to_data_url(p)}})
            if image is not None:
                for url in _image_to_data_urls(image):
                    media.append({"type": "image_url", "image_url": {"url": url}})
            if audio is not None:
                media.append({"type": "input_audio", "input_audio": {"data": _audio_to_b64(audio), "format": "wav"}})
            if video is not None:
                media.append({"type": "input_video", "input_video": {"data": _video_to_b64(video), "format": "mp4"}})

            content = [{"type": "text", "text": prompt}] + media if media else prompt

            sys_parts = []
            if skill:
                if skill.get("instruction"):
                    sys_parts.append(skill["instruction"])
                if skill.get("references"):
                    sys_parts.append(skill["references"])
            if system_prompt.strip():
                sys_parts.append(system_prompt)
            system_text = "\n\n".join(p for p in sys_parts if p.strip())

            if history.strip():
                try:
                    messages = list(json.loads(history))
                except Exception as e:
                    return ("history JSON error: %s" % e, "")
                if system_text:
                    messages.insert(0, {"role": "system", "content": system_text})
                messages.append({"role": "user", "content": content})
            else:
                messages = []
                if system_text:
                    messages.append({"role": "system", "content": system_text})
                messages.append({"role": "user", "content": content})

            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "repeat_penalty": repeat_penalty,
                "seed": seed,
            }
            url = "http://%s:%d/v1/chat/completions" % (host, port)
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                return ("LLM HTTP error %d: %s" % (e.code, detail), "")
            except Exception as e:
                return ("LLM request error: %s" % e, "")

            msg = data["choices"][0]["message"]
            text = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            return (text, reasoning)
        finally:
            if release_after_use and llamacpp_model.get("managed"):
                sm.stop_server(port)


class LLamaCppHelperStop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "port": ("INT", {"default": 8080, "min": 1, "max": 65535}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "stop"
    CATEGORY = MODEL_CATEGORY

    def stop(self, port):
        sm.stop_server(port)
        return ("stopped server on port %d" % port,)
