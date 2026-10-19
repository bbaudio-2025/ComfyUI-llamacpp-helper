import base64
import io
import json
import os
import shlex
import tempfile
import urllib.error
import urllib.request
import wave

import torch
import folder_paths

from PIL import Image

from comfy_api.latest import io as comfy_io

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


def _clean_path(value):
    return str(value or "").strip().strip('"')


def _resolve_root(path):
    """Relative roots in config.json are relative to the ComfyUI root."""
    path = _clean_path(path)
    if not path:
        return ""
    if not os.path.isabs(path):
        path = os.path.join(folder_paths.base_path, path)
    return os.path.normpath(path)


try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)
except Exception:
    _cfg = {}

MODEL_ROOTS = [p for p in map(_resolve_root, _cfg.get("model_roots", _DEFAULT_ROOTS)) if os.path.isdir(p)]
MMPROJ_ROOTS = [p for p in map(_resolve_root, _cfg.get("mmproj_roots", _DEFAULT_ROOTS)) if os.path.isdir(p)]
# A key of its own rather than reusing model_roots: the dropdown it feeds lists
# ADAPTERS, and an adapter has the same .gguf extension as a model - only the
# folder tells them apart. Falling back to the model roots keeps the common
# setup (base model and adapters side by side) working.
LORA_ROOTS = [p for p in map(_resolve_root, _cfg.get("lora_roots", _cfg.get("model_roots", _DEFAULT_ROOTS)))
              if os.path.isdir(p)]
SKILL_ROOTS = [p for p in map(_resolve_root, _cfg.get("skill_roots", _DEFAULT_SKILL_ROOTS)) if os.path.isdir(p)]


def _resolve_path(chosen, override=""):
    """The effective path: a manual override wins when it is set.

    The dropdowns are built while ComfyUI builds node definitions, so a file
    dropped into a scanned folder afterwards cannot show up until a refresh -
    `override` is that escape hatch. An override of 'none' wins as well and
    turns the file OFF (e.g. drop an mmproj without touching the dropdown).
    """
    over = _clean_path(override)
    if over:
        return "" if over.lower() == "none" else over
    val = _clean_path(chosen)
    return "" if val.lower() == "none" else val


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


# Same "off" entry as the LoRA dropdown: picking it clears the mmproj path, so
# an unrelated projector sitting in a scanned folder is never loaded by default.
_MMPROJ_NONE = "none"


def _mmproj_options():
    projectors = [p for p in _scan_gguf(MMPROJ_ROOTS) if "mmproj" in os.path.basename(p).lower()]
    return [_MMPROJ_NONE] + projectors


# The dropdown entry that means "no adapter". A Combo's option IS the value it
# stores, so the off state has to be a readable string (an empty one would be a
# blank, near-invisible row); `_resolve_lora` turns it back into the empty
# string every consumer downstream compares against.
_LORA_NONE = "none"


def _lora_options():
    """The adapter ggups to offer, `_LORA_NONE` first ("no adapter").

    "Is this file an adapter?" lives in the GGUF's own metadata, not in its
    name - reading every header just to build a dropdown is not worth it, so
    the one name test that IS reliable stands in: a projector is never an
    adapter. The off entry comes first because it is also the default, so an
    adapter merely sitting in the folder never starts changing answers.
    """
    adapters = [p for p in _scan_gguf(LORA_ROOTS)
                if "mmproj" not in os.path.basename(p).lower()]
    return [_LORA_NONE] + adapters


def _resolve_lora(chosen, override=""):
    """The effective adapter path: a manual override wins when it is set.

    A path dropped into a scanned folder after the dropdown was built cannot
    appear in it until a refresh, so `override` is that escape hatch; an
    override of `_LORA_NONE`/"" turns the adapter OFF without touching the
    dropdown. Both spellings of "no adapter" collapse to "" so everything
    downstream sees one value.
    """
    over = str(override or "").strip().strip('"')
    val = over if over else str(chosen or "").strip().strip('"')
    return "" if val.lower() == _LORA_NONE else val


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


def _resolve_skill(chosen, skill_override=""):
    """The effective skill directory: an override wins when it is set.

    The dropdown list itself is the search surface - like the model and mmproj
    dropdowns, the front-end combo picker takes a filter string right on it,
    so nothing extra is needed to narrow a long list. `skill_override` is the
    escape hatch for a directory outside the scanned roots; 'none' turns the
    skill OFF (empty string).
    """
    return _resolve_path(chosen, skill_override)


def _fit_even(w, h, max_side):
    """Scale (w, h) down to fit within max_side, keeping the aspect ratio and
    even dimensions. Smaller inputs are left as-is but made even."""
    if max_side and max_side > 0 and max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        w = max(2, round(w * scale / 2.0) * 2)
        h = max(2, round(h * scale / 2.0) * 2)
    else:
        w -= w % 2
        h -= h % 2
    return w, h


def _image_to_data_urls(image, max_size=None):
    import torch
    from PIL import Image
    arr = image.detach().cpu().float().numpy() if hasattr(image, "cpu") else image
    urls = []
    for i in range(arr.shape[0]):
        img = (arr[i] * 255.0).clip(0, 255).astype("uint8")
        im = Image.fromarray(img, "RGB")
        if max_size:
            im = im.resize(_fit_even(im.size[0], im.size[1], max_size), Image.LANCZOS)
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
    # ComfyUI core "VIDEO" type is a VideoInput object with get_stream_source()
    # (file path or in-memory buffer) and save_to() for composited videos.
    if hasattr(video, "get_stream_source"):
        src = None
        try:
            src = video.get_stream_source()
        except Exception:
            src = None
        if isinstance(src, str) and os.path.isfile(src):
            path = src
        elif isinstance(src, io.BytesIO):
            return base64.b64encode(src.getvalue()).decode("ascii")
        elif isinstance(src, (bytes, bytearray)):
            return base64.b64encode(bytes(src)).decode("ascii")
        else:
            fd, path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            try:
                video.save_to(path)
            except Exception:
                os.remove(path)
                raise
    elif isinstance(video, str):
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


# Max sampled-frame dimension before encoding. Images above this are downscaled
# so gemma-class vision models don't blow up into many 896x896 patch tokens.
_MAX_VIDEO_SIDE = 768


def _sample_indices(total, count):
    """Return evenly spaced frame indices across [0, total) for 'count' takes."""
    count = max(1, min(int(count), total))
    if total <= 1:
        return [0]
    if count <= 1:
        return [total // 2]
    return sorted(set(round(i * (total - 1) / (count - 1)) for i in range(count)))


def _frame_to_data_url(pil_img, max_side):
    pil_img = pil_img.resize(_fit_even(*pil_img.size, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _video_frames_to_data_urls(video, max_frames, max_side=_MAX_VIDEO_SIDE):
    """Sample a ComfyUI VIDEO object into JPEG data URLs.

    Vision LLMs take images, not raw MP4s. ffmpeg extracts only the requested
    number of frames (bounded memory); PyAV is the guaranteed fallback. Both
    raise clear errors instead of silently returning an empty list, so a
    text-only request is never sent by accident.
    """
    source = None
    if hasattr(video, "get_stream_source"):
        try:
            source = video.get_stream_source()
        except Exception:
            source = None

    if isinstance(source, (str, io.BytesIO)):
        try:
            urls = _video_frames_via_ffmpeg(source, max_frames, max_side)
            if urls:
                return urls
        except Exception:
            pass  # no/failed ffmpeg binary on this host -> use bundled PyAV
        return _frames_via_av(source, max_frames, max_side)

    # Composited in-memory video without a stream source: decode once, sample.
    imgs = video.get_components().images
    if hasattr(imgs, "cpu"):
        imgs = imgs.cpu().numpy()
    n = imgs.shape[0]
    if n <= 0:
        raise RuntimeError("video has no frames")
    return [_frame_to_data_url(Image.fromarray(
        (imgs[i] * 255.0 if imgs[i].max() <= 1.01 else imgs[i]).round().clip(0, 255).astype("uint8")), max_side)
        for i in _sample_indices(n, max_frames)]


def _frames_via_av(source, max_frames, max_side):
    """Single streaming PyAV pass keeping only the sampled frames in memory."""
    import av

    counts = max(1, int(max_frames))
    with av.open(source) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Count by decoding: container nb_frames metadata can be wrong (e.g.
        # edited files), which would sample all frames from the video head.
        total = sum(1 for _ in stream.decode())
    if not total:
        raise RuntimeError("video stream reports no decodable frames")
    idx = set(_sample_indices(total, counts))
    images = {}
    with av.open(source) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for pos, frame in enumerate(stream.decode()):
            if pos in idx:
                images[pos] = frame.to_image()
    if not images:
        raise RuntimeError("PyAV decoded 0 video frames")
    return [_frame_to_data_url(images[i], max_side) for i in sorted(idx) if i in images]


def _video_frames_via_ffmpeg(source, max_frames, max_side):
    """Extract evenly-spaced frames with the ffmpeg binary from a path/BytesIO."""
    import ffmpeg

    cleanup_src = False
    if isinstance(source, io.BytesIO):
        fd, src = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with open(src, "wb") as f:
            f.write(source.getvalue())
        cleanup_src = True
    else:
        src = source
    try:
        probe = ffmpeg.probe(src)
        video_stream = next(
            (s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
        duration = float(
            video_stream.get("duration") or probe.get("format", {}).get("duration") or 0)
        if duration <= 0:
            raise RuntimeError("could not determine video duration")
        w = int(video_stream.get("width", 0) or 0)
        h = int(video_stream.get("height", 0) or 0)
        nw, nh = _fit_even(w, h, max_side)

        counts = max(1, int(max_frames))
        # Midpoints of N equal time slices: always strictly inside the file,
        # so -ss never lands on t=0 or EOF.
        times = [(k + 0.5) * duration / counts for k in range(counts)]

        urls = []
        last_err = None
        for t in times:
            fd, out = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)  # ffmpeg.exe must be able to open the file for writing
            try:
                args = dict(vframes=1)
                if nw > 0 and nh > 0:
                    args["vf"] = "scale=%d:%d" % (nw, nh)
                ffmpeg.input(src, ss=t).output(out, **args).overwrite_output().run(
                    quiet=True, capture_stdout=True, capture_stderr=True)
                with open(out, "rb") as f:
                    urls.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii"))
            except Exception as e:
                last_err = e
            finally:
                if os.path.exists(out):
                    os.remove(out)
        if not urls and last_err is not None:
            raise last_err
        return urls
    finally:
        if cleanup_src and os.path.exists(src):
            os.remove(src)


def _video_audio_to_b64(video, max_seconds):
    """Decode the video's audio track into base64 WAV (16 kHz mono, capped).

    Returns None when the video has no audio track, so callers can skip it.
    """
    import av

    source = None
    if hasattr(video, "get_stream_source"):
        try:
            source = video.get_stream_source()
        except Exception:
            source = None
    if not isinstance(source, (str, io.BytesIO)):
        return None  # composited videos carry no audio track

    rate = 16000
    cap = int(max(0, max_seconds)) * rate * 2  # 2 bytes per 16-bit sample
    with av.open(source) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        chunks = []
        total = 0
        for frame in stream.decode():
            res = resampler.resample(frame)
            for f in (res if isinstance(res, list) else [res]):
                data = f.to_ndarray().tobytes()
                chunks.append(data)
                total += len(data)
            if cap and total >= cap:
                break
    pcm = b"".join(chunks)[:cap] if cap else b"".join(chunks)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class LLamaCppModel:
    @classmethod
    def INPUT_TYPES(cls):
        models = _model_options()
        mmproj = _mmproj_options()
        lora = _lora_options()
        return {
            "required": {
                "server_exe": ("STRING", {"default": r"C:\Users\bbaudio\MyApps\AItemp\llama-bin-win-cuda-13.3-x64\llama-server.exe"}),
                "model_path": (models, {"default": models[0]}),
                "model_path_override": ("STRING", {"default": "",
                                                   "tooltip": "手动指定扫描目录之外的 .gguf 模型路径（填了优先于下拉框）。填 'none' 可不加载模型。"}),
                "mmproj_path": (mmproj, {"default": mmproj[0] if mmproj else ""}),
                "mmproj_path_override": ("STRING", {"default": "",
                                                    "tooltip": "手动指定扫描目录之外的 mmproj 路径（填了优先于下拉框）；填 'none' 可关闭 mmproj，即使上面选中了也无效。"}),
                "lora_path": (lora, {"default": lora[0],
                                     "tooltip": "可选的 LoRA 适配器（必须是 .gguf，llama.cpp 只读 GGUF 格式的 adapter）。'none' = 不加载；选中后随服务器启动通过 --lora 生效。"}),
                "lora_path_override": ("STRING", {"default": "",
                                                  "tooltip": "手动指定扫描目录之外的 adapter .gguf 路径（填了优先于下拉框）；填 'none' 可关闭。"}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                                         "tooltip": "adapter 的作用强度（--lora-scaled）。1.0（默认）等价于 --lora；0.0 = 加载但不应用；大于 1 放大，负值反向推离 adapter 学到的方向。修改会触发服务器重启。"}),
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8080, "min": 1, "max": 65535}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 9999,
                                         "tooltip": "放到 GPU 上的层数（-ngl）。-1（默认）= 全部层在 GPU 上，llama.cpp 把 -1、99、999 都当同一回事；0 = 纯 CPU，慢 10-40 倍；中间值是混合模式，只要有一层在 CPU 每个 token 都要传输一次，建议要么全上要么全不上。"}),
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

    def load(self, server_exe, model_path, model_path_override="", mmproj_path="",
             mmproj_path_override="", lora_path=_LORA_NONE, lora_path_override="", lora_scale=1.0,
             host="127.0.0.1", port=8080, n_gpu_layers=-1, context_size=4096,
             auto_start=True, mmproj_gpu_offload=True,
             advanced_launch_args="", model_name="llamacpp-helper"):
        model_path = _resolve_path(model_path, model_path_override)
        mmproj = _resolve_path(mmproj_path, mmproj_path_override)
        lora = _resolve_lora(lora_path, lora_path_override)
        extra = []
        if advanced_launch_args.strip():
            for line in advanced_launch_args.splitlines():
                line = line.strip()
                if line:
                    extra += shlex.split(line)
        if auto_start:
            sm.ensure_server(server_exe, model_path, host, port, n_gpu_layers, context_size,
                             extra_args=extra, timeout=180, mmproj=mmproj,
                             mmproj_gpu_offload=mmproj_gpu_offload, alias=model_name,
                             lora=lora, lora_scale=lora_scale)
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
            "lora": lora,
            "lora_scale": lora_scale,
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
                "skill_override": ("STRING", {"default": "",
                                              "tooltip": "手动指定扫描目录之外的 skill 目录路径（填了优先于下拉框）；填 'none' 可关闭 skill。下拉菜单本身的快速搜索框可以按名字筛选列表（和 model path / mmproj path 一样）。"}),
                "include_files": ("BOOLEAN", {"default": True}),
                "include_images": ("BOOLEAN", {"default": True}),
                "max_file_chars": ("INT", {"default": 8000, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("LLAMACPP_SKILL",)
    RETURN_NAMES = ("skill",)
    FUNCTION = "load"
    CATEGORY = MODEL_CATEGORY

    def load(self, skill, skill_override="", include_files=True, include_images=True,
             max_file_chars=8000):
        skill = _resolve_skill(skill, skill_override)
        if not skill:
            raise ValueError("no skill selected (dropdown empty, or someone "
                             "typed 'none' into 'skill_override')")
        if not os.path.isdir(skill):
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


LLAMACPP_MODEL = comfy_io.Custom("LLAMACPP_MODEL")
LLAMACPP_SKILL = comfy_io.Custom("LLAMACPP_SKILL")

# Same ceiling as the core 'Batch Images' node: enough headroom for any sane
# multi-image prompt without letting one node grow without bound.
_MAX_IMAGES = 50


class LLamaCppHelperLLM(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        images_template = comfy_io.Autogrow.TemplatePrefix(
            comfy_io.Image.Input("image", tooltip="单张图片（输入 batch 时只取第一张）。每个插槽独立缩放后发送，所以不同插槽的分辨率可以不同。"),
            prefix="image", min=0, max=_MAX_IMAGES)
        return comfy_io.Schema(
            node_id="LLamaCppHelperLLM",
            display_name="LLM (llama.cpp server)",
            category=MODEL_CATEGORY,
            description="Send one prompt to a llama.cpp server. Images use a 'Batch Images'-style autogrow input, so each slot can carry a different resolution instead of being forced into one batch.",
            inputs=[
                LLAMACPP_MODEL.Input("llamacpp_model"),
                comfy_io.String.Input("prompt", default="", multiline=True),
                comfy_io.String.Input("system_prompt", default="", multiline=True, optional=True),
                comfy_io.Autogrow.Input(
                    "images", optional=True, template=images_template,
                    tooltip="要发给模型的多张图片。和 'Batch Images' 节点一样可以点 '+' 增加插槽；每个插槽单独按 max_size 缩放后发送，各插槽分辨率可以不同。插槽只取输入 batch 的第一张，要发送整个 batch 请用下面的 image batch 输入。"),
                comfy_io.Image.Input("image_batch", display_name="image batch", optional=True,
                                     tooltip="整个 IMAGE batch 的每一帧都会发给模型（逐帧按 max_size 缩放）。单张或不同分辨率的图片请用上面的 image0、image1 插槽。"),
                comfy_io.Audio.Input("audio", optional=True),
                comfy_io.Video.Input("video", optional=True),
                LLAMACPP_SKILL.Input("skill", optional=True),
                comfy_io.String.Input("history", default="", multiline=True, optional=True),
                comfy_io.Combo.Input("video_mode", options=["frames", "blob"], default="frames", optional=True,
                               tooltip="frames: 抽取视频关键帧作为多张图片发送(推荐, 适合 gemma 等图像视觉模型)。blob: 把整个 mp4 以 base64 作为 input_video 发送(需要 server 支持 ffmpeg 抽帧)。"),
                comfy_io.Int.Input("video_frames", default=8, min=1, max=64, step=1, optional=True,
                             tooltip="video_mode=frames 时最多抽取的帧数，越多 token 越多、越慢。"),
                comfy_io.Boolean.Input("video_audio", default=False, optional=True,
                                 tooltip="同时把视频音轨转成 16kHz 单声道 wav 作为 input_audio 发送，需要 server 的 mmproj 支持音频（gemma 系列）。"),
                comfy_io.Int.Input("video_audio_max_seconds", default=60, min=0, max=1800, step=10, optional=True,
                             tooltip="发送音频的最大时长（秒），0 = 不限制；音频 token 随时长增长，超长建议截断。"),
                comfy_io.Int.Input("max_size", default=768, min=64, max=4096, step=64, optional=True,
                             tooltip="图片/视频最大边长，另一条边按原长宽比自动计算并取偶；更大输入会被缩小，用于控制 token 数。"),
                comfy_io.Boolean.Input("auto_start", default=True, optional=True),
                comfy_io.Boolean.Input("release_after_use", default=True, optional=True),
                comfy_io.Boolean.Input("thinking", default=False, optional=True,
                                 tooltip="启用模型的思考模式（qwen3 风格 /think）。关闭（默认）时会在 system prompt 后附加 '/no_think' 关掉推理输出：思考会拖慢回答、消耗 token 预算并加大被 max_tokens 截断的风险。"),
                comfy_io.Float.Input("temperature", default=0.8, min=0.0, max=2.0, step=0.01, optional=True),
                comfy_io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01, optional=True),
                comfy_io.Int.Input("top_k", default=40, min=0, max=1000, optional=True),
                comfy_io.Int.Input("max_tokens", default=1024, min=1, max=131072, optional=True),
                comfy_io.Float.Input("repeat_penalty", default=1.0, min=0.0, max=2.0, step=0.01, optional=True),
                comfy_io.Int.Input("seed", default=-1, min=-1, max=2**31 - 1, optional=True),
            ],
            outputs=[
                comfy_io.String.Output(display_name="text"),
                comfy_io.String.Output(display_name="reasoning"),
            ],
        )

    @classmethod
    def execute(cls, llamacpp_model, prompt, system_prompt="", images=None, image_batch=None, audio=None, video=None,
                skill=None, history="", video_mode="frames", video_frames=8, video_audio=False,
                video_audio_max_seconds=60, max_size=768,
                auto_start=True, release_after_use=True, thinking=False,
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
                    llamacpp_model.get("n_gpu_layers", -1),
                    llamacpp_model.get("context_size", 4096),
                    extra_args=llamacpp_model.get("extra_args"),
                    timeout=180,
                    mmproj=llamacpp_model.get("mmproj", ""),
                    mmproj_gpu_offload=llamacpp_model.get("mmproj_gpu_offload", True),
                    alias=model_name,
                    lora=llamacpp_model.get("lora", ""),
                    lora_scale=llamacpp_model.get("lora_scale", 1.0),
                )

        try:
            media = []
            if skill:
                for p in skill.get("images", []):
                    media.append({"type": "image_url", "image_url": {"url": _image_file_to_data_url(p)}})
            if images:
                for image in images.values():
                    if image is None:
                        continue
                    for url in _image_to_data_urls(image[:1], max_size):
                        media.append({"type": "image_url", "image_url": {"url": url}})
            if image_batch is not None:
                for url in _image_to_data_urls(image_batch, max_size):
                    media.append({"type": "image_url", "image_url": {"url": url}})
            if audio is not None:
                media.append({"type": "input_audio", "input_audio": {"data": _audio_to_b64(audio), "format": "wav"}})
            if video is not None:
                if video_mode == "blob":
                    media.append({"type": "input_video", "input_video": {"data": _video_to_b64(video), "format": "mp4"}})
                else:
                    for url in _video_frames_to_data_urls(video, video_frames, max_size):
                        media.append({"type": "image_url", "image_url": {"url": url}})
                    if video_audio:
                        aud_b64 = _video_audio_to_b64(video, video_audio_max_seconds)
                        if aud_b64:
                            media.append({"type": "input_audio", "input_audio": {"data": aud_b64, "format": "wav"}})
                        else:
                            print("[llamacpp-helper] video has no audio track, audio skipped")

            n_img = sum(1 for m in media if m.get("type") == "image_url")
            if n_img:
                print("[llamacpp-helper] sending %d image(s) to the model" % n_img)

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
            if not thinking:
                system_text = (system_text + "\n\n/no_think") if system_text else "/no_think"

            if history.strip():
                try:
                    messages = list(json.loads(history))
                except Exception as e:
                    return comfy_io.NodeOutput("history JSON error: %s" % e, "")
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
                return comfy_io.NodeOutput("LLM HTTP error %d: %s" % (e.code, detail), "")
            except Exception as e:
                return comfy_io.NodeOutput("LLM request error: %s" % e, "")

            msg = data["choices"][0]["message"]
            text = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            usage = data.get("usage") or {}
            print("[llamacpp-helper] server usage: prompt_tokens=%s completion_tokens=%s"
                  % (usage.get("prompt_tokens", "?"), usage.get("completion_tokens", "?")))
            return comfy_io.NodeOutput(text, reasoning)
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
