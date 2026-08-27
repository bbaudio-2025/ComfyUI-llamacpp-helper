# ComfyUI-llamacpp-helper

通过 ComfyUI 驱动本地 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的
`llama-server`。插件会把 `llama-server.exe` 作为独立进程启动，并通过 HTTP
（`/v1/chat/completions`）与之通信，因此 ComfyUI 本身从不加载模型。

## 中文

### 为什么用这个插件
- **无需额外 Python 依赖。** 插件只用到 Python 标准库和 ComfyUI 自带的
  `folder_paths`，不需要在 ComfyUI 环境里安装 `llama_cpp_python`（或任何其它包）。
- **用完即释放显存 / 内存。** 推理在独立的 `llama-server` 进程中进行，开启
  `release_after_use`（默认）后，生成结束会终止该进程，GPU 显存和系统内存立即释放，
  不会被持续占用。
- **多模态。** 支持文本、图片、音频、视频输入（图片/音频/视频需要对应的 `mmproj`
  且模型本身支持多模态）。
- **Skills。** 可将一个技能目录（`SKILL.md` + 文本/媒体文件）作为系统提示注入。
- **服务自动管理。** 设 `auto_start`（默认）后，首次使用自动启动服务，
  在 `release_after_use` 关停后会再次自动拉起。

### 节点
- **Load llama.cpp Model** — 配置 `server_exe`（llama-server 可执行文件）、
  `model_path`（`.gguf` 模型），以及可选的 `mmproj_path`（多模态投影，也是 `.gguf`）。
  其它参数：host/port、`n_gpu_layers`、`context_size`、`auto_start`。
- **Load llama.cpp Skill** — 从 `skill_roots` 加载技能目录。
- **LLM (llama.cpp server)** — 生成节点。输入：模型、prompt，可选 `system_prompt`、
  `image`、`audio`、`video`、`skill`、`history`，以及采样参数（`temperature`、
  `top_p`、`top_k`、`max_tokens`、`repeat_penalty`、`seed`）和生命周期开关
  （`auto_start`、`release_after_use`）。输出：`(text, reasoning)`。
- **Stop llama.cpp server** — 手动停止指定端口上的服务。

### 使用说明
1. **手动编辑 `config.json`。** 这是最重要的一步：在插件目录下打开 `config.json`，
   把 `model_roots` / `mmproj_roots` / `skill_roots` 改成你机器上真实的目录
   （默认值包含 ComfyUI 的 `models/text_encoders` 与 `models/LLM`，以及示例路径）。
   模型下拉框只会扫描这些目录里的 `.gguf` 文件，填对路径才能在下拉框里选到模型；
   若留空或没有匹配项，可手动粘贴完整路径。
2. 把本插件放进 `ComfyUI/custom_nodes/`（若使用管理器安装或者git clone可跳过）。
3. 添加 **Load llama.cpp Model**，把 `server_exe` 指向你的 `llama-server.exe`，
   `model_path` 指向一个 GGUF 模型；仅多模态模型才需要填 `mmproj_path`。
4. 添加 **LLM (llama.cpp server)**，连上模型，填写 prompt，可选连接
   图片 / 音频 / 视频 / 技能。
5. 执行队列。服务会自动启动（`auto_start`），结束后释放显存/内存
   （`release_after_use`）。
6. 模型必须是 GGUF 格式——`llama-server` 不能直接加载 safetensors。

---

## English

Drive a local [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`
from ComfyUI. The plugin launches `llama-server.exe` as a separate process and
talks to it over HTTP (`/v1/chat/completions`), so ComfyUI itself never loads
the model.

### Why use it
- **No extra Python dependencies.** The plugin only uses the Python standard
  library plus ComfyUI's own `folder_paths`. You do **not** need to install
  `llama_cpp_python` (or anything else) into the ComfyUI environment.
- **VRAM / RAM is released after use.** Inference runs in an external
  `llama-server` process. With `release_after_use` enabled (default), the server
  is terminated when generation finishes, so the GPU memory and system memory
  are freed immediately instead of staying occupied.
- **Multimodal.** Text, image, audio and video inputs are supported (image /
  audio / video need a matching `mmproj` and the model's multimodal support).
- **Skills.** A skill directory (`SKILL.md` + text/media files) can be injected
  as a system prompt.
- **Auto-managed server.** Set `auto_start` (default) and the server is launched
  on first use and re-launched automatically after a `release_after_use` stop.

### Nodes
- **Load llama.cpp Model** — configure `server_exe` (the llama-server binary),
  `model_path` (a `.gguf` file), and optional `mmproj_path` (multimodal
  projector, also `.gguf`). Other params: host/port, `n_gpu_layers`,
  `context_size`, `auto_start`.
- **Load llama.cpp Skill** — load a skill folder from `skill_roots`.
- **LLM (llama.cpp server)** — the generation node. Inputs: model, prompt,
  optional `system_prompt`, `image`, `audio`, `video`, `skill`, `history`, plus
  sampling params (`temperature`, `top_p`, `top_k`, `max_tokens`, `repeat_penalty`,
  `seed`) and lifecycle flags (`auto_start`, `release_after_use`). Outputs:
  `(text, reasoning)`.
- **Stop llama.cpp server** — manually stop the server on a given port.

### Usage
1. **Edit `config.json` manually.** This is the most important step: open
  `config.json` in the plugin folder and set `model_roots` / `mmproj_roots` /
  `skill_roots` to the real directories on your machine (the defaults already
  include ComfyUI's `models/text_encoders` and `models/LLM`, plus an example
  path). The model dropdown only scans `.gguf` files inside these directories,
  so set them correctly to see models in the list; if empty or no match, you can
  still paste a full path manually.
2. Put this node into `ComfyUI/custom_nodes/` (skip if installed by manager or git clone or already done).
3. Add **Load llama.cpp Model** and set `server_exe` to your `llama-server.exe`,
   `model_path` to a GGUF model. Add `mmproj_path` only for multimodal models.
4. Add **LLM (llama.cpp server)**, connect the model, type a prompt. Optionally
   connect an image / audio / video / skill.
5. Queue a prompt. The server starts automatically (`auto_start`) and frees
   VRAM/RAM when done (`release_after_use`).
6. Models must be GGUF — `llama-server` cannot load safetensors directly.
