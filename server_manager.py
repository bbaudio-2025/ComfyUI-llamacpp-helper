import atexit
import os
import subprocess
import time
import urllib.error
import urllib.request

_SERVERS = {}


def _health_ok(host, port):
    url = "http://%s:%d/health" % (host, port)
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _lora_scale_arg(value):
    """The scaling factor to hand over, normalised; anything unreadable is 1.0.

    A NEGATIVE scale is legal in llama.cpp (it pushes generation away from what
    the adapter learned), so it is passed through rather than clamped to 0.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _lora_args(path, scale=1.0):
    """The llama-server arguments that apply ONE LoRA adapter (or none: []).

    Both spellings are llama.cpp's own, and in both the path is a SEPARATE argv
    element - an adapter path on Windows starts with a drive letter and often
    carries spaces, so joining path and scale with a separator would be lossy:

        --lora FNAME                 (scaling 1.0)
        --lora-scaled FNAME SCALE    (any other scaling)

    Scale 1.0 uses the shorter spelling because that is exactly what it means;
    0.0 is a real setting too (the adapter is loaded but not applied).
    """
    p = str(path or "").strip().strip('"')
    if not p:
        return []
    s = _lora_scale_arg(scale)
    if abs(s - 1.0) < 1e-9:
        return ["--lora", p]
    # `%g` keeps 0.5 as '0.5' rather than '0.500000' - llama.cpp parses either
    # way, but the command line reads better
    return ["--lora-scaled", p, "%g" % s]


def _build_signature(exe, model, host, port, n_gpu_layers, context_size, extra_args, mmproj,
                     mmproj_gpu_offload, lora="", lora_scale=1.0):
    # The adapter is in the signature for the same reason -ngl is: /health says
    # nothing about the launch flags, so leaving it out would let a swapped
    # adapter keep answering from the previous one while the log stays true.
    return repr((exe, model, host, port, n_gpu_layers, context_size,
                 tuple(extra_args or []), mmproj, mmproj_gpu_offload,
                 str(lora or "").strip().strip('"'),
                 round(_lora_scale_arg(lora_scale), 6)))


def ensure_server(exe, model, host, port, n_gpu_layers, context_size, extra_args=None,
                  timeout=180, mmproj=None, mmproj_gpu_offload=True, alias="llamacpp-helper",
                  lora=None, lora_scale=1.0):
    sig = _build_signature(exe, model, host, port, n_gpu_layers, context_size,
                           extra_args, mmproj, mmproj_gpu_offload, lora, lora_scale)
    existing = _SERVERS.get(port)
    if existing is not None:
        if (existing.get("signature") == sig
                and existing.get("proc") is not None
                and existing["proc"].poll() is None
                and _health_ok(host, port)):
            return existing["proc"]
        _stop(port)

    if not os.path.exists(exe):
        raise FileNotFoundError("llama-server executable not found: %s" % exe)
    if not os.path.exists(model):
        raise FileNotFoundError("model not found: %s" % model)
    lora_path = str(lora or "").strip().strip('"')
    if lora_path and not os.path.exists(lora_path):
        raise FileNotFoundError("LoRA adapter not found: %s" % lora_path)

    cmd = [
        exe,
        "-m", model,
        "--host", host,
        "--port", str(port),
        "-ngl", str(n_gpu_layers),
        "-c", str(context_size),
        "--alias", alias,
    ]
    if mmproj:
        cmd += ["--mmproj", mmproj]
        if not mmproj_gpu_offload:
            cmd += ["--no-mmproj-offload"]
    cmd += _lora_args(lora_path, lora_scale)
    if extra_args:
        cmd += list(extra_args)

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_server.log")
    logf = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    _SERVERS[port] = {"proc": proc, "signature": sig}

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError("llama-server exited early on port %d. See llama_server.log" % port)
        if _health_ok(host, port):
            return proc
        time.sleep(1.0)
    raise TimeoutError("llama-server did not become healthy within %ds on port %d" % (timeout, port))


def _stop(port):
    entry = _SERVERS.pop(port, None)
    if entry is None:
        return
    proc = entry.get("proc")
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def stop_server(port):
    _stop(port)


def _atexit_cleanup():
    for port in list(_SERVERS.keys()):
        _stop(port)


atexit.register(_atexit_cleanup)
