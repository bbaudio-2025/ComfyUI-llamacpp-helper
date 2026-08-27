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


def _build_signature(exe, model, host, port, n_gpu_layers, context_size, extra_args, mmproj, mmproj_gpu_offload):
    return repr((exe, model, host, port, n_gpu_layers, context_size,
                 tuple(extra_args or []), mmproj, mmproj_gpu_offload))


def ensure_server(exe, model, host, port, n_gpu_layers, context_size, extra_args=None,
                  timeout=180, mmproj=None, mmproj_gpu_offload=True, alias="llamacpp-helper"):
    sig = _build_signature(exe, model, host, port, n_gpu_layers, context_size,
                           extra_args, mmproj, mmproj_gpu_offload)
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
