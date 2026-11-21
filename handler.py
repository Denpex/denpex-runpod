#!/usr/bin/env python3
"""Thin RunPod serverless wrapper for the public Denpex CLI and offline engine."""
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

# Must be set before importing denpex_local so local_diagnose uses the canonical engine.
os.environ.setdefault("DENPEX_LOCAL", "1")

import denpex_local  # noqa: E402

CLI_VERSION = "1.1.36"
CLOUD_DIAGNOSE_URL = "https://api.denpex.com/api/diagnose"
MAX_LOG_CHARS = 256000
SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9._/-]{1,160}$")
SUPPORTED_ACTIONS = ("diagnose", "preflight", "hf-mem", "lstopo", "gpud", "health")
TRIAL_HINT = (
    "Start a free 30-day card-free Team trial for cloud deep diagnosis and all 14 fleet tools: "
    "run `denpex trial --email you@company.com` or visit https://denpex.com/pricing?utm_source=runpod_hub"
)


def _is_truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _resolve_denpex_cmd():
    exe = shutil.which("denpex")
    if exe:
        return [exe]
    try:
        import denpex as denpex_mod

        return [sys.executable, denpex_mod.__file__]
    except Exception:
        return [sys.executable, "-m", "denpex"]


def _call_cloud_diagnose(log_text, api_key):
    """Send log text to Denpex Cloud when an API key is provided and offline mode is off."""
    payload = json.dumps({
        "logs": log_text,
        "source": "runpod-hub",
        "jobName": "runpod-hub",
    }).encode("utf-8")
    req = urllib.request.Request(
        CLOUD_DIAGNOSE_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Denpex-Key": api_key,
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"Denpex-RunPod/{CLI_VERSION}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            if resp.status != 200:
                return None, f"HTTP {resp.status}"
            data = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
            if not isinstance(data, dict):
                return None, "Unexpected response format"
            if data.get("success") is False or (data.get("error") and not data.get("fix")):
                return None, str(data.get("error") or data.get("guidance") or "Cloud diagnosis declined")[:200]
            return data, None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except Exception as exc:
        return None, f"Cloud unreachable: {type(exc).__name__}"


def _handle_diagnose(job_input):
    raw_log = job_input.get("log") if job_input.get("log") is not None else job_input.get("logs")
    if raw_log is None:
        log_text = ""
    elif isinstance(raw_log, str):
        log_text = raw_log
    else:
        log_text = str(raw_log)

    if not log_text.strip():
        return {"status": "error", "error": "Provide a non-empty 'log' string to diagnose."}

    truncated = len(log_text) > MAX_LOG_CHARS
    if truncated:
        log_text = log_text[-MAX_LOG_CHARS:]

    offline_only = _is_truthy(os.environ.get("DENPEX_OFFLINE_ONLY")) or job_input.get("offline_only") is True
    api_key = str(job_input.get("api_key") or os.environ.get("DENPEX_API_KEY") or "").strip()
    allow_cloud = job_input.get("allow_cloud", True) is not False

    fallback_reason = None
    if api_key and not offline_only and allow_cloud:
        cloud_data, err = _call_cloud_diagnose(log_text, api_key)
        if cloud_data is not None:
            return {
                "status": "ok",
                "mode": "cloud",
                "cli_version": CLI_VERSION,
                "truncated": truncated,
                "diagnosis": cloud_data,
            }
        fallback_reason = err

    diag = denpex_local.local_diagnose(log_text)
    result = {
        "status": "ok",
        "mode": "offline",
        "cli_version": CLI_VERSION,
        "diagnosis": diag,
        "truncated": truncated,
        "trial_hint": TRIAL_HINT,
    }
    if fallback_reason:
        result["cloud_fallback_reason"] = fallback_reason
    return result


def _handle_cli_action(action, job_input):
    if action == "preflight":
        args = ["preflight", "--json"]
    elif action == "hf-mem":
        model = str(job_input.get("model") or "").strip()
        if (
            not model
            or not SAFE_MODEL_RE.fullmatch(model)
            or model.startswith("-")
            or ".." in model.split("/")
        ):
            return {
                "status": "error",
                "error": "Provide a valid Hugging Face 'model' ID (for example 'meta-llama/Llama-3.1-8B-Instruct').",
            }
        args = ["hf-mem", model]
    elif action == "lstopo":
        args = ["lstopo"]
    elif action == "gpud":
        args = ["gpud"]
    else:
        return {
            "status": "error",
            "error": f"Unsupported action '{action}'. Use one of: {', '.join(SUPPORTED_ACTIONS)}.",
        }

    cmd = _resolve_denpex_cmd() + args
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "action": action, "error": f"Command timed out after 30s."}
    except Exception as exc:
        return {"status": "error", "action": action, "error": f"Failed to execute {action}: {type(exc).__name__}"}

    return {
        "status": "ok" if proc.returncode == 0 else "warning",
        "action": action,
        "exit_code": proc.returncode,
        "stdout": (proc.stdout or "")[:32000],
        "stderr": (proc.stderr or "")[:8000],
    }


def _query_gpus():
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return []
    try:
        proc = subprocess.run(
            [nvsmi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return []
        return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    except Exception:
        return []


def _handle_health():
    return {
        "status": "ok",
        "service": "denpex-runpod",
        "cli_version": CLI_VERSION,
        "offline_engine_ready": bool(shutil.which("node")),
        "gpus": _query_gpus(),
        "supported_actions": list(SUPPORTED_ACTIONS),
        "docs": "https://denpex.com/docs/integrations",
        "trial": "https://denpex.com/pricing?utm_source=runpod_hub",
    }


def handler(job):
    """RunPod serverless handler entry point."""
    job_input = job.get("input") if isinstance(job, dict) else None
    if not isinstance(job_input, dict):
        job_input = {}

    raw_action = job_input.get("action")
    if raw_action is not None and str(raw_action).strip():
        action = str(raw_action).strip().lower()
    elif job_input.get("log") or job_input.get("logs"):
        action = "diagnose"
    else:
        action = "health"

    if action == "diagnose":
        return _handle_diagnose(job_input)
    if action == "health":
        return _handle_health()
    if action in ("preflight", "hf-mem", "lstopo", "gpud"):
        return _handle_cli_action(action, job_input)

    return {
        "status": "error",
        "error": f"Unsupported action '{action}'. Use one of: {', '.join(SUPPORTED_ACTIONS)}.",
    }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
