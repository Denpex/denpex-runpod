# Denpex GPU Crash & NCCL Diagnostics for RunPod Hub

RunPod serverless worker wrapping the public [`denpex`](https://pypi.org/project/denpex/) CLI (`1.1.36`) and deterministic offline engine so teams on RunPod can triage CUDA out-of-memory crashes, NCCL hangs, vLLM KV-cache failures, and NVIDIA Xid hardware faults without uploading logs unless they explicitly supply a Denpex API key.

## Supported Actions

Pass `"action"` inside `input`:

| Action | Description | Example Input |
|---|---|---|
| `diagnose` | Offline canonical diagnosis (default when `log` is provided). Uses Denpex Cloud when `DENPEX_API_KEY` or `input.api_key` is set and `DENPEX_OFFLINE_ONLY` is not `1`. | `{"action": "diagnose", "log": "RuntimeError: CUDA out of memory..."}` |
| `preflight` | Run `denpex preflight --json` environment and fabric checks. | `{"action": "preflight"}` |
| `hf-mem` | Estimate Hugging Face model GPU memory and KV-cache requirements. | `{"action": "hf-mem", "model": "meta-llama/Llama-3.1-8B-Instruct"}` |
| `lstopo` | Inspect PCIe / NUMA topology via `denpex lstopo`. | `{"action": "lstopo"}` |
| `gpud` | Run `denpex gpud` host health checks. | `{"action": "gpud"}` |
| `health` | Check worker readiness, bundled Node.js runtime, and visible GPUs. | `{"action": "health"}` |

## Zero-Egress by Default

- By default, all diagnosis happens locally inside your RunPod container using the bundled deterministic engine (`DENPEX_LOCAL=1`).
- Set `DENPEX_OFFLINE_ONLY=1` to strictly forbid outbound API calls even if a key is supplied in input.
- To enable cloud deep diagnosis and all 14 GPU fleet tools, start a free 30-day card-free Team trial (`denpex trial --email you@company.com`) or visit [denpex.com/pricing](https://denpex.com/pricing?utm_source=runpod_hub).

## Links

- Documentation: https://denpex.com/docs/integrations
- PyPI CLI: https://pypi.org/project/denpex/
- Plans & Free Team Trial: https://denpex.com/pricing?utm_source=runpod_hub
