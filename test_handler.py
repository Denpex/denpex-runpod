import json
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = r"C:\Users\ohsoo\scratch\fleet\pubimg-1.1.36\agent"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.isdir(AGENT_DIR) and AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import handler  # noqa: E402


class HandlerTests(unittest.TestCase):
    def test_cuda_oom_offline_diagnosis(self):
        res = handler.handler({
            "input": {
                "action": "diagnose",
                "log": "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 23.69 GiB of which 120.00 MiB is free.",
            }
        })
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["mode"], "offline")
        self.assertEqual(res["cli_version"], "1.1.36")
        self.assertFalse(res["truncated"])
        self.assertIn("CUDA_OOM", str(res["diagnosis"].get("failureType", "")))

    def test_xid_79_hardware_diagnosis(self):
        res = handler.handler({
            "input": {
                "action": "diagnose",
                "log": "NVRM: Xid (PCI:0000:1b:00): 79, pid=1234, GPU has fallen off the bus.",
            }
        })
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["mode"], "offline")
        self.assertIn("79", json.dumps(res["diagnosis"]))

    def test_empty_job_defaults_to_health(self):
        for bad_job in ({}, {"input": {}}, {"input": None}, None):
            res = handler.handler(bad_job)
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["service"], "denpex-runpod")
            self.assertEqual(res["cli_version"], "1.1.36")

    def test_empty_log_returns_error(self):
        res = handler.handler({"input": {"action": "diagnose", "log": "   "}})
        self.assertEqual(res["status"], "error")

    def test_unknown_action_returns_error(self):
        res = handler.handler({"input": {"action": "rm -rf /"}})
        self.assertEqual(res["status"], "error")
        self.assertIn("Unsupported action", res["error"])

    def test_hf_mem_rejects_shell_metacharacters_and_traversal(self):
        for bad_model in ("", "meta-llama/test;id", "$(whoami)", "../etc/passwd", "-flag"):
            res = handler.handler({"input": {"action": "hf-mem", "model": bad_model}})
            self.assertEqual(res["status"], "error", f"Expected error for {bad_model!r}")

    def test_hf_mem_valid_model_invokes_without_shell(self):
        fake_proc = mock.Mock(returncode=0, stdout='{"estimated_vram_gb": 16}', stderr="")
        with mock.patch("handler.subprocess.run", return_value=fake_proc) as run_mock:
            res = handler.handler({"input": {"action": "hf-mem", "model": "meta-llama/Llama-3.1-8B-Instruct"}})
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["action"], "hf-mem")
        run_mock.assert_called_once()
        called_args, called_kwargs = run_mock.call_args
        self.assertNotIn("shell", called_kwargs)
        self.assertEqual(called_args[0][-2:], ["hf-mem", "meta-llama/Llama-3.1-8B-Instruct"])

    def test_offline_only_blocks_cloud_call_even_with_api_key(self):
        with mock.patch.dict(os.environ, {"DENPEX_OFFLINE_ONLY": "1"}):
            with mock.patch("handler.urllib.request.urlopen") as urlopen_mock:
                res = handler.handler({
                    "input": {
                        "action": "diagnose",
                        "api_key": "dpx_test_dummy_key",
                        "log": "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.",
                    }
                })
        urlopen_mock.assert_not_called()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["mode"], "offline")

    def test_runpod_hub_and_tests_json_valid(self):
        hub_path = os.path.join(HERE, ".runpod", "hub.json")
        tests_path = os.path.join(HERE, ".runpod", "tests.json")
        with open(hub_path, encoding="utf-8") as fh:
            hub = json.load(fh)
        with open(tests_path, encoding="utf-8") as fh:
            tests = json.load(fh)
        self.assertEqual(hub["type"], "serverless")
        self.assertIn("config", hub)
        self.assertGreaterEqual(len(tests["tests"]), 3)
        self.assertEqual(tests["config"]["gpuCount"], 1)
        for tc in tests["tests"]:
            res = handler.handler({"input": tc["input"]})
            self.assertEqual(res["status"], "ok", f"Test case {tc['name']} failed: {res}")


if __name__ == "__main__":
    unittest.main()
