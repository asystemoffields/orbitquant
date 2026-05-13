import glob
import json
import subprocess
from datetime import datetime
from pathlib import Path

import modal


APP_NAME = "smol135-fused-policy-search"
REMOTE_WORKDIR = "/root/workspace"
RESULT_VOLUME_DIR = "/root/modal-results"


image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", add_python="3.11")
    .pip_install(
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "sentencepiece",
        "transformers",
    )
    .add_local_dir(".", remote_path=REMOTE_WORKDIR)
)

app = modal.App(APP_NAME)
result_volume = modal.Volume.from_name("smol135-fused-policy-results", create_if_missing=True)


def default_args(profile: str) -> list[str]:
    base = [
        "scripts/smol135_fused_policy_search.py",
        "--kv-impact-result",
        "results/smol135_kv_impact_3bit_20260513_084903.json",
        "--local-result",
        "results/smol135_mlp_hadamard_plus_2bit_20260513_131208.json",
        "--mlp-choice-impact-result",
        "results/smol135_mlp_hadamard_plus_impact_2bit_20260513_132208.json",
        "--base-kv-layers",
        "28,24,26,14",
        "--base-mlp-choices",
        "16:plain,17:plain,15:plain,12:plain",
        "--kv-bits",
        "3",
        "--kv-alpha",
        "0.75",
        "--mlp-bits",
        "2",
        "--block-size",
        "512",
        "--calibration-prompts",
        "prompts/broad_calibration.txt",
        "--eval-prompts",
        "prompts/broad_eval.txt",
        "--max-length",
        "64",
        "--device",
        "auto",
    ]
    profiles = {
        "smoke": [
            "--kv-candidate-count",
            "3",
            "--mlp-candidate-count",
            "3",
            "--max-extra-buses",
            "2",
            "--beam-width-per-shape",
            "1",
            "--global-beam-cap",
            "4",
            "--swap-rounds",
            "1",
            "--eval-prompt-limit",
            "16",
            "--calibration-prompt-limit",
            "16",
            "--max-evals",
            "80",
        ],
        "quality16": [
            "--kv-candidate-count",
            "8",
            "--mlp-candidate-count",
            "8",
            "--max-extra-buses",
            "8",
            "--beam-width-per-shape",
            "2",
            "--global-beam-cap",
            "24",
            "--swap-rounds",
            "2",
            "--objective",
            "kl",
        ],
        "quality20": [
            "--kv-candidate-count",
            "10",
            "--mlp-candidate-count",
            "10",
            "--max-extra-buses",
            "12",
            "--beam-width-per-shape",
            "2",
            "--global-beam-cap",
            "32",
            "--swap-rounds",
            "2",
            "--objective",
            "kl",
        ],
        "memory20": [
            "--kv-candidate-count",
            "10",
            "--mlp-candidate-count",
            "10",
            "--max-extra-buses",
            "12",
            "--beam-width-per-shape",
            "2",
            "--global-beam-cap",
            "32",
            "--swap-rounds",
            "2",
            "--objective",
            "kl_per_saved_mb",
            "--memory-context-length",
            "2048",
            "--mlp-lifetime-tokens",
            "64",
        ],
    }
    if profile not in profiles:
        raise ValueError(f"Unknown profile {profile!r}. Choose one of {sorted(profiles)}")
    return base + profiles[profile]


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    cpu=8,
    memory=32768,
    volumes={RESULT_VOLUME_DIR: result_volume},
)
def run_remote(profile: str = "quality16", extra_args: list[str] | None = None) -> dict:
    args = default_args(profile)
    if extra_args:
        args.extend(extra_args)

    before = set(glob.glob(f"{REMOTE_WORKDIR}/results/smol135_fused_policy_search_*.json"))
    proc = subprocess.Popen(
        ["python", *args],
        cwd=REMOTE_WORKDIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    stdout_tail = ""
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="", flush=True)
            stdout_tail = (stdout_tail + line)[-12000:]
    returncode = proc.wait()
    after = set(glob.glob(f"{REMOTE_WORKDIR}/results/smol135_fused_policy_search_*.json"))
    new_files = sorted(after - before, key=lambda path: Path(path).stat().st_mtime)
    result_path = new_files[-1] if new_files else None
    payload = json.loads(Path(result_path).read_text(encoding="utf-8")) if result_path else None

    result = {
        "profile": profile,
        "returncode": returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": "",
        "remote_result_path": result_path,
        "payload": payload,
        "args": args,
    }
    stem = Path(result_path).name if result_path else f"smol135_fused_policy_search_{profile}.json"
    persisted_path = Path(RESULT_VOLUME_DIR) / f"modal_{profile}_{stem}"
    result["persisted_result_path"] = str(persisted_path)
    persisted_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result_volume.commit()
    return result


@app.function(image=image, volumes={RESULT_VOLUME_DIR: result_volume})
def read_latest_persisted(profile: str = "quality16") -> str:
    paths = sorted(
        Path(RESULT_VOLUME_DIR).glob(f"modal_{profile}_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        return ""
    return paths[-1].read_text(encoding="utf-8")


@app.local_entrypoint()
def main(profile: str = "quality16", background: bool = False):
    if background:
        call = run_remote.spawn(profile=profile)
        out_dir = Path("results") / "modal_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        launch_info = {
            "profile": profile,
            "function_call_id": call.object_id,
            "dashboard_url": call.get_dashboard_url(),
            "persisted_volume": "smol135-fused-policy-results",
            "persisted_pattern": f"{RESULT_VOLUME_DIR}/modal_{profile}_*.json",
            "launched_at": datetime.now().isoformat(timespec="seconds"),
        }
        launch_path = out_dir / f"fused-{profile}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.spawn.json"
        launch_path.write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
        print(json.dumps(launch_info, indent=2))
        print(f"Wrote {launch_path}")
        return json.dumps(launch_info)

    result = run_remote.remote(profile=profile)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    stamp = result["remote_result_path"]
    if stamp:
        stem = Path(stamp).name
    else:
        stem = f"modal_fused_policy_search_{profile}.json"
    local_path = out_dir / f"modal_{stem}"
    local_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["stdout_tail"])
    if result["stderr_tail"]:
        print(result["stderr_tail"])
    print(f"Wrote {local_path}")
    print(f"Persisted remote copy at {result['persisted_result_path']}")
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
    return str(local_path)
