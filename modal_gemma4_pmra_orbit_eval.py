from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal


APP_NAME = "orbitquant-gemma4-pmra-stack"
VOLUME_NAME = "codex-jmrc-cache"
RESULT_ROOT = "/cache/results/orbitquant_gemma4_pmra_stack"
REMOTE_ORBIT = "/workspace/orbitquant"
REMOTE_PMRA = "/workspace/pmra"

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_DIR = "/cache/models/gemma4-e2b-it"
PMRA_RESULT_JSON = f"{REMOTE_PMRA}/results/gemma4_e2b_it/selector_result_knapsack.json"
TRIMMED_NO14_15_MLP_CHOICES = (
    "19:plus:preperm_activation_max_hadamard:0.375,"
    "20:plus:preperm_activation_max_hadamard:0.375,"
    "18:plain:block_hadamard:0.375,"
    "6:plus:preperm_boundary_rms_hadamard:0.375,"
    "16:plain:block_hadamard:0.375"
)
SAFE3_MLP_CHOICES = (
    "20:plus:preperm_activation_max_hadamard:0.375,"
    "19:plus:preperm_activation_max_hadamard:0.375,"
    "6:plus:preperm_boundary_rms_hadamard:0.375"
)
MLP20_CHOICES = "20:plus:preperm_activation_max_hadamard:0.375"

LOCAL_PMRA_ROOT = Path("C:/Users/power/Documents/PMRA")
LOCAL_PMRA_SCRIPTS = LOCAL_PMRA_ROOT / "scripts"
LOCAL_PMRA_RESULTS = LOCAL_PMRA_ROOT / "results" / "gemma4_e2b_it"

GGUF_REPO = "mradermacher/gemma-4-E2B-it-GGUF"
GGUF_FILES = {
    "q2_k": "gemma-4-E2B-it.Q2_K.gguf",
    "q3_k_s": "gemma-4-E2B-it.Q3_K_S.gguf",
    "q3_k_m": "gemma-4-E2B-it.Q3_K_M.gguf",
    "q3_k_l": "gemma-4-E2B-it.Q3_K_L.gguf",
    "iq4_xs": "gemma-4-E2B-it.IQ4_XS.gguf",
    "q4_k_m": "gemma-4-E2B-it.Q4_K_M.gguf",
}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "accelerate",
        "datasets",
        "gguf",
        "huggingface_hub",
        "numpy",
        "safetensors",
        "sentencepiece",
        "torch",
        "transformers==5.5.0",
    )
    .add_local_dir("scripts", remote_path=f"{REMOTE_ORBIT}/scripts")
    .add_local_dir(str(LOCAL_PMRA_SCRIPTS), remote_path=f"{REMOTE_PMRA}/scripts")
    .add_local_dir(str(LOCAL_PMRA_RESULTS), remote_path=f"{REMOTE_PMRA}/results/gemma4_e2b_it")
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def profile_args(profile: str) -> dict:
    profiles = {
        "smoke": {
            "variants": "fp16,q3_k_s,pmra,pmra_kv_only,pmra_mlp_only,pmra_orbitquant",
            "prompt_count": 16,
            "calib_prompt_count": 8,
            "eval_max_length": 96,
            "calib_max_length": 96,
            "memory_context_length": 8192,
        },
        "stack64": {
            "variants": "fp16,q3_k_s,orbitquant,pmra,pmra_orbitquant",
            "prompt_count": 64,
            "calib_prompt_count": 16,
            "eval_max_length": 128,
            "calib_max_length": 128,
            "memory_context_length": 8192,
        },
        "split64": {
            "variants": "fp16,q3_k_s,pmra,pmra_kv_only,pmra_mlp_only,pmra_orbitquant",
            "prompt_count": 64,
            "calib_prompt_count": 16,
            "eval_max_length": 128,
            "calib_max_length": 128,
            "memory_context_length": 8192,
        },
        "trim64_no14_15": {
            "variants": "fp16,q3_k_s,pmra,pmra_kv_only,pmra_mlp_only,pmra_orbitquant",
            "prompt_count": 64,
            "calib_prompt_count": 16,
            "eval_max_length": 128,
            "calib_max_length": 128,
            "memory_context_length": 8192,
            "mlp_choices": TRIMMED_NO14_15_MLP_CHOICES,
        },
        "trim64_safe3": {
            "variants": "fp16,q3_k_s,pmra,pmra_kv_only,pmra_mlp_only,pmra_orbitquant",
            "prompt_count": 64,
            "calib_prompt_count": 16,
            "eval_max_length": 128,
            "calib_max_length": 128,
            "memory_context_length": 8192,
            "mlp_choices": SAFE3_MLP_CHOICES,
        },
        "trim64_mlp20": {
            "variants": "fp16,q3_k_s,pmra,pmra_kv_only,pmra_mlp_only,pmra_orbitquant",
            "prompt_count": 64,
            "calib_prompt_count": 16,
            "eval_max_length": 128,
            "calib_max_length": 128,
            "memory_context_length": 8192,
            "mlp_choices": MLP20_CHOICES,
        },
        "trim128_safe3": {
            "variants": "fp16,q3_k_s,pmra,pmra_kv_only,pmra_mlp_only,pmra_orbitquant",
            "prompt_count": 128,
            "calib_prompt_count": 24,
            "eval_max_length": 192,
            "calib_max_length": 192,
            "memory_context_length": 8192,
            "mlp_choices": SAFE3_MLP_CHOICES,
        },
        "trim128_safe3_folded": {
            "variants": "fp16,q3_k_s,pmra,pmra_kv_only,pmra_mlp_only,pmra_orbitquant",
            "prompt_count": 128,
            "calib_prompt_count": 24,
            "eval_max_length": 192,
            "calib_max_length": 192,
            "memory_context_length": 8192,
            "mlp_choices": SAFE3_MLP_CHOICES,
            "mlp_fold_down_proj": True,
        },
        "stack128": {
            "variants": "fp16,q3_k_s,orbitquant,pmra,pmra_orbitquant",
            "prompt_count": 128,
            "calib_prompt_count": 24,
            "eval_max_length": 192,
            "calib_max_length": 192,
            "memory_context_length": 8192,
        },
    }
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(profiles)}")
    return profiles[profile]


def _run(cmd: list[str], cwd: str = REMOTE_ORBIT) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_ORBIT}/scripts:{REMOTE_PMRA}/scripts:" + env.get("PYTHONPATH", "")
    env["PMRA_SCRIPTS"] = f"{REMOTE_PMRA}/scripts"
    print("[orbit-modal] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    stdout_tail = ""
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="", flush=True)
            stdout_tail = (stdout_tail + line)[-16000:]
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, output=stdout_tail)
    return stdout_tail


def _ensure_model_snapshot() -> str:
    from huggingface_hub import snapshot_download

    model_dir = Path(MODEL_DIR)
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "model.safetensors.index.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) == 1:
        return str(shards[0])

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer*",
            "*.model",
            "*.tiktoken",
            "merges.txt",
            "vocab.json",
        ],
    )
    cache_volume.commit()
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) == 1:
        return str(shards[0])
    raise FileNotFoundError(f"could not find a safetensors file or index under {MODEL_DIR}")


def _ensure_sources(labels: set[str]) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    out = {}
    base_dir = Path("/cache/models/baselines") / GGUF_REPO.replace("/", "__")
    base_dir.mkdir(parents=True, exist_ok=True)
    for label in sorted(labels):
        filename = GGUF_FILES[label]
        path = base_dir / filename
        if not path.exists():
            hf_hub_download(repo_id=GGUF_REPO, filename=filename, local_dir=str(base_dir))
            cache_volume.commit()
        out[label] = str(path)
    return out


def _read_result(output_dir: str) -> dict:
    result_path = Path(output_dir) / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "output_dir": output_dir,
        "variants": result.get("variants", {}),
        "baseline": result.get("baseline", {}),
        "rows": result.get("rows", []),
        "rows_by_delta_desc": result.get("rows_by_delta_desc", []),
        "rows_by_delta_asc": result.get("rows_by_delta_asc", []),
        "runtime_savings_estimate": result.get("runtime_savings_estimate", {}),
        "wall_time_seconds": result.get("wall_time_seconds"),
        "result_path": str(result_path),
    }


def _base_eval_cmd(script_name: str, output_dir: str, settings: dict, hf_file: str, source_paths: dict[str, str]) -> list[str]:
    cmd = [
        sys.executable,
        f"{REMOTE_ORBIT}/scripts/{script_name}",
        "--model-dir",
        MODEL_DIR,
        "--hf",
        hf_file,
        "--result-json",
        PMRA_RESULT_JSON,
        "--output-dir",
        output_dir,
        "--pmra-variant",
        "c2_calib_knapsack_mixed",
        "--prompt-count",
        str(settings["prompt_count"]),
        "--calib-prompt-count",
        str(settings["calib_prompt_count"]),
        "--eval-max-length",
        str(settings["eval_max_length"]),
        "--calib-max-length",
        str(settings["calib_max_length"]),
        "--device",
        "cuda",
    ]
    for label, path in source_paths.items():
        cmd.extend(["--source", f"{label}={path}"])
    for setting_name, flag in (
        ("kv_layers", "--kv-layers"),
        ("kv_bits", "--kv-bits"),
        ("kv_rotation", "--kv-rotation"),
        ("kv_alpha", "--kv-alpha"),
        ("mlp_choices", "--mlp-choices"),
        ("mlp_bits", "--mlp-bits"),
        ("mlp_alpha", "--mlp-alpha"),
        ("mlp_block_size", "--mlp-block-size"),
    ):
        if setting_name in settings:
            cmd.extend([flag, str(settings[setting_name])])
    if settings.get("mlp_fold_down_proj"):
        cmd.append("--mlp-fold-down-proj")
    return cmd


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 4,
)
def run_stack_eval(profile: str = "stack64", extra_args: list[str] | None = None) -> dict:
    settings = profile_args(profile)
    hf_file = _ensure_model_snapshot()
    source_paths = _ensure_sources(set(GGUF_FILES))
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = f"{RESULT_ROOT}/{profile}_{stamp}"

    cmd = _base_eval_cmd(
        "gemma4_pmra_orbit_stack_eval.py",
        output_dir,
        settings,
        hf_file,
        source_paths,
    )
    cmd.extend(
        [
            "--variants",
            settings["variants"],
            "--orbit-base-source",
            "q3_k_s",
            "--memory-context-length",
            str(settings["memory_context_length"]),
        ]
    )
    if extra_args:
        cmd.extend(extra_args)

    stdout_tail = _run(cmd)
    cache_volume.commit()
    result = _read_result(output_dir)
    result.update(
        {
            "profile": profile,
            "stdout_tail": stdout_tail,
            "persisted_volume": VOLUME_NAME,
        }
    )
    persisted_path = Path(RESULT_ROOT) / f"modal_{profile}_{stamp}.json"
    persisted_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cache_volume.commit()
    result["persisted_result_path"] = str(persisted_path)
    return result


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 4,
)
def run_layer_sweep(profile: str = "split64", sides: str = "kv,mlp", extra_args: list[str] | None = None) -> dict:
    settings = profile_args(profile)
    hf_file = _ensure_model_snapshot()
    source_paths = _ensure_sources(set(GGUF_FILES))
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    side_label = sides.replace(",", "_")
    output_dir = f"{RESULT_ROOT}/layer_{side_label}_{profile}_{stamp}"
    cmd = _base_eval_cmd(
        "gemma4_pmra_orbit_layer_sweep.py",
        output_dir,
        settings,
        hf_file,
        source_paths,
    )
    cmd.extend(["--sides", sides])
    if extra_args:
        cmd.extend(extra_args)

    stdout_tail = _run(cmd)
    cache_volume.commit()
    result = _read_result(output_dir)
    result.update(
        {
            "profile": profile,
            "sides": sides,
            "stdout_tail": stdout_tail,
            "persisted_volume": VOLUME_NAME,
        }
    )
    persisted_path = Path(RESULT_ROOT) / f"modal_layer_{side_label}_{profile}_{stamp}.json"
    persisted_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cache_volume.commit()
    result["persisted_result_path"] = str(persisted_path)
    return result


@app.function(image=image, volumes={"/cache": cache_volume})
def read_latest_persisted(profile: str = "stack64") -> str:
    paths = sorted(
        glob.glob(f"{RESULT_ROOT}/modal_{profile}_*.json"),
        key=lambda path: Path(path).stat().st_mtime,
    )
    if not paths:
        return ""
    return Path(paths[-1]).read_text(encoding="utf-8")


@app.function(image=image, volumes={"/cache": cache_volume})
def read_latest_layer_persisted(profile: str = "split64", sides: str = "kv,mlp") -> str:
    side_label = sides.replace(",", "_")
    paths = sorted(
        glob.glob(f"{RESULT_ROOT}/modal_layer_{side_label}_{profile}_*.json"),
        key=lambda path: Path(path).stat().st_mtime,
    )
    if not paths:
        return ""
    return Path(paths[-1]).read_text(encoding="utf-8")


@app.local_entrypoint()
def main(profile: str = "stack64", background: bool = False):
    if background:
        call = run_stack_eval.spawn(profile=profile)
        out_dir = Path("results") / "modal_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        launch_info = {
            "profile": profile,
            "function_call_id": call.object_id,
            "dashboard_url": call.get_dashboard_url(),
            "persisted_volume": VOLUME_NAME,
            "persisted_pattern": f"{RESULT_ROOT}/modal_{profile}_*.json",
            "launched_at": datetime.now().isoformat(timespec="seconds"),
        }
        launch_path = out_dir / f"gemma4-stack-{profile}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.spawn.json"
        launch_path.write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
        print(json.dumps(launch_info, indent=2))
        print(f"Wrote {launch_path}")
        return json.dumps(launch_info)

    result = run_stack_eval.remote(profile=profile)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    local_path = out_dir / f"modal_gemma4_pmra_orbit_stack_{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    local_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {local_path}")
    return str(local_path)


@app.local_entrypoint()
def fetch_latest(profile: str = "stack64"):
    payload = read_latest_persisted.remote(profile=profile)
    if not payload:
        print("")
        return ""
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    local_path = out_dir / f"modal_gemma4_pmra_orbit_stack_{profile}_latest.json"
    local_path.write_text(payload, encoding="utf-8")
    print(payload)
    print(f"Wrote {local_path}")
    return str(local_path)


@app.local_entrypoint()
def layer_sweep(profile: str = "split64", sides: str = "kv,mlp", background: bool = False):
    if background:
        call = run_layer_sweep.spawn(profile=profile, sides=sides)
        out_dir = Path("results") / "modal_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        side_label = sides.replace(",", "_")
        launch_info = {
            "profile": profile,
            "sides": sides,
            "function_call_id": call.object_id,
            "dashboard_url": call.get_dashboard_url(),
            "persisted_volume": VOLUME_NAME,
            "persisted_pattern": f"{RESULT_ROOT}/modal_layer_{side_label}_{profile}_*.json",
            "launched_at": datetime.now().isoformat(timespec="seconds"),
        }
        launch_path = out_dir / f"gemma4-layer-{side_label}-{profile}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.spawn.json"
        launch_path.write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
        print(json.dumps(launch_info, indent=2))
        print(f"Wrote {launch_path}")
        return json.dumps(launch_info)

    result = run_layer_sweep.remote(profile=profile, sides=sides)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    side_label = sides.replace(",", "_")
    local_path = out_dir / f"modal_gemma4_pmra_orbit_layer_{side_label}_{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    local_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {local_path}")
    return str(local_path)


@app.local_entrypoint()
def fetch_latest_layer(profile: str = "split64", sides: str = "kv,mlp"):
    payload = read_latest_layer_persisted.remote(profile=profile, sides=sides)
    if not payload:
        print("")
        return ""
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    side_label = sides.replace(",", "_")
    local_path = out_dir / f"modal_gemma4_pmra_orbit_layer_{side_label}_{profile}_latest.json"
    local_path.write_text(payload, encoding="utf-8")
    print(payload)
    print(f"Wrote {local_path}")
    return str(local_path)
