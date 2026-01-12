import json
import os
import subprocess
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

# ---------------------------
# Project / runtime config
# ---------------------------

PROJECT_DIR = (
    Path(os.environ.get("AGENT_PROJECT_DIR", "")).expanduser().resolve()
    if os.environ.get("AGENT_PROJECT_DIR")
    else Path(__file__).resolve().parent
)
TOOLS_DIR = PROJECT_DIR / "tools"

JULIA = os.environ.get("JULIA", "julia")
PYTHON = os.environ.get("PYTHON", sys.executable)

SSH_BAND_JL = TOOLS_DIR / "ssh_band_tool.jl"
KITEAV_BAND_JL = TOOLS_DIR / "kitaev_band_tool.jl"
AA_HARPER_JL = TOOLS_DIR / "aa_harper_tool.jl"
CLEAN_OUT_PY = TOOLS_DIR / "clean_out.py"


DEFAULT_OUT_DIR = "out"

API_KEY = (os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("OPENAI_MODEL", "deepseek-chat")

DEBUG_TOOL_CALLS = os.environ.get("AGENT_DEBUG_TOOL_CALLS", "").strip().lower() in {"1", "true", "yes", "y"}


# ---------------------------
# Generic subprocess helpers
# ---------------------------

def _run_subprocess_json_stdin(cmd: list[str], cwd: Path, payload: dict) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        text=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, stdout, stderr


def _parse_last_json_from_stdout(stdout: str) -> dict | None:
    s = stdout.strip()
    j = s.rfind("{")
    if j == -1:
        return None
    try:
        obj = json.loads(s[j:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def run_julia_json_tool(script_path: Path, payload: dict) -> dict:
    if not script_path.exists():
        raise FileNotFoundError(f"Julia tool script not found: {script_path}")

    rc, stdout, stderr = _run_subprocess_json_stdin(
        [JULIA, "--project=.", str(script_path)],
        cwd=PROJECT_DIR,
        payload=payload,
    )

    if rc != 0:
        raise RuntimeError(
            "Julia tool failed.\n"
            f"CMD: {JULIA} --project=. {script_path}\n"
            f"STDERR:\n{stderr}\n"
            f"STDOUT:\n{stdout}\n"
        )

    out = _parse_last_json_from_stdout(stdout)
    if out is None:
        raise RuntimeError(
            "Julia tool did not return JSON.\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}\n"
        )
    return out


def run_python_best_effort_tool(script_path: Path, payload: dict) -> dict:
    if not script_path.exists():
        raise FileNotFoundError(f"Python tool script not found: {script_path}")

    rc, stdout, stderr = _run_subprocess_json_stdin(
        [PYTHON, str(script_path)],
        cwd=PROJECT_DIR,
        payload=payload,
    )

    if rc != 0:
        raise RuntimeError(
            "Python tool failed.\n"
            f"CMD: {PYTHON} {script_path}\n"
            f"STDERR:\n{stderr}\n"
            f"STDOUT:\n{stdout}\n"
        )

    out = _parse_last_json_from_stdout(stdout)
    if out is not None:
        return out

    return {
        "ok": True,
        "tool": str(script_path),
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
    }


# ---------------------------
# Tool executors (local)
# ---------------------------

def _norm_boundary(x: Any) -> str:
    if x is None:
        return "pbc"
    s = str(x).strip().lower()
    if s in {"pbc", "periodic"}:
        return "pbc"
    if s in {"obc", "open"}:
        return "obc"
    raise ValueError(f"Invalid boundary: {x!r}. Must be 'pbc' or 'obc'.")


def exec_ssh_band(args: dict) -> dict:
    if "p" not in args:
        raise ValueError("Missing required argument: p")

    boundary = _norm_boundary(args.get("boundary", "pbc"))

    # Julia tool supports these keys; we forward them 1:1.
    payload: dict[str, Any] = {
        "boundary": boundary,
        "p": float(args["p"]),
        "b": float(args.get("b", 0.0)),
        "n_k": int(args.get("n_k", 301)),
        "N": int(args.get("N", 80)),  # used in OBC branch; harmless for PBC
        "overlay_pbc": bool(args.get("overlay_pbc", False)),
        "out_dir": str(args.get("out_dir", DEFAULT_OUT_DIR)),
        "overwrite": bool(args.get("overwrite", False)),
    }

    prefix = args.get("prefix", None)
    if prefix is not None and str(prefix).strip():
        payload["prefix"] = str(prefix)

    # Light sanity: if OBC and user explicitly requested overlay, keep n_k meaningful
    if boundary == "obc":
        payload["N"] = max(int(payload["N"]), 2)
        payload["n_k"] = max(int(payload["n_k"]), 5)

    return run_julia_json_tool(SSH_BAND_JL, payload)


def exec_kitaev_band(args: dict) -> dict:
    # Required
    if "mu" not in args and "μ" not in args:
        raise ValueError("Missing required argument: mu")

    boundary = _norm_boundary(args.get("boundary", "pbc"))

    payload: dict[str, Any] = {
        "boundary": boundary,
        "t": float(args.get("t", 1.0)),
        "mu": float(args.get("mu", args.get("μ"))),
        "delta": float(args.get("delta", args.get("Δ", 1.0))),
        "n_k": int(args.get("n_k", 401)),
        "N": int(args.get("N", 80)),  # used in OBC; harmless for PBC
        "overlay_pbc": bool(args.get("overlay_pbc", False)),
        "out_dir": str(args.get("out_dir", DEFAULT_OUT_DIR)),
        "overwrite": bool(args.get("overwrite", False)),
    }

    prefix = args.get("prefix", None)
    if prefix is not None and str(prefix).strip():
        payload["prefix"] = str(prefix)

    if boundary == "obc":
        payload["N"] = max(int(payload["N"]), 2)
        payload["n_k"] = max(int(payload["n_k"]), 5)

    return run_julia_json_tool(KITEAV_BAND_JL, payload)

def exec_aah_harper(args: dict) -> dict:
    # Required
    if "lambda" not in args and "λ" not in args:
        raise ValueError("Missing required argument: lambda")

    boundary = _norm_boundary(args.get("boundary", "obc"))

    # Optional: pbc_mode ('bloch' or 'ring') for boundary='pbc'
    pbc_mode = args.get("pbc_mode", args.get("mode", None))
    if pbc_mode is not None:
        pbc_mode = str(pbc_mode).strip().lower()
        if not pbc_mode:
            pbc_mode = None

    payload: dict[str, Any] = {
        "boundary": boundary,
        "pbc_mode": pbc_mode,  # may be None; Julia side will auto-pick if omitted
        "t": float(args.get("t", 1.0)),
        "lambda": float(args.get("lambda", args.get("λ"))),
        "phi": float(args.get("phi", 0.0)),
        "n_k": int(args.get("n_k", 401)),
        "N": int(args.get("N", 200)),
        "out_dir": str(args.get("out_dir", DEFAULT_OUT_DIR)),
        "overwrite": bool(args.get("overwrite", False)),
    }

    if payload["pbc_mode"] is None:
        payload.pop("pbc_mode", None)
    else:
        if payload["pbc_mode"] not in {"bloch", "ring"}:
            raise ValueError("Invalid pbc_mode. Expected 'bloch' or 'ring'.")

    # beta handling:
    # - If beta_num & beta_den exist -> use rational beta=p/q
    # - Else use beta (float). Also accept beta as string like '1/3'.
    beta_num = args.get("beta_num", None)
    beta_den = args.get("beta_den", None)
    beta = args.get("beta", args.get("β", None))

    if isinstance(beta, str) and ("/" in beta) and (beta_num is None) and (beta_den is None):
        a, b = beta.split("/", 1)
        beta_num, beta_den = int(a.strip()), int(b.strip())
        beta = None

    if beta_num is not None and beta_den is not None:
        payload["beta_num"] = int(beta_num)
        payload["beta_den"] = int(beta_den)
    elif beta is not None:
        payload["beta"] = float(beta)
    else:
        if boundary == "pbc" and payload.get("pbc_mode") == "bloch":
            raise ValueError("Missing required arguments for bloch PBC: beta_num and beta_den (or beta='p/q').")
        raise ValueError("Missing required argument: beta (or beta_num/beta_den).")

    prefix = args.get("prefix", None)
    if prefix is not None and str(prefix).strip():
        payload["prefix"] = str(prefix)

    # Light sanity
    payload["N"] = max(int(payload.get("N", 2)), 2)
    payload["n_k"] = max(int(payload.get("n_k", 5)), 5)

    return run_julia_json_tool(AA_HARPER_JL, payload)


def exec_clean_out(args: dict) -> dict:
    payload = {
        "out_dir": str(args.get("out_dir", DEFAULT_OUT_DIR)),
        "dry_run": bool(args.get("dry_run", False)),
    }
    return run_python_best_effort_tool(CLEAN_OUT_PY, payload)


# ---------------------------
# Tool registry (extensible)
# ---------------------------

ToolExecutor = Callable[[dict], dict]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    executor: ToolExecutor

    def as_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="ssh_band",
        description=(
            "Compute SSH spectrum locally via Julia (supports PBC and OBC) and save PNG + data JSON.\n"
            "Key args:\n"
            "- boundary: 'pbc' (bands) or 'obc' (finite-chain spectrum)\n"
            "- OBC uses N (chain length). If overlay_pbc=true, also outputs a PBC band plot for comparison.\n"
            "Returns absolute file paths and metadata as JSON."
        ),
        parameters={
            "type": "object",
            "properties": {
                "boundary": {"type": "string", "enum": ["pbc", "obc"], "description": "Boundary condition. 'pbc' or 'obc'. Default 'pbc'."},
                "p": {"type": "number", "description": "SSH inter-cell hopping ratio t2/t1 (t1 fixed to 1.0 in tool)."},
                "b": {"type": "number", "description": "Staggered onsite term (mass). Default 0."},
                "n_k": {"type": "integer", "description": "Number of k points for PBC bands (and for OBC overlay).", "minimum": 5, "maximum": 5001},
                "N": {"type": "integer", "description": "OBC finite chain length (number of unit cells). Used when boundary='obc'.", "minimum": 2, "maximum": 2000},
                "overlay_pbc": {"type": "boolean", "description": "If boundary='obc', also output a PBC band plot/data for comparison."},
                "out_dir": {"type": "string", "description": 'Output directory, e.g. "out".'},
                "prefix": {"type": "string", "description": "Filename prefix (recommended to avoid mixing runs)."},
                "overwrite": {"type": "boolean", "description": "Overwrite existing files if name collision occurs. Default false."},
            },
            "required": ["p"],
            "additionalProperties": False,
        },
        executor=exec_ssh_band,
    ),
    ToolSpec(
        name="kitaev_band",
        description=(
            "Compute Kitaev chain (1D p-wave superconductor) spectrum locally via Julia (supports PBC and OBC) "
            "and save PNG + data JSON.\n"
            "Key args:\n"
            "- boundary: 'pbc' (BdG bands) or 'obc' (finite-chain BdG spectrum)\n"
            "- Required: mu (chemical potential)\n"
            "- OBC uses N (chain length). If overlay_pbc=true, also outputs a PBC band plot/data for comparison.\n"
            "Returns absolute file paths and metadata as JSON."
        ),
        parameters={
            "type": "object",
            "properties": {
                "boundary": {"type": "string", "enum": ["pbc", "obc"], "description": "Boundary condition. 'pbc' or 'obc'. Default 'pbc'."},
                "t": {"type": "number", "description": "Nearest-neighbor hopping t. Default 1.0."},
                "mu": {"type": "number", "description": "Chemical potential μ. (required)"},
                "delta": {"type": "number", "description": "Pairing amplitude Δ. Default 1.0."},
                "n_k": {"type": "integer", "description": "Number of k points for PBC bands (and for OBC overlay).", "minimum": 5, "maximum": 5001},
                "N": {"type": "integer", "description": "OBC finite chain length (number of sites). Used when boundary='obc'.", "minimum": 2, "maximum": 20000},
                "overlay_pbc": {"type": "boolean", "description": "If boundary='obc', also output a PBC band plot/data for comparison."},
                "out_dir": {"type": "string", "description": 'Output directory, e.g. "out".'},
                "prefix": {"type": "string", "description": "Filename prefix (recommended to avoid mixing runs)."},
                "overwrite": {"type": "boolean", "description": "Overwrite existing files if name collision occurs. Default false."},
            },
            "required": ["mu"],
            "additionalProperties": False,
        },
        executor=exec_kitaev_band,
    ),
    ToolSpec(
        name="aah_harper",
        description=(
            "Compute Aubry–André / Harper (AAH) model spectrum locally via Julia and save PNG + data JSON.\n"
            "Supports:\n"
            "- boundary='obc': finite chain spectrum with IPR-based visualization.\n"
            "- boundary='pbc': either Bloch bands for beta=p/q (pbc_mode='bloch') or finite-ring spectrum (pbc_mode='ring').\n"
            "Required: lambda.\n"
            "For Bloch bands: provide beta_num and beta_den (beta=p/q) or beta='p/q'.\n"
            "For quasi-periodic: provide beta (float).\n"
            "Returns absolute file paths and metadata as JSON."
        ),
        parameters={
            "type": "object",
            "properties": {
                "boundary": {"type": "string", "enum": ["pbc", "obc"], "description": "Boundary condition. Default 'obc'."},
                "pbc_mode": {"type": "string", "enum": ["bloch", "ring"], "description": "Only for boundary='pbc'. 'bloch' requires beta_num/beta_den. If omitted, tool auto-chooses."},
                "t": {"type": "number", "description": "Nearest-neighbor hopping t. Default 1.0."},
                "lambda": {"type": "number", "description": "Potential strength λ (required)."},
                "beta": {"type": "number", "description": "Incommensurate beta for quasi-periodic potential (use with OBC or pbc_mode='ring')."},
                "beta_num": {"type": "integer", "description": "Rational beta=p/q numerator (use with pbc_mode='bloch').", "minimum": 1},
                "beta_den": {"type": "integer", "description": "Rational beta=p/q denominator (use with pbc_mode='bloch').", "minimum": 1},
                "phi": {"type": "number", "description": "Phase offset ϕ. Default 0.0."},
                "n_k": {"type": "integer", "description": "Number of k points for Bloch bands.", "minimum": 5, "maximum": 5001},
                "N": {"type": "integer", "description": "Chain length for finite chain/ring.", "minimum": 2, "maximum": 20000},
                "out_dir": {"type": "string", "description": "Output directory, e.g. 'out'."},
                "prefix": {"type": "string", "description": "Filename prefix."},
                "overwrite": {"type": "boolean", "description": "Overwrite existing files if name collision occurs. Default false."},
            },
            "required": ["lambda"],
            "additionalProperties": False,
        },
        executor=exec_aah_harper,
    ),
    ToolSpec(
        name="clean_out",
        description=(
            "Clean generated artifacts in out_dir (typically ./out). "
            "Use this before a new run to avoid mixing old/new artifacts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "out_dir": {"type": "string", "description": 'Output directory to clean, e.g. "out".'},
                "dry_run": {"type": "boolean", "description": "If true, do not delete; only report intended actions."},
            },
            "required": [],
            "additionalProperties": False,
        },
        executor=exec_clean_out,
    ),
]

TOOL_MAP: dict[str, ToolSpec] = {t.name: t for t in TOOL_SPECS}


# ---------------------------
# Main agent loop
# ---------------------------

def main():
    user_query = " ".join(sys.argv[1:]).strip()
    if not user_query:
        print('Usage: python agent_ssh_runner.py "先清理out，再画SSH：OBC N=80, p=0.6, overlay"')
        sys.exit(2)

    if not API_KEY:
        raise RuntimeError("API key not set. Set DEEPSEEK_API_KEY or OPENAI_API_KEY in your environment.")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    tools = [t.as_openai_tool() for t in TOOL_SPECS]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a scientific assistant.\n"
                "Available local tools:\n"
                "- ssh_band: SSH spectrum with boundary='pbc' or 'obc'.\n"
                "- kitaev_band: Kitaev chain (p-wave superconductor) with boundary='pbc' or 'obc'.\n"
                "- aah_harper: Aubry–André/Harper (AAH) model with boundary='pbc' or 'obc'.\n"
                "- clean_out: clean output directory.\n\n"
                "Hard rules:\n"
                "- Choose the tool by model keywords:\n"
                "  * SSH / Su-Schrieffer-Heeger / dimerized chain -> ssh_band\n"
                "  * Kitaev / Majorana / p-wave / topological superconductor / 拓扑超导 -> kitaev_band\n"
                "  * Aubry–André / Harper / AAH / quasi-periodic / 准周期势 -> aah_harper\n"
                "- Boundary condition: if user says OBC/open boundary/开放边界 -> set boundary='obc'; if PBC/periodic/周期边界 -> boundary='pbc'.\n"
                "- If user asks 'clean then run', call clean_out first, then the requested model tool.\n"
                "- For ssh_band: p is required. If boundary='obc', set N; if user requests overlay/compare with PBC/对比PBC, set overlay_pbc=true.\n"
                "- For kitaev_band: mu is required. If boundary='obc', set N; overlay works via overlay_pbc=true.\n"
                "- For aah_harper: lambda is required.\n"
                "  * For PBC Bloch bands with beta=p/q (e.g. 1/3): boundary='pbc', pbc_mode='bloch', set beta_num/beta_den (or beta='p/q').\n"
                "  * For quasi-periodic beta (e.g. 0.618...): boundary='obc' (or boundary='pbc', pbc_mode='ring'), set beta as float.\n"
                "After tool outputs, report the returned file paths and what each contains."
        ),

        },
        {"role": "user", "content": user_query},
    ]

    for _ in range(8):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        msg = resp.choices[0].message
        messages.append(msg)

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            print(msg.content or "")
            return

        for tc in tool_calls:
            name = tc.function.name
            spec = TOOL_MAP.get(name)
            if spec is None:
                tool_out = {"ok": False, "error": f"Unknown tool: {name}"}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_out, ensure_ascii=False)})
                continue

            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("Arguments must be a JSON object.")
            except Exception as e:
                tool_out = {"ok": False, "error": f"Invalid JSON arguments: {e}", "raw": tc.function.arguments}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_out, ensure_ascii=False)})
                continue

            if DEBUG_TOOL_CALLS:
                print(f"[debug] tool_call name={name} args={json.dumps(args, ensure_ascii=False)}")

            try:
                tool_out = spec.executor(args)
                if not isinstance(tool_out, dict):
                    tool_out = {"ok": True, "result": tool_out}
            except Exception as e:
                tool_out = {"ok": False, "error": str(e), "tool": name}

            if DEBUG_TOOL_CALLS:
                print(f"[debug] tool_out name={name} ok={tool_out.get('ok')} boundary={tool_out.get('boundary')}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_out, ensure_ascii=False)})

    raise RuntimeError("Too many tool-call iterations (possible loop).")


if __name__ == "__main__":
    main()
