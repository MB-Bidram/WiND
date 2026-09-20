"""
Memory-bounded WiND training using the SmolLM2 tokenizer.

Important:
    The exact parameter count depends on the tokenizer vocabulary because
    token embedding and LM-head tables use its vocabulary size.

Dataset preprocessing:
    - Never reads the entire corpus into RAM.
    - Tokenizes bounded text blocks.
    - Stops once MAX_EXAMPLES examples have been collected.
    - Stores token IDs in a compact array.
"""

from __future__ import annotations

import os

# --- profiling: must be set before torch/inductor are imported -------------------
# Makes Inductor name each Triton kernel after the ops it fuses (only affects names).
os.environ.setdefault("TORCHINDUCTOR_UNIQUE_KERNEL_NAMES", "1")

import getpass
import gc
import gzip
import importlib
import json
import linecache
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler, record_function
from array import array
from pathlib import Path
from typing import Iterator

from torch._inductor import config as inductor_config

if hasattr(inductor_config, "cpp") and hasattr(inductor_config.cpp, "openmp"):
    inductor_config.cpp.openmp = False

import torch._dynamo

# Enable explanation logging to see why Dynamo keeps extracting symfloats
torch._dynamo.config.verbose = True

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Set before importing transformers/tokenizers to avoid unnecessary worker threads.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# torch.amp.autocast(dtype=torch.float16)

# =============================================================================
# Project imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from wind import compile as wind_compile
from wind.language import LMConfig, LanguageModel, LMTrainer


# =============================================================================
# User settings
# =============================================================================

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
DATA_PATH = Path(r"F:\Deep Learning\Programming\Python\Rout\wiki_training_data_cleane3d.txt")
CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "wind_checkpoints"

SEED = 42
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256
MAX_EXAMPLES = 5000
TEXT_BLOCK_CHARS = 8 * 1024

# The prior batch=64 at sequence length 256 materializes a ~1.9 GiB FP32 PKM
# factor-score tensor before backward allocations. Keep the effective batch
# (192 sequences) but bound activation memory with micro-batches.
BATCH_SIZE = 16
GRAD_ACCUMULATION = 4
EPOCHS = 3

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

NUM_WORKERS = 0
USE_AMP = True
USE_COMPILE = True
COMPILE_MODE = "auto"
ACTIVATION_CHECKPOINTING = True

# -----------------------------------------------------------------------------
# Profiling settings (added). Model, loss and training math are NOT changed.
# -----------------------------------------------------------------------------
PROFILE_COMPILED_PASS = True   # pass 1: your exact run (BATCH_SIZE, compiled if USE_COMPILE)
PROFILE_EAGER_PASS = True      # pass 2: NOT compiled -> kernels/allocations map to YOUR source lines
EAGER_BATCH_SIZE = 16          # smaller so the eager pass fits in 4 GB; set = BATCH_SIZE to match pass 1
ACTIVE_STEPS = 3               # profiled steps per pass (wait=1, warmup=1, active=ACTIVE_STEPS)
SYNC_BEFORE_PROF_STEP = True   # cuda sync before prof.step() so each step owns its GPU work
MEMORY_SNAPSHOT = True         # allocator history with Python stacks (+ pickle for pytorch.org/memory_viz)
MEMORY_HISTORY_ENTRIES = 300_000
SAMPLE_NVIDIA_SMI = True       # background nvidia-smi: utilisation / dedicated memory / clocks
TOP_N = 25                     # rows per report table

# A/B toggles for testing fixes inside this same harness. Defaults = your original behaviour.
VOCAB_PAD_MULTIPLE = 1         # 1 = unchanged (vocab 49154). Try 64: avoids Inductor's padded 1.5 GiB copy + align2 GEMMs
LOSS_ONLY_FORWARD = False      # True = call model(..., return_logits=False)  (needs the patched wind/language/model.py)


# =============================================================================
# Utilities
# =============================================================================
import traceback
from collections import Counter

_original_item = torch.Tensor.item

# (file, line, function, source) -> count
_item_calls = Counter()


def debug_item(self):
    # Skip this function's own frame and get the first useful caller.
    frame = traceback.extract_stack(limit=4)[-2]

    key = (
        frame.filename,
        frame.lineno,
        frame.name,
        frame.line or "",
    )

    _item_calls[key] += 1

    return _original_item(self)


torch.Tensor.item = debug_item


def print_item_report():
    print("\n" + "=" * 110)
    print("TORCH TENSOR.item() REPORT")
    print("=" * 110)

    if not _item_calls:
        print("No .item() calls detected.")
        return

    print(f"{'COUNT':>8}  {'FILE':<55} {'LINE':>6}  {'FUNCTION':<25}  SOURCE")
    print("-" * 110)

    for (filename, lineno, function, source), count in _item_calls.most_common():
        # Keep paths readable
        if len(filename) > 55:
            filename = "..." + filename[-52:]

        print(
            f"{count:>8}  "
            f"{filename:<55} "
            f"{lineno:>6}  "
            f"{function:<25}  "
            f"{source or ''}"
        )

    print("=" * 110)
    print(f"TOTAL .item() CALLS: {sum(_item_calls.values()):,}")
    print("=" * 110)

def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device_and_precision() -> tuple[torch.device, bool, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = USE_AMP and device.type == "cuda"
    if not amp_enabled:
        return device, False, torch.float32

    # LMTrainer accepts bf16 or fp16 whenever AMP is enabled.  Select fp16 on
    # CUDA devices without native bf16 support instead of passing float32.
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device, amp_enabled, amp_dtype


# =============================================================================
# Tokenizer
# =============================================================================

class WindTokenizer:
    def __init__(self, model_name: str):
        print_section("TOKENIZER")
        self.model_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        special_tokens = {}
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            special_tokens["pad_token"] = "<|wind_pad|>"
        if tokenizer.bos_token_id is None or tokenizer.bos_token_id == tokenizer.eos_token_id or tokenizer.bos_token_id == tokenizer.pad_token_id:
            special_tokens["bos_token"] = "<|wind_bos|>"
        if tokenizer.eos_token_id is None:
            special_tokens["eos_token"] = "<|wind_eos|>"

        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)

        self._tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if any(token_id is None for token_id in (self.pad_token_id, self.bos_token_id, self.eos_token_id)):
            raise RuntimeError("Tokenizer must define PAD, BOS, and EOS tokens.")
        if len({self.pad_token_id, self.bos_token_id, self.eos_token_id}) != 3:
            raise RuntimeError("PAD, BOS, and EOS token IDs must be distinct.")

    def encode(self, text, add_special_tokens=False, **kwargs):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)

    def decode(self, ids, **kwargs):
        return self._tokenizer.decode(ids, **kwargs)

    def save_pretrained(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save_pretrained(str(directory))


# =============================================================================
# Bounded text reader
# =============================================================================

def iter_text_blocks(file_path: Path, block_chars: int) -> Iterator[str]:
    if block_chars < 1024:
        raise ValueError("block_chars must be at least 1024.")
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        carry = ""
        while True:
            remaining = block_chars - len(carry)
            incoming = handle.read(remaining)
            if not incoming:
                if carry:
                    yield carry
                break
            text = carry + incoming
            carry = ""
            if len(text) < block_chars:
                yield text
                break
            search_start = max(1, len(text) - 2048)
            split_at = len(text)
            for index in range(len(text) - 1, search_start - 1, -1):
                if text[index].isspace():
                    split_at = index
                    break
            carry = text[split_at:]
            yield text[:split_at]


# =============================================================================
# Dataset
# =============================================================================

class WikiTextDataset(Dataset):
    def __init__(self, file_path: Path, tokenizer: WindTokenizer, max_source_len: int = 128, max_target_len: int = 128, max_examples: int = 5000, block_chars: int = 64 * 1024):
        super().__init__()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.tokens_per_example = max_source_len + (max_target_len - 2)

        self._tokens = array("I")
        token_limit = max_examples * self.tokens_per_example
        
        block_iterator = iter_text_blocks(file_path, block_chars)
        for text in block_iterator:
            block_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            remaining = token_limit - len(self._tokens)
            self._tokens.extend(block_ids[:remaining])
            if len(self._tokens) >= token_limit:
                break
        
        self.num_examples = len(self._tokens) // self.tokens_per_example
        if self.num_examples == 0:
            raise RuntimeError("Corpus too small for one example.")
        del self._tokens[self.num_examples * self.tokens_per_example:]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        start = index * self.tokens_per_example
        source_end = start + self.max_source_len
        target_end = start + self.tokens_per_example
        return {
            "input_ids": torch.tensor(self._tokens[start:source_end], dtype=torch.long),
            "labels": torch.tensor([self.bos_id, *self._tokens[source_end:target_end], self.eos_id], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([example["input_ids"] for example in batch]),
        "labels": torch.stack([example["labels"] for example in batch]),
    }


def compile_model(model: LanguageModel, dataloader: DataLoader, device: torch.device) -> torch.nn.Module:
    """Compile the model with a real fixed-shape training batch as warmup."""
    if not USE_COMPILE:
        return model

    warmup_batch = next(iter(dataloader))
    warmup_input = {
        name: value.to(device, non_blocking=True)
        for name, value in warmup_batch.items()
    }
    return wind_compile(
        model,
        mode=COMPILE_MODE,
        fullgraph=False,
        fallback_to_eager=False,
        warmup=True,
        warmup_input=warmup_input,
    )


# =============================================================================
# Model configuration
# =============================================================================

def build_config(tokenizer: WindTokenizer):
    return LMConfig(
        vocab_size=-(-tokenizer.vocab_size // VOCAB_PAD_MULTIPLE) * VOCAB_PAD_MULTIPLE,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        dim=128, 
        heads=4, 
        wide_type="pkm",
        pkm_memory_size=1024, pkm_num_factors=2, pkm_heads=4, pkm_topk=8,
        pkm_topk_per_factor=16, pkm_key_dtype="float32", pkm_value_dtype="float32",
        pkm_similarity="cosine", pkm_exact_candidate_pruning=True,
        use_alpha_learning=True, alpha_init=0.0,
        encoder_depth=2, encoder_type="standard",
        depth=2, iterations=1, decoder_depth=1,
        state_tokens=8, bank_tokens=16,
        max_source_length=MAX_SOURCE_LENGTH, max_target_length=MAX_TARGET_LENGTH,
        mlp_ratio=2.0, dropout=0.0, norm="rmsnorm", use_rope=True,
        rope_theta=10000.0, rope_max_seq_len=2048,
        checkpointing=ACTIVATION_CHECKPOINTING, custom_modules={}
    )


# =============================================================================
# PROFILING HELPERS (added).  None of this touches the model or the training
# math; it only records and analyses.  Every analysis step is wrapped so a
# failure in a report can never kill the run.
# =============================================================================

_SCRIPT_START = time.time()
_OUTER_FRAME_NAMES = {"<module>", "main", "run_profile_pass"}
_PYF_RE = re.compile(r"^(.*)\((\d+)\): (.+)$")


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path)) if path else ""


_PROJECT_ROOT_NORM = _norm(str(PROJECT_ROOT))


def _is_project_file(filename: str) -> bool:
    p = _norm(filename)
    return bool(p) and p.startswith(_PROJECT_ROOT_NORM) and "site-packages" not in p \
        and (os.sep + ".tmp" + os.sep) not in p


def _is_generated_inductor_file(filename: str) -> bool:
    return "torchinductor" in _norm(filename)


def _is_interesting_frame(filename: str) -> bool:
    """Frames worth blaming: your own code, or Inductor-generated code."""
    return _is_project_file(filename) or _is_generated_inductor_file(filename)


def _innermost_first(frames):
    """frames: [(file, line, name)].  Return them innermost-first."""
    if frames and frames[0][2] in _OUTER_FRAME_NAMES and frames[-1][2] not in _OUTER_FRAME_NAMES:
        return list(reversed(frames))
    return list(frames)


def _fmt_frame(fr) -> str:
    fn, ln, name = fr
    src = linecache.getline(fn, int(ln)).strip() if fn and not str(fn).startswith("<") else ""
    src = (src[:110] + "...") if len(src) > 113 else src
    return f"{os.path.basename(fn)}:{ln} {name}" + (f"  |  {src}" if src else "")


def _mib(n) -> str:
    return f"{n / 2**20:,.1f} MiB"


def _table(rows, headers, widths):
    line = "  ".join(f"{h:<{w}}" if i == len(headers) - 1 else f"{h:>{w}}"
                     for i, (h, w) in enumerate(zip(headers, widths)))
    print(line)
    print("-" * min(len(line) + 40, 200))
    for r in rows:
        print("  ".join(f"{str(c):<{w}}" if i == len(r) - 1 else f"{str(c):>{w}}"
                        for i, (c, w) in enumerate(zip(r, widths))))


# -----------------------------------------------------------------------------
# nvidia-smi sampler (utilisation, dedicated memory used, PCIe link)
# -----------------------------------------------------------------------------

class GpuSampler:
    def __init__(self, path: Path):
        self.path, self.proc, self.handle = path, None, None

    def start(self):
        if not SAMPLE_NVIDIA_SMI or shutil.which("nvidia-smi") is None:
            return
        try:
            self.handle = open(self.path, "w")
            self.proc = subprocess.Popen(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,"
                 "pcie.link.gen.current,pcie.link.width.current,clocks.sm",
                 "--format=csv,noheader,nounits", "-lms", "250"],
                stdout=self.handle, stderr=subprocess.DEVNULL)
        except Exception as exc:
            print(f"[gpu sampler] could not start nvidia-smi: {exc}")
            self.proc = None

    def stop_and_report(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            self.handle.close()
            util, used, total, gen, width, clk = [], [], 0, "?", "?", []
            for line in open(self.path):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                try:
                    util.append(float(parts[0])); used.append(float(parts[1]))
                    total = float(parts[2]); gen, width = parts[3], parts[4]; clk.append(float(parts[5]))
                except ValueError:
                    continue
            if util:
                print(f"[nvidia-smi] samples={len(util)}  GPU util avg={sum(util)/len(util):.0f}% max={max(util):.0f}%  "
                      f"dedicated mem used max={max(used):.0f}/{total:.0f} MiB  "
                      f"PCIe gen{gen} x{width}  SM clock avg={sum(clk)/len(clk):.0f} MHz")
                print(f"[nvidia-smi] raw samples: {self.path}")
        except Exception as exc:
            print(f"[gpu sampler] report failed: {exc}")


# -----------------------------------------------------------------------------
# Allocator snapshot: who owns the memory at the PEAK
# -----------------------------------------------------------------------------

def analyze_memory_snapshot(snapshot: dict, tag: str, base_allocated: int, top_n: int) -> None:
    print_section(f"[{tag}] MEMORY AT PEAK  (allocator snapshot, with Python stacks)")
    traces = snapshot.get("device_traces") or []
    events = traces[0] if traces else []
    if not events:
        print("No allocator trace recorded (CPU run, or record_memory_history failed).")
        return

    live, total, peak, peak_i = {}, 0, 0, -1
    for i, ev in enumerate(events):
        action = ev.get("action")
        if action == "alloc":
            live[ev["addr"]] = ev["size"]
            total += ev["size"]
            if total > peak:
                peak, peak_i = total, i
        elif action == "free_completed":
            total -= live.pop(ev["addr"], 0)

    oom_events = [ev for ev in events if ev.get("action") == "oom"]
    print(f"allocated before recording started (weights, optimizer, ...): {_mib(base_allocated)}")
    print(f"peak of allocations made while recording:                     {_mib(peak)}")
    print(f"=> estimated peak                                             {_mib(base_allocated + peak)}")
    if oom_events:
        print(f"!! OOM event: allocator failed to get {_mib(oom_events[-1].get('size', 0))}")
    if peak_i < 0:
        return

    at_peak, total = {}, 0
    for ev in events[: peak_i + 1]:
        action = ev.get("action")
        if action == "alloc":
            at_peak[ev["addr"]] = (ev["size"], ev.get("frames") or [])
        elif action == "free_completed":
            at_peak.pop(ev["addr"], None)

    def frames_of(fr_list):
        return _innermost_first([(f.get("filename", ""), f.get("line", 0), f.get("name", "")) for f in fr_list])

    def caller(fr_list):
        fr = frames_of(fr_list)
        pick = next((f for f in fr if _is_interesting_frame(f[0])), None)
        return pick, fr

    print(f"\n-- largest live allocations at the peak (top {top_n}) --")
    biggest = sorted(at_peak.values(), key=lambda x: -x[0])[:top_n]
    for size, fr_list in biggest:
        pick, fr = caller(fr_list)
        proj = next((f for f in fr if _is_project_file(f[0])), None)
        print(f"{_mib(size):>14}  {_fmt_frame(pick) if pick else '<no frame in your code / generated code>'}")
        if proj and pick and proj != pick:
            print(f"{'':>14}  called from your code at {_fmt_frame(proj)}")

    groups = defaultdict(lambda: [0, 0])
    for size, fr_list in at_peak.values():
        pick, _ = caller(fr_list)
        key = _fmt_frame(pick) if pick else "<unattributed>"
        groups[key][0] += size
        groups[key][1] += 1
    print(f"\n-- live memory at the peak, grouped by source line (top {top_n}) --")
    rows = [(_mib(v[0]), v[1], k) for k, v in sorted(groups.items(), key=lambda kv: -kv[1][0])[:top_n]]
    _table(rows, ["BYTES", "COUNT", "SOURCE LINE"], [14, 6, 100])


# -----------------------------------------------------------------------------
# Chrome-trace analysis: every kernel / allocation -> range, module, aten op,
# Python source line
# -----------------------------------------------------------------------------

def _stacks_at(intervals, points):
    """intervals: [(ts, end, payload)]; points: [(ts, idx)].
    Returns {idx: [payload outermost..innermost]} for the intervals open at ts."""
    items = [(iv[0], 0, -(iv[1] - iv[0]), iv) for iv in intervals]
    items += [(p[0], 1, 0, p) for p in points]
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    stack, out = [], {}
    for ts, kind, _, obj in items:
        while stack and stack[-1][1] <= ts:
            stack.pop()
        if kind == 0:
            stack.append(obj)
        else:
            out[obj[1]] = [s[2] for s in stack]
    return out


def analyze_trace_json(path: Path, tag: str, n_steps: int, top_n: int) -> list[str]:
    """Returns the names of the top kernels (for the Inductor source lookup)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    events = data["traceEvents"]

    kernels = [e for e in events if e.get("cat") == "kernel"]
    runtime = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None and corr not in runtime:
                runtime[corr] = e

    py_iv, op_iv = defaultdict(list), defaultdict(list)
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat == "python_function":
            py_iv[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
        elif cat in ("cpu_op", "user_annotation"):
            op_iv[e["tid"]].append((e["ts"], e["ts"] + e["dur"], (cat, e["name"])))
    have_python = any(py_iv.values())

    k_points = defaultdict(list)
    for idx, k in enumerate(kernels):
        r = runtime.get(k.get("args", {}).get("correlation"))
        if r is not None:
            k_points[r["tid"]].append((r["ts"], idx))

    mem_events = [e for e in events if e.get("name") == "[memory]"
                  and e.get("args", {}).get("Device Type") == 1 and e["args"].get("Bytes", 0) > 0]
    m_points = defaultdict(list)
    for idx, m in enumerate(mem_events):
        m_points[m["tid"]].append((m["ts"], idx))

    def resolve(tid, idx, point_map_py, point_map_op):
        pys = point_map_py.get(tid, {}).get(idx, [])
        ops = point_map_op.get(tid, {}).get(idx, [])
        ranges = [n for cat, n in ops if cat == "user_annotation"
                  and not n.startswith(("ProfilerStep", "enumerate("))]
        aten = next((n for cat, n in reversed(ops) if cat == "cpu_op" and n.startswith("aten::")), "")
        frames = []
        for name in reversed(pys):
            m = _PYF_RE.match(name)
            if m and _is_interesting_frame(m.group(1)):
                frames.append((m.group(1), int(m.group(2)), m.group(3)))
        modules = [n[len("nn.Module: "):] for n in pys if n.startswith("nn.Module: ")]
        return " / ".join(ranges) or "(no range)", aten or "(no aten op)", \
            (frames[0] if frames else None), " > ".join(modules[-3:]) or "(no module)"

    def stacks_for(points_by_tid):
        py = {t: _stacks_at(py_iv.get(t, []), pts) for t, pts in points_by_tid.items()}
        op = {t: _stacks_at(op_iv.get(t, []), pts) for t, pts in points_by_tid.items()}
        return py, op

    k_py, k_op = stacks_for(k_points)
    idx_to_tid = {idx: tid for tid, pts in k_points.items() for _, idx in pts}

    by_name, by_range, by_aten, by_line, by_module = (defaultdict(lambda: [0, 0.0]) for _ in range(5))
    total_us, unattributed = 0.0, 0.0
    for idx, k in enumerate(kernels):
        dur = k.get("dur", 0.0)
        total_us += dur
        by_name[k["name"]][0] += 1; by_name[k["name"]][1] += dur
        tid = idx_to_tid.get(idx)
        if tid is None:
            unattributed += dur
            continue
        rng, aten, frame, module = resolve(tid, idx, k_py, k_op)
        for table, key in ((by_range, rng), (by_aten, aten), (by_module, module),
                           (by_line, _fmt_frame(frame) if frame else "(no python frame in your code / generated code)")):
            table[key][0] += 1
            table[key][1] += dur

    per_step_ms = lambda us: us / max(n_steps, 1) / 1000.0
    pct = lambda us: 100.0 * us / total_us if total_us else 0.0

    print_section(f"[{tag}] GPU KERNELS  ({len(kernels)} kernels in {n_steps} steps, "
                  f"{per_step_ms(total_us):,.1f} ms GPU time per step)")

    def show(title, table, width, limit=top_n):
        print(f"\n-- {title} --")
        rows = [(f"{per_step_ms(v[1]):,.1f}", f"{pct(v[1]):.1f}%", f"{v[0] / max(n_steps, 1):.0f}", k[:200])
                for k, v in sorted(table.items(), key=lambda kv: -kv[1][1])[:limit]]
        _table(rows, ["ms/step", "share", "k/step", "WHAT"], [10, 7, 7, width])

    show("by kernel name", by_name, 110)
    show("by named range (record_function: forward/backward/encoder_stack/loss/...)", by_range, 100)
    show("by aten op that launched it", by_aten, 60)
    show("by nn.Module (last 3 in the call path)", by_module, 100)
    if have_python:
        show("by YOUR source line (or Inductor-generated line)", by_line, 150)
    else:
        print("\n(no python_function events in this trace: per-line attribution unavailable in this "
              "torch build; use the named ranges / nn.Module / allocator-snapshot sections)")
    if unattributed:
        print(f"\nkernels with no matching launch event: {per_step_ms(unattributed):.1f} ms/step")

    # ---- allocations seen by the profiler, with owners ----
    if mem_events:
        m_py, m_op = stacks_for(m_points)
        idx_to_tid_m = {idx: tid for tid, pts in m_points.items() for _, idx in pts}
        allocs = defaultdict(lambda: [0, 0])
        for idx, m in enumerate(mem_events):
            tid = idx_to_tid_m.get(idx)
            if tid is None:
                continue
            rng, aten, frame, module = resolve(tid, idx, m_py, m_op)
            key = f"{rng}  |  {aten}  |  {_fmt_frame(frame) if frame else '(no frame)'}"
            allocs[key][0] += m["args"]["Bytes"]
            allocs[key][1] += 1
        print_section(f"[{tag}] GPU ALLOCATIONS  (profiler memory events, all steps, top {top_n} by bytes)")
        rows = [(_mib(v[0] / max(n_steps, 1)), f"{v[1] / max(n_steps, 1):.0f}", k[:220])
                for k, v in sorted(allocs.items(), key=lambda kv: -kv[1][0])[:top_n]]
        _table(rows, ["MiB/step", "n/step", "RANGE  |  ATEN OP  |  SOURCE LINE"], [12, 7, 150])
        print("(MiB/step = total bytes requested per step, not simultaneous live memory; "
              "see the allocator-snapshot section for the peak)")

    return [name for name, _ in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:top_n]]


# -----------------------------------------------------------------------------
# Compiled-mode helper: Inductor kernel name -> the source nodes it fuses
# -----------------------------------------------------------------------------

def describe_inductor_kernels(kernel_names: list[str]) -> None:
    names = [n for n in kernel_names if n.startswith("triton_")]
    if not names:
        return
    print_section("INDUCTOR: what the top Triton kernels compute (from the generated code comments)")
    cache_dir = None
    for mod, fn in (("torch._inductor.runtime.cache_dir_utils", "cache_dir"),
                    ("torch._inductor.codecache", "cache_dir")):
        try:
            cache_dir = getattr(importlib.import_module(mod), fn)()
            break
        except Exception:
            continue
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "torchinductor_" + getpass.getuser())
    print(f"scanning generated code in: {cache_dir}")
    remaining, found = set(names), {}
    deadline = time.time() + 60
    for root, _, files in os.walk(cache_dir):
        for fname in files:
            if not fname.endswith(".py") or not remaining or time.time() > deadline:
                continue
            fpath = os.path.join(root, fname)
            try:
                if os.path.getsize(fpath) > 40 * 2**20:
                    continue
                text = open(fpath, "r", encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for name in list(remaining):
                pos = text.find(f"async_compile.triton('{name}'")
                if pos < 0:
                    continue
                before = text[:pos].splitlines()[-14:]
                comments = [ln.strip() for ln in before if ln.strip().startswith("#")]
                found[name] = (fpath, text[:pos].count("\n") + 1, comments)
                remaining.discard(name)
    for name in names:
        print(f"\n{name}")
        if name in found:
            fpath, line, comments = found[name]
            print(f"   defined at {fpath}:{line}")
            for c in comments:
                print(f"   {c[:200]}")
        else:
            print("   (not found in the Inductor cache)")


# -----------------------------------------------------------------------------
# One profiling pass
# -----------------------------------------------------------------------------

def run_profile_pass(tag: str, trainer, dataloader, device: torch.device, out_dir: Path) -> None:
    print_section(f"PROFILING PASS: {tag}   (batch={dataloader.batch_size}, out={out_dir})")
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda = device.type == "cuda"
    total_mem = torch.cuda.get_device_properties(device).total_memory if cuda else 0
    base_alloc = 0
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()

    recording = False
    if cuda and MEMORY_SNAPSHOT:
        try:
            torch.cuda.memory._record_memory_history(max_entries=MEMORY_HISTORY_ENTRIES, stacks="python")
            recording = True
        except Exception:
            try:
                torch.cuda.memory._record_memory_history(max_entries=MEMORY_HISTORY_ENTRIES)
                recording = True
            except Exception as exc:
                print(f"[memory history] unavailable: {exc}")

    sampler = GpuSampler(out_dir / f"{tag}_nvidia_smi.csv")
    sampler.start()

    holder = {}
    tb_handler = tensorboard_trace_handler(str(out_dir / "tensorboard"), worker_name=tag)

    def on_ready(p):
        holder["prof"] = p
        tb_handler(p)

    n_steps = 2 + ACTIVE_STEPS            # wait=1 + warmup=1 + active
    prof_schedule = schedule(wait=1, warmup=1, active=ACTIVE_STEPS, repeat=1)
    step_rows, oom = [], False

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=on_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        with_flops=True,
    ) as prof:

        # Run a small profiling window rather than the entire training run.
        trainer.model.train()

        for step, batch in enumerate(dataloader):
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                # Move batch to GPU.
                with record_function("h2d"):
                    batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}

                trainer.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=trainer.amp_dtype, enabled=trainer.amp):
                    with record_function("forward"):
                        output = trainer.model(**batch, **({"return_logits": False} if LOSS_ONLY_FORWARD else {}))
                        loss = output.loss

                with record_function("backward"):
                    loss.backward()

                if SYNC_BEFORE_PROF_STEP and cuda:
                    torch.cuda.synchronize()      # so each ProfilerStep really contains its own GPU work
                prof.step()
            except torch.cuda.OutOfMemoryError as exc:
                oom = True
                print(f"\n!!! CUDA OUT OF MEMORY at step {step + 1}: {str(exc).splitlines()[0]}")
                break

            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak_a = torch.cuda.max_memory_allocated() if cuda else 0
            peak_r = torch.cuda.max_memory_reserved() if cuda else 0
            step_rows.append((step + 1, f"{dt:.3f}", _mib(peak_a), _mib(peak_r)))

            print(f"Profiler step {step + 1}/{n_steps} ")
            print(f"| loss={loss.detach().item():.6f}")

            if step >= n_steps - 1:
                break

    sampler.stop_and_report()

    print_section(f"[{tag}] PER-STEP SUMMARY")
    _table(step_rows, ["STEP", "SECONDS", "PEAK ALLOC", "PEAK RESERVED"], [6, 9, 14, 14]) if step_rows else None
    if cuda:
        stats = torch.cuda.memory_stats()
        peak_res = torch.cuda.max_memory_reserved()
        print(f"device total: {_mib(total_mem)}   overall peak reserved: {_mib(peak_res)}   "
              f"alloc retries: {stats.get('num_alloc_retries', 0)}   ooms: {stats.get('num_ooms', 0)}")
        if peak_res > total_mem * 0.98 or oom:
            print("!! The allocator wants >= the whole card.  On Windows the driver then spills to system "
                  "RAM over PCIe (kernels get 10-30x slower) or you get an OOM.")

    if recording:
        try:
            snap = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)
            snap_path = out_dir / f"{tag}_memory_snapshot.pickle"
            with open(snap_path, "wb") as handle:
                pickle.dump(snap, handle)
            print(f"[memory snapshot] {snap_path}   (drop it on https://pytorch.org/memory_viz)")
            analyze_memory_snapshot(snap, tag, base_alloc, TOP_N)
        except Exception as exc:
            print(f"[memory snapshot] failed: {exc}")

    p = holder.get("prof", prof)
    # tensorboard_trace_handler already saved the trace (a profiler can only save once),
    # so pick that file up instead of exporting again.
    trace_path = None
    try:
        cands = sorted((out_dir / "tensorboard").glob(f"{tag}.*.pt.trace.json*"),
                       key=lambda f: f.stat().st_mtime)
        trace_path = cands[-1] if cands else None
    except Exception:
        pass
    if trace_path is not None:
        print(f"\n[chrome trace] {trace_path}   (open in https://ui.perfetto.dev)")
    else:
        print("[chrome trace] no trace file found in the tensorboard folder")

    table_txt = None
    for metric in ("self_device_time_total", "self_cuda_time_total"):
        for extra in ({"max_shapes_column_width": 60}, {}):
            try:
                table_txt = p.key_averages(group_by_input_shape=True).table(
                    sort_by=metric, row_limit=TOP_N, max_name_column_width=55, **extra)
                break
            except Exception:
                continue
        if table_txt:
            print_section(f"[{tag}] TOP OPS BY {metric} (profiler table, with shapes)")
            print(table_txt)
            break

    try:
        rows = []
        for ka in p.key_averages():
            flops = getattr(ka, "flops", 0) or 0
            dev_us = getattr(ka, "device_time_total", None)
            if dev_us is None:
                dev_us = getattr(ka, "cuda_time_total", 0)
            if flops > 0 and dev_us > 0:
                rows.append((dev_us, ka.key, ka.count, flops, flops / (dev_us * 1e-6) / 1e12))
        rows.sort(reverse=True)
        print_section(f"[{tag}] ACHIEVED TFLOP/s PER OP  (profiler FLOP estimate / device time)")
        _table([(f"{d / 1000 / max(ACTIVE_STEPS, 1):,.1f}", n, c, f"{f / 1e9:,.1f}", f"{t:.2f}")
                for d, n, c, f, t in rows[:12]],
               ["ms/step", "OP", "CALLS", "GFLOP", "TFLOP/s"], [10, 30, 6, 12, 8])
    except Exception as exc:
        print(f"[tflops table] failed: {exc}")

    if trace_path is not None:
        try:
            names = analyze_trace_json(trace_path, tag, ACTIVE_STEPS, TOP_N)
            describe_inductor_kernels(names)
        except Exception as exc:
            print(f"[trace analysis] failed: {exc!r}")
            traceback.print_exc()

    for exporter, fname, arg in (
        ("export_stacks", f"{tag}_stacks_cuda.txt", "self_cuda_time_total"),
        ("export_stacks", f"{tag}_stacks_cuda.txt", "self_device_time_total"),
    ):
        try:
            getattr(p, exporter)(str(out_dir / fname), arg)
            print(f"[flamegraph stacks] {out_dir / fname}   (feed to flamegraph.pl / speedscope)")
            break
        except Exception:
            continue



# =============================================================================
# Main
# =============================================================================

def main():
    seed_everything(SEED)
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training text file does not exist: {DATA_PATH}")

    tokenizer = WindTokenizer(TOKENIZER_NAME)
    config = build_config(tokenizer)
    dataset = WikiTextDataset(DATA_PATH, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, MAX_EXAMPLES, TEXT_BLOCK_CHARS)

    device, amp_enabled, amp_dtype = select_device_and_precision()
    model = LanguageModel(config).to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    model = compile_model(model, dataloader, device)
    print("Compile done.")
    trainer = LMTrainer(
        model,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=device,
        amp=amp_enabled,
        amp_dtype=amp_dtype,
        grad_accumulation=GRAD_ACCUMULATION,
        clip_grad=GRAD_CLIP,
        logger="console",
    )

    # =========================================================================
    # PROFILING
    # =========================================================================

    run_dir = PROJECT_ROOT / ".tmp" / "profile_runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print_section(f"PROFILING -> {run_dir}")

    if PROFILE_COMPILED_PASS:
        tag = "compiled" if USE_COMPILE else "eager_full_batch"
        try:
            run_profile_pass(tag, trainer, dataloader, device, run_dir)
        except Exception as exc:
            print(f"[{tag}] pass aborted: {exc!r}")
            traceback.print_exc()

    if PROFILE_EAGER_PASS:
        # Free the compiled model first so the eager pass starts from a clean card.
        try:
            del trainer, model
        except NameError:
            pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seed_everything(SEED)
        eager_model = LanguageModel(config).to(device)
        eager_loader = DataLoader(
            dataset,
            batch_size=EAGER_BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
            persistent_workers=NUM_WORKERS > 0,
        )
        eager_trainer = LMTrainer(
            eager_model,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            device=device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            grad_accumulation=GRAD_ACCUMULATION,
            clip_grad=GRAD_CLIP,
            logger="console",
        )
        try:
            run_profile_pass("eager", eager_trainer, eager_loader, device, run_dir)
        except Exception as exc:
            print(f"[eager] pass aborted: {exc!r}")
            traceback.print_exc()

    print()
    print(f"All profiling output written to: {run_dir}")
    print("  *_chrome_trace.json        -> https://ui.perfetto.dev")
    print("  *_memory_snapshot.pickle   -> https://pytorch.org/memory_viz")
    print("  tensorboard/               -> python -m tensorboard.main --logdir \"" + str(run_dir / "tensorboard") + "\"")

if __name__ == "__main__":
    main()
    print_item_report()