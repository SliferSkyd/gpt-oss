#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contest Evaluation Script for LLM Generation-Only Outputs
- Primary metrics: BERTScore (F1), METEOR
- Auxiliary: distinct-1/2, repetition rate, average length
- Modes:
  * Reference-based: requires --references, computes METEOR + BERTScore vs gold
- Fairness: evaluate on the intersection of IDs across all candidates (and refs if provided).

Multi-GPU:
- Dùng --devices để chọn GPU: "auto" (mặc định, dùng tất cả), "cpu", hoặc "0,1,3".
- Script sẽ chia dữ liệu thành N shard ~ đều nhau và chạy BERTScore trên từng GPU ("cuda:<id>").
- Trên AMD ROCm, PyTorch vẫn hiển thị device dưới namespace torch.cuda với device string "cuda".
"""

import argparse
import hashlib
import json
import math
import pathlib
import statistics
from typing import Dict, List, Sequence, Tuple

import evaluate
from jsonschema import validate
from jsonschema import ValidationError
import nltk
import pandas as pd
import tiktoken
import torch
import multiprocessing as mp


# ---------------------- Device helpers ---------------------- #
def detect_backend() -> str:
    if torch.cuda.is_available():
        return "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "MPS"
    return "CPU"


def parse_devices_arg(dev_arg: str) -> List[int]:
    """
    Parse --devices:
      - "auto" / "all" => tất cả GPU sẵn có
      - "cpu" => dùng CPU (trả về list rỗng)
      - "0,1,3" => danh sách GPU chỉ định
    """
    dev_arg = (dev_arg or "auto").strip().lower()
    if dev_arg in ("cpu", "none"):
        return []
    if dev_arg in ("auto", "all"):
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    # danh sách id
    result = []
    for tok in dev_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx = int(tok)
            result.append(idx)
        except ValueError:
            pass
    # lọc các id hợp lệ
    max_idx = torch.cuda.device_count() - 1
    return [i for i in result if 0 <= i <= max_idx]


def log_devices(device_ids: List[int]):
    backend = detect_backend()
    if device_ids:
        names = [torch.cuda.get_device_name(i) for i in device_ids]
        print(f"[DEVICE] Backend: {backend} | Using GPUs: {device_ids} -> {names}")
    else:
        if backend == "MPS":
            print("[DEVICE] Apple MPS detected; forcing CPU for BERTScore to be safe.")
        else:
            print("[DEVICE] Using CPU")


# ---------------------- CLI ---------------------- #
def parseCLI():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompts", default="../data/input.txt", type=pathlib.Path,
                        help="Path to prompts file")
    parser.add_argument("-s", "--submission", default="../data/output.txt", type=pathlib.Path,
                        help="Path to trainee's completion file, e.g., ../data/subm.txt")
    parser.add_argument("-r", "--references", default="refs_fp32.txt", type=pathlib.Path,
                        help="Path to reference completion file used for evaluation")

    # BERTScore options
    parser.add_argument("--bs-model", default="microsoft/deberta-xlarge-mnli", type=str,
                        help="BERTScore backbone model_type")
    parser.add_argument("--bs-batch", default=4, type=int,
                        help="BERTScore batch_size (per GPU). Set 1 if VRAM is tight.")

    # Multi-GPU options
    parser.add_argument("--devices", default="auto", type=str,
                        help='GPU selection: "auto" (all), "cpu", or comma list like "0,1,2,3"')
    parser.add_argument("--workers", default="auto", type=str,
                        help='Number of worker processes: "auto" (=len(devices)) or an int')

    return parser.parse_args()


# ---------------------- Schema & data utils ---------------------- #
def validate_references(refs: List[Dict], schema_path: pathlib.Path) -> bool:
    with open(schema_path, "r") as f:
        schema = json.load(f)

    ok = True
    for i, entry in enumerate(refs, start=1):
        try:
            validate(instance=entry, schema=schema)
        except ValidationError as e:
            print(f"[SCHEMA ERROR] Line {i}: {e.message}")
            ok = False
    if ok:
        print("[SCHEMA CHECK] All references match schema")
    return ok


def ensure_nltk_resources(verbose: bool = True):
    resources = ["punkt", "wordnet", "omw-1.4"]
    for r in resources:
        try:
            nltk.data.find("tokenizers/punkt" if r == "punkt" else f"corpora/{r}")
        except LookupError:
            if verbose:
                print(f"[INFO] downloading NLTK resource: {r}")
            nltk.download(r, quiet=True)


def get_encoding():
    for name in ("o200k_harmony", "o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            continue
    raise RuntimeError("No suitable tiktoken encoding found. Please upgrade tiktoken.")


def process_data(prompts: pathlib.Path, completion: pathlib.Path, enc):
    data = []
    with open(prompts, "r", encoding="utf-8") as f_prompts, open(
        completion, "r", encoding="utf-8"
    ) as f_subm:
        prompt_lines = f_prompts.readlines()
        completion_lines = f_subm.readlines()
        prompt_lines = prompt_lines[1:] if prompt_lines else []

        for (prompt, token_ids) in zip(prompt_lines, completion_lines):
            ids = [int(x) for x in token_ids.strip().split()]
            entry = {
                "id": hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest(),
                "prompt": prompt.strip(),
                "completion": enc.decode(ids),
            }
            data.append(entry)
    return data


def coverage_report(subm: List[Dict], refs: List[Dict]):
    subm_ids = [d["id"] for d in subm]
    refs_ids = [d["id"] for d in refs]
    filtered = [x for x in subm_ids if x in set(refs_ids)]
    coverage = len(filtered) / max(1, len(refs_ids)) * 100.0
    print(f"[COVERAGE] Your submission covers {coverage:.1f}% of the prompts ({len(filtered)}/{len(refs_ids)})")
    return filtered


# ---------------------- Text metrics ---------------------- #
def distinct_n(text: str, n: int) -> float:
    toks = text.split()
    if len(toks) < n:
        return 0.0
    ngrams = set(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    return len(ngrams) / max(1, (len(toks) - n + 1))


def repetition_rate(text: str, min_run: int = 3) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    rep = 0
    run = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            run += 1
        else:
            if run >= min_run:
                rep += run
            run = 1
    if run >= min_run:
        rep += run
    return rep / len(toks)


# ---------------------- BERTScore (single & multi-GPU) ---------------------- #
def _bertscore_worker(args) -> Tuple[int, List[float], List[float], List[float]]:
    """
    Worker chạy trong process riêng.
    args = (start_idx, preds_chunk, refs_chunk, model_type, batch_size, device_str)
    Trả về: (start_idx, precision[], recall[], f1[])
    """
    start_idx, preds, refs, model_type, batch_size, device_str = args
    bscore = evaluate.load("bertscore")
    out = bscore.compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type=model_type,
        batch_size=max(1, int(batch_size)),
        device=device_str,
    )
    return start_idx, out["precision"], out["recall"], out["f1"]


def compute_bertscore_multigpu(
    preds: Sequence[str],
    refs: Sequence[str],
    model_type: str,
    batch_size: int,
    device_ids: List[int],
) -> Dict[str, List[float]]:
    """
    Nếu device_ids có >=2 phần tử, chạy song song nhiều GPU.
    Nếu device_ids rỗng => chạy CPU.
    Nếu device_ids có 1 phần tử => chạy một GPU đó.
    """
    n = len(preds)
    if n == 0:
        return {"precision": [], "recall": [], "f1": []}

    if not device_ids:
        # CPU
        bscore = evaluate.load("bertscore")
        out = bscore.compute(
            predictions=list(preds),
            references=list(refs),
            lang="en",
            model_type=model_type,
            batch_size=max(1, int(batch_size)),
            device="cpu",
        )
        return {"precision": out["precision"], "recall": out["recall"], "f1": out["f1"]}

    if len(device_ids) == 1:
        dev = device_ids[0]
        bscore = evaluate.load("bertscore")
        out = bscore.compute(
            predictions=list(preds),
            references=list(refs),
            lang="en",
            model_type=model_type,
            batch_size=max(1, int(batch_size)),
            device=f"cuda:{dev}",
        )
        return {"precision": out["precision"], "recall": out["recall"], "f1": out["f1"]}

    # Multi-GPU: chia shard
    num_workers = len(device_ids)
    shard_size = math.ceil(n / num_workers)
    tasks = []
    for rank, dev in enumerate(device_ids):
        start = rank * shard_size
        end = min(n, (rank + 1) * shard_size)
        if start >= end:
            continue
        preds_chunk = list(preds[start:end])
        refs_chunk = list(refs[start:end])
        tasks.append((start, preds_chunk, refs_chunk, model_type, batch_size, f"cuda:{dev}"))

    # Dùng spawn để tránh các vấn đề fork + HF transformers
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_bertscore_worker, tasks)

    # Ghép theo đúng vị trí
    precision = [0.0] * n
    recall = [0.0] * n
    f1 = [0.0] * n
    for start, p_list, r_list, f_list in results:
        L = len(p_list)
        precision[start:start + L] = p_list
        recall[start:start + L] = r_list
        f1[start:start + L] = f_list

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------- Main evaluation ---------------------- #
def eval_reference_based(
    ids: List[str],
    refs: List[Dict],
    subm: List[Dict],
    bs_model: str,
    bs_batch: int,
    device_ids: List[int],
):
    ensure_nltk_resources()

    # Chuẩn bị dữ liệu
    ids_set = set(ids)
    refs_dict = {d["id"]: d["completion"] for d in refs if d["id"] in ids_set}
    preds_dict = {d["id"]: d["completion"] for d in subm if d["id"] in ids_set}
    preds_completion = [preds_dict[i] for i in ids]
    gts_completion = [refs_dict[i] for i in ids]

    # METEOR (CPU)
    meteor = evaluate.load("meteor")
    meteor_value = meteor.compute(predictions=preds_completion, references=gts_completion)["meteor"]

    # BERTScore (CPU/1GPU/multi-GPU)
    log_devices(device_ids)
    bs = compute_bertscore_multigpu(
        preds=preds_completion,
        refs=gts_completion,
        model_type=bs_model,
        batch_size=bs_batch,
        device_ids=device_ids,
    )

    # Các metric phụ
    d1 = [distinct_n(p, 1) for p in preds_completion]
    d2 = [distinct_n(p, 2) for p in preds_completion]
    rep = [repetition_rate(p) for p in preds_completion]
    length = [len(p.split()) for p in preds_completion]

    agg = {
        "items": len(ids),
        "METEOR": round(float(meteor_value), 6),
        "BERTScore_F1": round(statistics.fmean(bs["f1"]), 6),
        "BERTScore_P": round(statistics.fmean(bs["precision"]), 6),
        "BERTScore_R": round(statistics.fmean(bs["recall"]), 6),
        "distinct1": round(statistics.fmean(d1), 6),
        "distinct2": round(statistics.fmean(d2), 6),
        "repetition_rate": round(statistics.fmean(rep), 6),
        "avg_len_tokens": round(statistics.fmean(length), 3),
    }
    thresholds = {
        "items": 32,
        "METEOR": 0.3,
        "BERTScore_F1": 0.8,
        "BERTScore_P": 0.8,
        "BERTScore_R": 0.8,
        "distinct1": 0.2,
        "distinct2": 0.3,
        "repetition_rate": 0.06,
    }

    df = pd.DataFrame(list(agg.items()), columns=["Metric", "Value"])
    df["Threshold"] = df["Metric"].map(thresholds).fillna("")

    def check_status(row):
        if row["Metric"] not in thresholds:
            return ""
        if row["Metric"] == "repetition_rate":
            return "PASS" if row["Value"] <= thresholds[row["Metric"]] else "FAIL"
        return "PASS" if row["Value"] >= thresholds[row["Metric"]] else "FAIL"

    df["Result"] = df.apply(check_status, axis=1)
    pd.set_option("display.colheader_justify", "right")
    print(df.to_string(index=False, justify="right"))


def main():
    args = parseCLI()

    # Thiết lập worker count
    device_ids = parse_devices_arg(args.devices)
    if args.workers.strip().lower() == "auto":
        workers = max(1, len(device_ids)) if device_ids else 1
    else:
        try:
            workers = max(1, int(args.workers))
        except Exception:
            workers = max(1, len(device_ids)) if device_ids else 1
    # cấu hình multiprocessing start method
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # đã được set ở nơi khác
        pass

    enc = get_encoding()
    subm = process_data(args.prompts, args.submission, enc)
    refs = process_data(args.prompts, args.references, enc)

    # Optional schema check
    # if not validate_references(refs, pathlib.Path("schema.json")) \
    #    or not validate_references(subm, pathlib.Path("schema.json")):
    #     print("Exiting due to schema validation failure.")
    #     return

    ids = coverage_report(subm, refs)
    if not ids:
        print("[ERROR] No overlapping IDs between submission and references.")
        return

    # Log tổng quan
    backend = detect_backend()
    if device_ids:
        names = [torch.cuda.get_device_name(i) for i in device_ids]
        print(f"[SUMMARY] Backend={backend} | GPUs={device_ids} -> {names} | workers={workers}")
    else:
        print(f"[SUMMARY] Backend={backend} | Using CPU | workers={workers}")

    eval_reference_based(ids, refs, subm, args.bs_model, args.bs_batch, device_ids)


if __name__ == "__main__":
    main()
