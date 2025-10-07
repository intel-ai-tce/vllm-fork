#!/usr/bin/env python3
import os
import csv
import argparse
from typing import List, Tuple

# Requires: pip install ruamel.yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# Import CPU_Binding directly from sibling cpu_binding.py
from cpu_binding import CPU_Binding

SERVICE_NAME = "vllm-server"     # single service
XSET_NAME    = "vllm_server_cpu" # x-sets key/anchor

REQUIRED_COLUMNS = ["model_id", "input length", "output length", "world_size", "num_allocated_cpu"]

def parse_int(v: str, name: str) -> int:
    try:
        return int(v)
    except Exception:
        raise ValueError(f"Invalid integer for {name!r}: {v!r}")

def pick_row_by_model(rows: List[dict], model: str) -> dict:
    matches = [r for r in rows if r.get("model_id", "").strip() == model]
    if not matches:
        available = ", ".join(sorted({r.get('model_id','') for r in rows}))
        raise ValueError(f"MODEL '{model}' not found in CSV. Available: {available}")
    return matches[0]

def build_cpuset_and_limit(world_size: int, num_alloc: int) -> Tuple[str, str]:
    cpus_list = ''
    for rank in range(world_size):
        #inst = CPU_Binding(world_size, rank, num_alloc)
        cpu_binder = CPU_Binding(world_size,rank, num_alloc)
        rank_to_cpus = cpu_binder.get_cpus_id_binding_based_on_numa_nodes()
        if cpus_list != '':
            cpus_list += ','
        cpus_list += rank_to_cpus
    print(cpus_list)
    return cpus_list

def main():
    ap = argparse.ArgumentParser(description="Generate override docker-compose YAML (x-sets) for single 'vllm-server'.")
    ap.add_argument("--settings", required=True,
                    help="CSV with columns: model_id,input length,output length,world_size,num_allocated_cpu")
    ap.add_argument("--output", required=True, help="Output compose YAML path")
    ap.add_argument("--compose-version", default="3.9")
    args = ap.parse_args()

    model = os.environ.get("MODEL")
    if not model:
        raise RuntimeError("Set environment variable MODEL to a model_id in the CSV (e.g., export MODEL='meta-llama/Llama-3.1-8B-Instruct').")

    with open(args.settings, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(col not in rows[0] for col in REQUIRED_COLUMNS):
        found = list(rows[0].keys()) if rows else "EMPTY CSV"
        raise ValueError(f"CSV missing required headers {REQUIRED_COLUMNS}. Found: {found}")

    row = pick_row_by_model(rows, model)
    world_size = parse_int(row["world_size"], "world_size")
    num_alloc  = parse_int(row["num_allocated_cpu"], "num_allocated_cpu")

    print(world_size)
    print(num_alloc)
    cpuset_csv = build_cpuset_and_limit(world_size, num_alloc)

    yaml = YAML()
    yaml.preserve_quotes = True

    root = CommentedMap()
    root["version"] = args.compose_version

    # x-sets anchor
    xsets = CommentedMap()
    root["x-sets"] = xsets
    block = CommentedMap()
    block["cpuset"] = cpuset_csv
    deploy = CommentedMap()
    deploy["resources"] = {"limits": {"cpus": num_alloc}}
    block["deploy"] = deploy
    xsets[XSET_NAME] = block
    block.yaml_set_anchor(XSET_NAME, always_dump=True)

    # single service merging the x-sets block
    services = CommentedMap()
    root["services"] = services
    merged = CommentedMap()
    merged.yaml_set_merge([block])
    services[SERVICE_NAME] = merged

    with open(args.output, "w") as f:
        yaml.dump(root, f)

    print(f"Wrote {args.output} for MODEL={model} (world_size={world_size}, num_allocated_cpu={num_alloc})")

if __name__ == "__main__":
    main()

