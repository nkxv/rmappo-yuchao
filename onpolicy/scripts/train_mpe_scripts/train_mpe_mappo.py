#!/usr/bin/env python3
"""
Train MAPPO on MPE environments using the optimal hyperparameters reported in
the MAPPO paper (Yu et al. 2022, "The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games"), Table 8.

Usage
-----
    python train_mpe_mappo.py                          # simple_spread (default)
    python train_mpe_mappo.py reference
    python train_mpe_mappo.py speaker_listener

Optional overrides forwarded directly to train_mpe.py
    --start_seed <int>  first seed to run (default: 1)
    --end_seed <int>    last seed inclusive (default: same as --start_seed)
    --user_name <str>   wandb entity name (required when wandb is on)
    --use_wandb         disable wandb, write TensorBoard logs instead
    --use_eval          enable periodic evaluation alongside training

Examples
--------
    # Single seed (default seed 1):
    python train_mpe_mappo.py spread

    # Specific seed:
    python train_mpe_mappo.py spread --start_seed 3

    # Seed sweep 1–5:
    python train_mpe_mappo.py spread --start_seed 1 --end_seed 5

    # Wandb run:
    python train_mpe_mappo.py reference --start_seed 1 --end_seed 3 --user_name my_entity
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Repo root: train_mpe_scripts/ → scripts/ → onpolicy/ → root
REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Shared optimal hyperparameters (MAPPO paper, Table 8) ────────────────────
#
# Key flag semantics (config.py uses store_false for several booleans):
#   --use_ReLU   store_false, default True  → passing it disables ReLU → Tanh
#   --use_wandb  store_false, default True  → passing it disables wandb → TensorBoard
#   --share_policy store_false, default True → passing it sets share_policy=False
#
BASE_ARGS = [
    "--env_name",               "MPE",
    "--algorithm_name",         "mappo",    # non-recurrent; best per MAPPO paper on MPE
    "--experiment_name",        "mappo_optimal",
    # Parallelism
    "--n_training_threads",     "1",
    "--n_rollout_threads",      "128",
    "--n_eval_rollout_threads", "8",
    # Rollout
    "--episode_length",         "25",
    "--num_mini_batch",         "1",
    # Optimiser
    "--lr",                     "7e-4",
    "--critic_lr",              "7e-4",
    # PPO
    "--clip_param",             "0.2",
    "--entropy_coef",           "0.01",
    "--value_loss_coef",        "1.0",
    "--max_grad_norm",          "10.0",
    "--gae_lambda",             "0.95",
    "--gamma",                  "0.99",
    # Network — Tanh activation (store_false flag disables ReLU)
    "--hidden_size",            "64",
    "--layer_N",                "1",
    "--gain",                   "0.01",
    "--use_ReLU",               # store_false → Tanh (paper default for MPE)
    # Logging / saving
    "--log_interval",           "5",
    "--save_interval",          "50",
    # Eval — run a deterministic evaluation pass every eval_interval episodes.
    # Metrics appear under eval/ in TensorBoard/wandb.
    # Change frequency with --eval_interval N (default here: every 25 episodes).
    "--use_eval",               # store_true → enables eval
    "--eval_interval",          "25",   # run eval every 25 training episodes
    "--eval_episodes",          "32",   # episodes per eval call (× n_eval_rollout_threads)
    # Disable wandb so the script works without a wandb account out of the box.
    # Remove this line (or pass --user_name) to re-enable wandb logging.
    "--use_wandb",              # store_false → TensorBoard
]

# ── Per-scenario configs ──────────────────────────────────────────────────────
#
# simple_speaker_listener requires the separated runner (share_policy=False).
# train_mpe.py hard-asserts this. Passing --share_policy on the CLI triggers
# store_false and sets share_policy=False (separated runner).
#
SCENARIOS = {
    "spread": {
        "scenario_name": "simple_spread",
        "num_agents":    3,
        "num_landmarks": 3,
        "num_env_steps": 20_000_000,
        "ppo_epoch":     10,           # 10 epochs for spread per paper
        "share_policy":  True,
    },
    "reference": {
        "scenario_name": "simple_reference",
        "num_agents":    2,
        "num_landmarks": 3,
        "num_env_steps": 3_000_000,
        "ppo_epoch":     15,
        "share_policy":  True,
    },
    "speaker_listener": {
        "scenario_name": "simple_speaker_listener",
        "num_agents":    2,
        "num_landmarks": 3,
        "num_env_steps": 2_000_000,
        "ppo_epoch":     15,
        "share_policy":  False,        # must use separated runner
    },
}


def build_args(scenario_key: str, seed: int, extra_flags: list) -> list:
    cfg = SCENARIOS[scenario_key]

    args = list(BASE_ARGS) + [
        "--scenario_name", cfg["scenario_name"],
        "--num_agents",    str(cfg["num_agents"]),
        "--num_landmarks", str(cfg["num_landmarks"]),
        "--num_env_steps", str(cfg["num_env_steps"]),
        "--ppo_epoch",     str(cfg["ppo_epoch"]),
        "--seed",          str(seed),
    ]

    if not cfg["share_policy"]:
        args.append("--share_policy")  # store_false → share_policy=False

    args.extend(extra_flags)
    return args


def run(scenario_key: str, seed: int = 1, extra_flags: list = None) -> None:
    cfg = SCENARIOS[scenario_key]
    args = build_args(scenario_key, seed=seed, extra_flags=extra_flags or [])

    print(
        f"\n{'='*60}\n"
        f"  Scenario : {cfg['scenario_name']}\n"
        f"  Seed     : {seed}\n"
        f"  Algorithm: mappo (non-recurrent)\n"
        f"  Steps    : {cfg['num_env_steps']:,}\n"
        f"  PPO ep.  : {cfg['ppo_epoch']}\n"
        f"{'='*60}\n"
    )

    cmd = [sys.executable, "-m", "onpolicy.scripts.train.train_mpe"] + args
    subprocess.run(
        cmd,
        check=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "scenario",
        nargs="?",
        default="spread",
        choices=list(SCENARIOS),
        help="MPE scenario to train (default: spread)",
    )
    p.add_argument(
        "--start_seed",
        type=int,
        default=1,
        help="First seed to run (default: 1)",
    )
    p.add_argument(
        "--end_seed",
        type=int,
        default=None,
        help="Last seed to run inclusive (default: same as --start_seed)",
    )
    known, passthrough = p.parse_known_args()

    end = known.end_seed if known.end_seed is not None else known.start_seed
    for seed in range(known.start_seed, end + 1):
        run(known.scenario, seed=seed, extra_flags=passthrough)
