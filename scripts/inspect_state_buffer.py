"""Inspect and fully validate a versioned state-buffer artifact."""

from __future__ import annotations

import argparse
import json

from src.environment import build_env
from src.experiment_config import load_runtime_config
from src.state_augmentation.buffer import inspect_state_buffer, load_state_buffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", required=True, help="State-buffer JSON or JSON.GZ")
    parser.add_argument(
        "--environment-config",
        default=None,
        help="Optional YAML whose environment section must match the buffer",
    )
    args = parser.parse_args()

    env = None
    environment_config = None
    if args.environment_config is not None:
        runtime = load_runtime_config(args.environment_config)
        environment_config = runtime.get("environment")
        if not isinstance(environment_config, dict):
            raise ValueError("Environment config must contain an environment mapping")
        env = build_env(environment_config)
    buffer = load_state_buffer(
        args.buffer,
        env=env,
        environment_config=environment_config,
    )
    print(json.dumps({"status": "valid", **inspect_state_buffer(buffer)}, indent=2))


if __name__ == "__main__":
    main()
