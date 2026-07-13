"""Collect a versioned state buffer from configured policy pairings."""

from __future__ import annotations

import argparse
import json

from src.state_augmentation.collection import (
    collect_state_buffer,
    load_state_collection_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="State-collection YAML")
    args = parser.parse_args()
    result = collect_state_buffer(load_state_collection_config(args.config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
