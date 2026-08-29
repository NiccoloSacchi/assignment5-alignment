"""
Generic runner to submit and execute commands in parallel on Modal GPU
containers.

Usage Examples:

1. Run a single script:
uv run modal run scripts/modal_runner.py \
    --command "scripts/prompting_baselines.py --batch-size 16"

2. Run parallel sweeps across multiple seeds/configurations:
uv run modal run scripts/modal_runner.py \
    --command "scripts/grpo.py --seed 0 --lr 1e-5" \
    --command "scripts/grpo.py --seed 1 --lr 1e-5" \
    --command "scripts/grpo.py --seed 2 --lr 1e-5"
"""

import shlex
import argparse
from cs336_alignment.modal_utils import app, submit_commands


@app.local_entrypoint()
def modal_main(*argv: str) -> None:
    # Recognize and parse the --command flags into a list of commands
    # (list[str]).
    parser = argparse.ArgumentParser(
        description="Parse the --command flags.",
    )
    parser.add_argument("--command", action="append", required=True)
    args = parser.parse_args(list(argv))
    commands = [["python", "-u"] + shlex.split(c) for c in args.command]
    submit_commands(commands)
