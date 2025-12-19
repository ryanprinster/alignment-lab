import os

# Helps manage reserved memory fragments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # NOTE: make sure this is imported before torch
)
# os.environ['HF_HOME'] = '/workspace/alignment-lab/hf_cache'


import argparse
import importlib
import inspect
import sys
import traceback
import typing
from datetime import datetime
from pathlib import Path

from experiments.runpod_utils import stop_runpod


def create_config_parser(config_class):
    config = config_class()
    parser = argparse.ArgumentParser(
        description="Parser created automatically from config class attributes"
    )

    attrs = [attr for attr in dir(config) if not attr.startswith("_")]

    for attr in attrs:
        value = getattr(config, attr)

        if callable(value):
            continue

        if isinstance(value, bool):
            if value:
                parser.add_argument(
                    f"--no_{attr}",
                    action="store_false",
                    dest=attr,
                    help=f"Disable {attr} (default: {value})",
                )
            else:
                parser.add_argument(
                    f"--{attr}",
                    action="store_true",
                    help=f"Enable {attr} (default: {value})",
                )
        elif isinstance(value, int):
            parser.add_argument(
                f"--{attr}", type=int, default=value, help=f"{attr} (default: {value})"
            )
        elif isinstance(value, float):
            parser.add_argument(
                f"--{attr}",
                type=float,
                default=value,
                help=f"{attr} (default: {value})",
            )
        elif isinstance(value, str):
            parser.add_argument(
                f"--{attr}", type=str, default=value, help=f"{attr} (default: {value})"
            )

    return parser


def print_config(config):
    print("Configuration:")
    config_vars = {k: v for k, v in vars(config).items() if not callable(v)}
    for key, value in sorted(config_vars.items()):
        print(f"  {key}: {value}")


def parse_cli_args():
    if len(sys.argv) < 5 or sys.argv[3] != "--config":
        raise ValueError(
            "Usage: python -m experiments <TrainerName> <method> --config <ConfigName> [options]"
        )

    trainer_name = sys.argv[1]
    method_name = sys.argv[2]
    config_name = sys.argv[4]
    config_args = sys.argv[5:]  # Everything after the required --config ConfigName

    return trainer_name, method_name, config_name, config_args


def get_trainer_class(trainer_name):

    available_trainers = {}
    trainers_dir = Path("experiments/trainers")

    # Find available trainers
    for file_path in trainers_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue

        module_name = f"experiments.trainers.{file_path.stem}"
        module = importlib.import_module(module_name)

        for name, cls in inspect.getmembers(module, inspect.isclass):
            if hasattr(cls, "__bases__") and any(
                base.__name__ == "BaseTrainer" for base in cls.__mro__
            ):
                available_trainers[name] = cls

    if trainer_name not in available_trainers:
        raise ValueError(
            f"Unknown trainer '{trainer_name}'. Available: {list(available_trainers.keys())}. Does your trainer class inhert from BaseTrainer?"
        )

    return available_trainers[trainer_name]


def get_config_class(trainer_class, config_name):

    # Get base_config_class for trainer_class
    try:
        type_hints = typing.get_type_hints(trainer_class.__init__)
        base_config_class = type_hints["config"]
    except (KeyError, AttributeError):
        raise ValueError(
            f"Trainer {trainer_class.__name__} must have a type-hinted 'config' parameter"
        )

    # Find all concrete subclasses of base_config_class
    available_configs = base_config_class.__subclasses__()

    if not available_configs:
        raise ValueError(f"No concrete config classes found for {base_config_class.__name__}")

    # Get the config_class corresponding to config_name
    config_class = next((c for c in available_configs if c.__name__ == config_name), None)
    if config_class is None:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {[c.__name__ for c in available_configs]}"
        )

    return config_class


def create_and_configure(config_class, config_args):
    # Create instance
    config = config_class()

    # Parse command line args and update config
    parser = create_config_parser(config_class)
    args = parser.parse_args(config_args)

    # Init config from cmd line args
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.compile()

    print_config(config)

    return config


def get_method(trainer, method_name):
    if not hasattr(trainer, method_name) or method_name.startswith("_"):
        available = [
            m for m in dir(trainer) if not m.startswith("_") and callable(getattr(trainer, m))
        ]
        raise ValueError(f"Method '{method_name}' not available. Available: {available}")

    return getattr(trainer, method_name)


def main():

    try:
        trainer_name, method_name, config_name, config_args = parse_cli_args()

        trainer_class = get_trainer_class(trainer_name)
        config = create_and_configure(get_config_class(trainer_class, config_name), config_args)

        method = get_method(trainer_class(config), method_name)

        # Execute
        method()

        print("Training Done! Woweee")

    except Exception as e:
        # If there are errors, stop runpod to save $$ and log for debugging

        print(f"ERROR: {e}")
        traceback.print_exc()

        with open("crash_log.txt", "w") as f:
            f.write(f"Crashed at {datetime.now()}\n")
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())

        print("Stopping runpod...")
        stop_runpod()


if __name__ == "__main__":
    main()
