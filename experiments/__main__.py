from experiments.sft_model_trainer import SFTTrainer
from experiments.config import SFTConfig2
from experiments.runpod_utils import stop_runpod
import argparse
import subprocess
import sys
import traceback
from datetime import datetime

def create_parser(config_class):
    config = config_class()
    parser = argparse.ArgumentParser(description='Parser created automatically from config class attributes')
        
    attrs = [attr for attr in dir(config) if not attr.startswith('_')]
    
    for attr in attrs:
        value = getattr(config, attr)
        
        if callable(value):
            continue
            
        if isinstance(value, bool):
            if value:
                parser.add_argument(f'--no_{attr}', action='store_false', dest=attr,
                                   help=f'Disable {attr} (default: {value})')
            else:
                parser.add_argument(f'--{attr}', action='store_true',
                                   help=f'Enable {attr} (default: {value})')
        elif isinstance(value, int):
            parser.add_argument(f'--{attr}', type=int, default=value,
                               help=f'{attr} (default: {value})')
        elif isinstance(value, float):
            parser.add_argument(f'--{attr}', type=float, default=value,
                               help=f'{attr} (default: {value})')
        elif isinstance(value, str):
            parser.add_argument(f'--{attr}', type=str, default=value,
                               help=f'{attr} (default: {value})')

        # TODO: Computed values such as accumulation_steps might have weird behavior here
    return parser


def update_config_from_cmd_line_args(config, args):
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

def print_config(config):
    print("Configuration:")
    config_vars = {k: v for k, v in vars(config).items() 
                if not k.startswith('_') and not callable(v)}
    for key, value in sorted(config_vars.items()):
        print(f"  {key}: {value}")

def main():

    # agent = Trainer()
    # agent.model_weights_folder_name='model_weights'
    # agent.video_folder_name='cartpole-ppo-videos'
    # agent.train()
    # agent.save_model_weights()
    # # agent.demonstrate(10)    
    # agent.record(10)  


    try:
        config = SFTConfig2()
        parser = create_parser(config.__class__)
        config = update_config_from_cmd_line_args(config, parser.parse_args())

        print_config(config)

        SFTTrainer(config).train()
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
