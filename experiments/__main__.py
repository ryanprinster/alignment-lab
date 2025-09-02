from experiments.sft_model_trainer import SFTTrainer

def main():

    # agent = Trainer()
    # agent.model_weights_folder_name='model_weights'
    # agent.video_folder_name='cartpole-ppo-videos'
    # agent.train()
    # agent.save_model_weights()
    # # agent.demonstrate(10)    
    # agent.record(10)  

    SFTTrainer().train()

if __name__ == "__main__":
    main()
