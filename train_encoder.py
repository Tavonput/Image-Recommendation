from encoder import SiameseTrainer

def main():
    data_path  = "data"
    save_path  = "encoder_params/120_epoch.pth"
    save_every = 10
    num_epochs = 120
    lr         = 0.0005

    trainer = SiameseTrainer(data_path, save_path, save_every, num_epochs, lr)

    trainer.train()

if __name__ == "__main__":
    main()