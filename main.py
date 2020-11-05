import torch

from config import get_config
from trainer import Trainer


def main():
    config = get_config()
    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if config.mode == 'train':
        print()
        print(config)
        print()

        if not config.device == 'cpu':
            torch.backends.cudnn.benchmark = True

        trainer = Trainer(config)
        with torch.autograd.detect_anomaly():
            trainer.train()

    elif config.mode == 'test':
        pass


if __name__ == "__main__":
    main()
