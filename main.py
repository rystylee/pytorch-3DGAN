import os
import json

import torch

from config import get_config
from trainer import Trainer
from tester import Tester


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
        if config.config_path == '':
            raise Exception('[!] config_path is empty')

        with open(config.config_path) as f:
            config_dict = json.load(f)

        for k, v in config_dict.items():
            if not k == 'checkpoint_path':
                setattr(config, k, v)

        config_name = config.checkpoint_dir.split(os.sep)[1]
        test_dir = os.path.join(config.test_dir, config_name)
        config.img_dir = os.path.join(test_dir, config.img_dir)
        config.binary_dir = os.path.join(test_dir, config.binary_dir)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(config.img_dir, exist_ok=True)
        os.makedirs(config.binary_dir, exist_ok=True)

        print()
        print(config)
        print()

        tester = Tester(config)
        tester.test()


if __name__ == "__main__":
    main()
