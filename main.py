import os
import argparse
import yaml

from solver import Solver
from torch.backends import cudnn
from data_loader import get_loader


def make_train_directory(config):
    # Create directories if not exist.
    if not os.path.exists(config['TRAINING_CONFIG']['TRAIN_DIR']):
        os.makedirs(config['TRAINING_CONFIG']['TRAIN_DIR'])
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['LOG_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['LOG_DIR']))
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['SAMPLE_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['SAMPLE_DIR']))
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['RESULT_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['RESULT_DIR']))
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['MODEL_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['MODEL_DIR']))


def main(config):

    assert config['TRAINING_CONFIG']['MODE'] in ['train', 'test']

    cudnn.benchmark = True
    solver = Solver(config, get_loader(config))
    print('{} is started'.format(config['TRAINING_CONFIG']['MODE']))
    if config['TRAINING_CONFIG']['MODE'] == 'train':
        solver.train()
    elif config['TRAINING_CONFIG']['MODE'] == 'test':
        solver.test()
    print('{} is finished'.format(config['TRAINING_CONFIG']['MODE']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='specifies config yaml file')

    params = parser.parse_args()

    if os.path.exists(params.config):
        config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)
        make_train_directory(config)
        main(config)
    else:
        print("Please check your config yaml file")


