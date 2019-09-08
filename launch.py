import os
from os.path import join, exists
import json
import argparse
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='',
                        help='configuration name defined in the checkpoint')
    parser.add_argument('--path', type=str, default='configs.json',
                        help='path to the configuration file')
    args = parser.parse_args()
    return args


def execute():
    args = parse_args()
    assert exists(args.path), 'configuration file does not exist'
    with open(args.path, 'r') as fp:
        config = json.load(fp)
        assert args.name in config, 'config name does not exist'
        cfg = config[args.name]
        cmd = ' '.join([' '.join([k, v]) for k, v in cfg.items()])
    os.system(cmd)


if __name__ == '__main__':
    execute()
