import os
import argparse

import yaml
import wandb

from ai_lib.helpers import *

def shutdown_handler(*_):
    print("ctrl-c invoked")
    exit(0)


def arg_parser():
    parser = argparse.ArgumentParser(description='IDK yet.')

    parser.add_argument("--wandb", type=bool, nargs='?',
                            const=True, default=False,
                            help="Log to W and B.")

    parser.add_argument("--debug", type=bool, nargs='?',
                            const=True, default=False,
                            help="Activate debug mode.")

    parser.add_argument("--display", type=bool, nargs='?',
                            const=True, default=False,
                            help="Set to display mode.")

    parser.add_argument("--train", type=bool, nargs='?',
                            const=True, default=False,
                            help="Train the network.")

    parser.add_argument("--shutdown", type=bool, nargs='?',
                            const=True, default=False,
                            help="Shutdown after training.")

    return parser.parse_args()


def parse_set_args():
    # parse arguments
    args = arg_parser()

    # parse settings
    set = None
    with open(os.path.join(os.path.dirname(__file__), 'settings.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        settings['debug'] = True if args.debug else False
        settings['device'] = helpers.get_device()
        set = argparse.Namespace(**settings)

    # weights and biases
    if args.wandb:
        if False:
            wandb.init(project='2d-world-models', entity='jjshoots', config=settings)

    return set, args
