#!/usr/bin/env python
import argparse

from cdi.eval.eval import load_model

DEVICE = 'cpu'

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Print model architecture')

    parser.add_argument(
        'model',
        metavar = 'MODEL',
        help    = 'model directory',
        type    = str,
    )

    parser.add_argument(
        '-e', '--epoch',
        default = -1,
        dest    = 'epoch',
        help    = 'epoch',
        type    = int,
    )

    parser.add_argument(
        '--device',
        default = DEVICE,
        dest    = 'device',
        help    = 'device used to load the model',
        type    = str,
    )

    return parser.parse_args()

def count_parameters(net):
    return sum(param.numel() for param in net.parameters())

def main():
    cmdargs = parse_cmdargs()

    _args, model = load_model(
        cmdargs.model, epoch = cmdargs.epoch, device = cmdargs.device
    )

    for name, net in model._nets.items():
        print(net)
        print(
            '[Network %s] Total number of parameters : %.3f M' % (
                name, count_parameters(net) / 1e6
            )
        )

if __name__ == '__main__':
    main()
