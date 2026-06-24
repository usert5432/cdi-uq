import logging
import torch

from cdi.bundled.leanbase.torch.funcs import seed_everything

from cdi.config.args import Args
from cdi.models import construct_model

LOGGER = logging.getLogger('cdi.eval')

def load_model(
    savedir, epoch, device = 'cuda', dtype = torch.float32
):
    args = Args.load(savedir)

    LOGGER.info('Starting evaluation: ')
    LOGGER.info(args.config.to_json(indent = 4))

    seed_everything(args.config.seed)

    model = construct_model(
        args.config, device, dtype, init_train = False, savedir = args.savedir
    )

    if epoch == -1:
        epoch = model.find_last_checkpoint_epoch()

    if epoch is None:
        model.load(None)
    elif epoch > 0:
        model.load(epoch)
    elif epoch == 0:
        model.epoch_start(epoch)

    model.eval()

    return (args, model)

def load_eval_dset(args, split = 'test'):
    from cdi.data.data   import construct_data_loader
    dl = construct_data_loader(args.config.data.eval, split = split)
    return dl

