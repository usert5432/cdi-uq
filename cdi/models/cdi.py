# pylint: disable=not-callable
import copy
import itertools

import torch

from cdi.bundled.leanbase.base.named_dict import NamedDict

from cdi.bundled.leanbase.torch.select     import select_optimizer, select_loss
from cdi.bundled.leanbase.torch.schedulers import select_scheduler

from cdi.bundled.diffusion      import select_diffusion_process
from cdi.bundled.diffusion.ddpm import generate_variance_schedule, DDPM

from cdi.nn.conditional_dm import construct_conditional_dm
from cdi.nn.encoder        import construct_encoder

from cdi.torch.funcs import update_ema_model

from .model_base import ModelBase

class CDI(ModelBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-ancestors

    def __init__(
        self, config, device, init_train, savedir, dtype, vsched,
        sched_metric  = 'val_l2',
        ema_momentum  = None,
        target_means  = None,
        target_stdevs = None,
        seed          = 0,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        shapes = next(iter(config.data.train.values())).shapes

        self._input_shape  = tuple(shapes[0])
        self._target_shape = tuple(shapes[1])
        self._ema_momentum = ema_momentum

        assert len(self._target_shape) == 1
        n_params = self._target_shape[0]

        self._prg = torch.Generator(device)
        self._prg.manual_seed(seed)

        self._seed_prg = torch.Generator(device)
        self._seed_prg.manual_seed(seed)

        self._vsched = generate_variance_schedule(vsched)
        self._dp     = DDPM(self._vsched, device, seed)
        self._seed   = seed

        super().__init__(config, device, init_train, savedir, dtype)
        self._sched_metric = sched_metric

        self.target_means = torch.zeros(
            (1, n_params), dtype = dtype, device = device
        )
        self.target_stdevs = torch.ones(
            (1, n_params), dtype = dtype, device = device
        )

        if target_means is not None:
            for (idx, mean) in enumerate(target_means):
                if mean is not None:
                    self.target_means[0, idx] = mean

        if target_stdevs is not None:
            for (idx, stdev) in enumerate(target_stdevs):
                if stdev is not None:
                    self.target_stdevs[0, idx] = stdev

    def _setup_data(self):
        # TODO: setup it properly
        return NamedDict(
            'dict_inputs',
            'dict_targets',
            'dict_noise',
            'dict_preds'
        )

    def _setup_losses(self):
        self.loss_fn = select_loss(self._config.losses['reg'])
        return NamedDict('reg')

    def _setup_nets(self):
        nets = {}

        nets['encoder'] = construct_encoder(
            self._config.nets['encoder'], self._input_shape, self._device
        )

        nets['gen'] = construct_conditional_dm(
            self._config.nets['gen'], self._target_shape, self._device
        )

        if (
                (self._ema_momentum is not None)
            and (self._ema_momentum > 0)
        ):
            nets['ema_encoder'] = copy.deepcopy(nets['encoder'])
            nets['ema_gen']     = copy.deepcopy(nets['gen'])

        return NamedDict(**{
            k : v.to(self._device) for (k, v) in nets.items()
        })

    def _setup_optimizers(self):
        optimizer = select_optimizer(
            itertools.chain(
                self._nets.gen    .parameters(),
                self._nets.encoder.parameters(),
            ),
            self._config.optimizers['main']
        )

        return NamedDict(main = optimizer)

    def _setup_schedulers(self):
        sched = select_scheduler(
            self._optimizers.main, self._config.schedulers['main'],
            compose = True
        )

        return NamedDict(main = sched)

    def _prepare_single_input(self, data):
        (input_data, targets) = data

        input_data = input_data.to(
            self._device, dtype = self._dtype, non_blocking = True
        )

        targets = targets.to(self._device, non_blocking = True)
        targets = (targets - self.target_means) / self.target_stdevs

        return (input_data, targets)

    def _set_inputs(self, data):
        dict_data = {
            k : self._prepare_single_input(v) for (k, v) in data.items()
        }

        self._data.dict_inputs  = { k : v[0] for (k, v) in dict_data.items() }
        self._data.dict_targets = { k : v[1] for (k, v) in dict_data.items() }

    def calc_loss_of_single_dataset(self, inputs, targets):
        n = len(inputs)

        t = torch.randint(
            low = 1, high = len(self._dp) + 1, size = (n, ),
            generator = self._prg,
            device    = self._device
        )

        noised_targets, noise = self._dp.forward_jump(t, targets)

        condition  = self._nets.encoder(inputs)
        pred_noise = self._nets.gen(noised_targets, condition, t)

        loss = self.loss_fn(pred_noise, noise)
        return loss

    def update_ema_states(self):
        if (
                (self._ema_momentum is None)
             or (self._ema_momentum <= 0)
        ):
            return

        update_ema_model(
            self._nets.ema_encoder, self._nets.encoder, self._ema_momentum
        )
        update_ema_model(
            self._nets.ema_gen, self._nets.gen, self._ema_momentum
        )

    def train_step(self):
        self.train()
        self._optimizers.main.zero_grad(set_to_none = True)

        n = len(self._data.dict_inputs)

        total_loss = 0

        for (k, input_data) in self._data.dict_inputs.items():
            targets   = self._data.dict_targets[k]
            curr_loss = self.calc_loss_of_single_dataset(input_data, targets)

            loss  = 1/n * curr_loss
            loss.backward()

            total_loss += loss.detach()
            self._losses[f'gen_{k}'] = curr_loss.detach()

        self._losses.gen = total_loss

        self._optimizers.main.step()

        self.step_scheduler(self._schedulers.main, event = 'batch')
        self.update_ema_states()

    def get_current_losses(self):
        result = {
            k : float(v.detach().cpu().mean())
                for (k, v) in self._losses.items() if v is not None
        }

        result['lr'] = self._optimizers.main.param_groups[0]['lr']

        return result

    @torch.no_grad()
    def diffuse_single_dataset(self, inputs, dp = None, subsample = None):
        if inputs is None:
            return None

        if dp is not None:
            dp = { 'name' : dp, 'prg' : self._dp.prg, 'vsched' : self._vsched }
            dp = select_diffusion_process(dp, self._device)
        else:
            dp = self._dp

        if subsample is not None:
            dp = dp.subsample(subsample)

        timesteps = list(reversed(range(1, len(dp)+1)))
        n         = len(inputs)

        result = dp.marginal_variance().sqrt() * torch.randn(
            (n, *self._target_shape),
            generator = self._seed_prg,
            device    = self._device,
            dtype     = self._dtype
        )

        if (
                (self._ema_momentum is not None)
            and (self._ema_momentum > 0)
        ):
            gen     = self._nets.ema_gen
            encoder = self._nets.ema_encoder
        else:
            gen     = self._nets.gen
            encoder = self._nets.encoder

        condition = encoder(inputs)

        for time in timesteps:
            t = time * torch.ones(n, device = self._device, dtype = torch.long)

            eps    = gen(result, condition, dp.map_time(t))
            result = dp.backward_step_given_eps(t, result, eps)

        return result

    @torch.no_grad()
    def predict_step(self):
        # pylint: disable=too-many-locals
        self.eval()
        preds = {}

        for (k, input_data) in self._data.dict_inputs.items():
            preds[k] = self.diffuse_single_dataset(input_data)

        self._data.dict_preds = preds

        dict_norm_preds = {
            k : self.target_means + v * self.target_stdevs
                for (k, v) in self._data.dict_preds.items()
        }
        dict_norm_targets = {
            k : self.target_means + v * self.target_stdevs
                for (k, v) in self._data.dict_targets.items()
        }

        result = {}

        for (k, preds) in dict_norm_preds.items():
            targets = dict_norm_targets[k]

            result[f'preds_{k}']   = preds
            result[f'targets_{k}'] = targets

        return result

    def reseed(self, seed):
        # TODO: add reseed interface to DiffusionProcess
        self._seed_prg.manual_seed(seed)
        self._dp._prg.manual_seed(seed)

    @torch.no_grad()
    def eval_step(self):
        # pylint: disable=too-many-locals
        self.eval()
        result = {}

        for (k, input_data) in self._data.dict_inputs.items():
            targets   = self._data.dict_targets[k]
            curr_loss = self.calc_loss_of_single_dataset(input_data, targets)

            result[f'gen_{k}'] = curr_loss

        result['gen'] = sum(result.values()) / len(result)

        return { k : float(v.cpu().item()) for (k, v) in result.items() }

    def epoch_end(self, metrics):
        self.step_scheduler(self._schedulers.main, 'epoch', metrics)
