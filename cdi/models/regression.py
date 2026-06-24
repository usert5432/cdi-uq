# pylint: disable=not-callable
import torch
import torch.nn.functional as F

from cdi.bundled.leanbase.base.named_dict import NamedDict

from cdi.bundled.leanbase.torch.select import select_optimizer, select_loss
from cdi.bundled.leanbase.torch.schedulers import select_scheduler

from cdi.nn.regressor import construct_regressor
from .model_base import ModelBase

class Regressor(ModelBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-ancestors

    def __init__(
        self, config, device, init_train, savedir, dtype,
        sched_metric  = 'val_l2',
        target_means  = None,
        target_stdevs = None,
    ):
        # pylint: disable=too-many-arguments
        shapes = next(iter(config.data.train.values())).shapes

        self._input_shape  = tuple(shapes[0])
        self._target_shape = tuple(shapes[1])

        assert len(self._target_shape) == 1
        n_params = self._target_shape[0]

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
        return NamedDict('dict_inputs', 'dict_targets', 'dict_preds')

    def _setup_losses(self):
        self.loss_fn = select_loss(self._config.losses['reg'])
        return NamedDict('reg')

    def _setup_nets(self):
        nets = {}

        nets['reg'] = construct_regressor(
            self._config.nets['reg'],
            self._input_shape, self._target_shape, self._device
        )

        return NamedDict(**{
            k : v.to(self._device) for (k, v) in nets.items()
        })

    def _setup_optimizers(self):
        optimizer = select_optimizer(
            self._nets.reg.parameters(),
            self._config.optimizers['reg']
        )

        return NamedDict(reg = optimizer)

    def _setup_schedulers(self):
        sched = select_scheduler(
            self._optimizers.reg, self._config.schedulers['reg'],
            compose = True
        )

        return NamedDict(reg = sched)

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

    def forward(self):
        self._data.dict_preds = {
            k : self._nets.reg(input_data)
                for (k, input_data) in self._data.dict_inputs.items()
        }

    def train_step(self):
        self.train()
        self._optimizers.reg.zero_grad(set_to_none = True)

        #self.forward()

        n = len(self._data.dict_inputs)

        total_loss = 0
        dict_preds = {}

        for (k, input_data) in self._data.dict_inputs.items():
            targets = self._data.dict_targets[k]

            preds = self._nets.reg(input_data)
            loss  = 1/n * self.loss_fn(preds, targets)
            loss.backward()

            total_loss    += loss.detach()
            dict_preds[k]  = preds.detach()

            self._losses[f'reg_{k}'] = n * loss.detach()

        self._optimizers.reg.step()

        self._losses.reg      = total_loss
        self._data.dict_preds = dict_preds

        self.step_scheduler(self._schedulers.reg, event = 'batch')

    def get_current_losses(self):
        result = {
            k : float(v.detach().cpu().mean())
                for (k, v) in self._losses.items() if v is not None
        }

        result['lr'] = self._optimizers.reg.param_groups[0]['lr']

        return result

    @torch.no_grad()
    def eval_step(self):
        # pylint: disable=too-many-locals
        self.eval()
        self.forward()

        result = {}

        l1 = 0
        l2 = 0

        n = len(self._data.dict_preds)

        for (k, preds) in self._data.dict_preds.items():
            targets = self._data.dict_targets[k]

            result[f'l1_{k}'] = F.l1_loss (preds, targets)
            result[f'l2_{k}'] = F.mse_loss(preds, targets)

            l1 += result[f'l1_{k}']
            l2 += result[f'l2_{k}']

        result['l1'] = l1 / n
        result['l2'] = l2 / n

        return { k : float(v.cpu().item()) for (k, v) in result.items() }

    @torch.no_grad()
    def predict_step(self, scale = None, shift = None):
        # pylint: disable=too-many-locals
        self.eval()

        for k in self._data.dict_inputs:
            if scale is not None:
                self._data.dict_inputs[k] = scale * self._data.dict_inputs[k]

            if shift is not None:
                self._data.dict_inputs[k] = self._data.dict_inputs[k] + shift

        self.forward()

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

    def epoch_end(self, metrics):
        self.step_scheduler(self._schedulers.reg, 'epoch', metrics)
