import torch
from torch.optim import lr_scheduler

from evlearn.bundled.leanbase.torch.checkpoint import (
    save, load, find_last_checkpoint_epoch
)

PREFIX_NET       = 'net'
PREFIX_OPTIMIZER = 'opt'
PREFIX_SCHEDULER = 'sched'

class ModelBase:
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-ancestors

    def __init__(
        self, config, device, init_train, savedir, dtype
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self._epoch       = None
        self._device      = device
        self._dtype       = dtype
        self._init_train  = init_train
        self._config      = config
        self._savedir     = savedir

        self._data       = self._setup_data()
        self._nets       = self._setup_nets()
        self._losses     = self._setup_losses()
        self._optimizers = self._setup_optimizers()
        self._schedulers = self._setup_schedulers()

        self._train_state = init_train

    def train(self):
        self._train_state = True

        for nn in self._nets.values():
            nn.train()

    def eval(self):
        self._train_state = False

        for nn in self._nets.values():
            nn.eval()

    def _setup_data(self):
        raise NotImplementedError

    def _setup_losses(self):
        raise NotImplementedError

    def _setup_nets(self):
        raise NotImplementedError

    def _setup_optimizers(self):
        raise NotImplementedError

    def _setup_schedulers(self):
        raise NotImplementedError

    def set_inputs(self, data):
        for k in self._data:
            self._data[k] = None

        self._set_inputs(data)

    def _set_inputs(self, data):
        raise NotImplementedError

    def predict_step(self):
        raise NotImplementedError

    def step_scheduler(self, scheduler, event, metrics = None):
        if scheduler is None:
            return

        if event == 'batch':
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()

        elif event == 'epoch':
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                metric = metrics.get()['metric']
                scheduler.step(metric)

            elif isinstance(scheduler, lr_scheduler.OneCycleLR):
                pass

            else:
                scheduler.step()

    def train_step(self):
        raise NotImplementedError

    def get_current_losses(self):
        return {
            k : float(v.detach().cpu().mean())
                for (k, v) in self._losses.items() if v is not None
        }

    @torch.no_grad()
    def eval_step(self):
        raise NotImplementedError

    def save(self, epoch):
        save(self._nets,       self._savedir, PREFIX_NET,       epoch)
        save(self._optimizers, self._savedir, PREFIX_OPTIMIZER, epoch)
        save(self._schedulers, self._savedir, PREFIX_SCHEDULER, epoch)

    def load(self, epoch):
        self.epoch_start(epoch)

        load(self._nets, self._savedir, PREFIX_NET, epoch, self._device)

        if self._init_train:
            load(
                self._optimizers, self._savedir, PREFIX_OPTIMIZER, epoch,
                self._device
            )
            load(
                self._schedulers, self._savedir, PREFIX_SCHEDULER, epoch,
                self._device
            )

    @property
    def current_epoch(self):
        return self._epoch

    def find_last_checkpoint_epoch(self):
        return find_last_checkpoint_epoch(self._savedir)

    def epoch_start(self, epoch):
        self._epoch = epoch
        self._epoch_start()

    def to_dtype(self, dtype):
        for k in self._nets:
            self._nets[k] = self._nets[k].to(dtype = dtype)

        self._dtype = dtype

    def compile(self, **kwargs):
        for m in self._nets.values():
            m.compile(**kwargs)

    def construct_inference_engine(self):
        raise NotImplementedError

    def _epoch_start(self):
        pass

    def eval_epoch_start(self):
        pass

    def epoch_end(self, metrics):
        pass

    def eval_epoch_end(self):
        pass

