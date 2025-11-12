import torch.nn as nn
import torch.optim as to
import typing as t
import importlib
import arguments as ag
import networks as nw
import optimizers as om
import schedulers as sd


class NetworkFactory(object):
    def __init__(self, create_params: nw.NetworkParams, network_name: t.Optional[str] = None):
        super().__init__()
        self.create_params = create_params
        self.network_name = network_name if network_name is not None else create_params.args.net_name

    def _create_optimizer_and_scheduler(self, net_params: t.Iterator[nn.Parameter], args: ag.Arguments) -> (to.Optimizer, t.Optional[object]):
        if isinstance(args.optimizer, om.AdamParams):
            optimizer = to.Adam(net_params, args.optimizer.learning_rate, args.optimizer.betas, weight_decay=args.optimizer.weight_decay)
        elif isinstance(args.optimizer, om.SGDParams):
            optimizer = to.SGD(net_params, args.optimizer.learning_rate, args.optimizer.momentum, weight_decay=args.optimizer.weight_decay, nesterov=args.optimizer.nesterov)
        elif args.optimizer is not None:
            raise NotImplementedError(f"optimizer={args.optimizer}")
        else:
            raise ValueError("optimizer can not be None!")

        if isinstance(args.scheduler, sd.StepLRParams):
            scheduler = to.lr_scheduler.StepLR(optimizer, args.scheduler.step_size, args.scheduler.gamma, args.scheduler.last_epoch)
        elif isinstance(args.scheduler, sd.MultiStepLRParams):
            scheduler = to.lr_scheduler.MultiStepLR(optimizer, args.scheduler.milestones, args.scheduler.gamma, args.scheduler.last_epoch)
        elif isinstance(args.scheduler, sd.ExponentialLRParams):
            scheduler = to.lr_scheduler.ExponentialLR(optimizer, args.scheduler.gamma, args.scheduler.last_epoch)
        elif isinstance(args.scheduler, sd.LambdaLRParams):
            scheduler = to.lr_scheduler.LambdaLR(optimizer, args.scheduler.lr_lambda, args.scheduler.last_epoch)
        elif isinstance(args.scheduler, sd.CosineAnnealingLRParams):
            scheduler = to.lr_scheduler.CosineAnnealingLR(optimizer, args.scheduler.T_max, args.scheduler.eta_min)
        elif args.scheduler is not None:
            raise NotImplementedError(f"scheduler={args.scheduler}")
        else:
            scheduler = None
        return optimizer, scheduler

    def create_network(self) -> (nn.Module, [to.Adam], [to.lr_scheduler.MultiStepLR]):
        """
        It can be overwritten in subclasses.
        """
        module = importlib.import_module("networks")
        class_constructor = getattr(module, self.network_name)
        network = class_constructor(self.create_params)
        assert isinstance(network, nn.Module)
        optimizer, scheduler = self._create_optimizer_and_scheduler(network.parameters(), self.create_params.args)
        return network, [optimizer], [scheduler] if scheduler is not None else []
