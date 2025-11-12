import torch
import torch.optim as to
import typing as t
import logging as l
import enums as es
import arguments as ag
import utils.GraphUtils as gu
import utils.BDGraphUtils as pu
import datasets as ds
import networks as nw
import solvers as s


class HyLaNetSolParams(s.MLPSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.HyLaNetArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_network(self):
        if self.args.adj_graph_type == es.EAdjGraphType.Phenotype:
            adjacent_graph = pu.build_phenotype_graph(self.all_samples, "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Euclidean:
            adjacent_graph = gu.build_feature_graph(self.all_data.cpu().numpy(), "")
        elif self.args.adj_graph_type == es.EAdjGraphType.Unweighted:
            adjacent_graph = pu.build_RA_GCN_graph1(self.all_samples, "")
        else:
            raise NotImplementedError(f"adj_graph_type={self.args.adj_graph_type}")

        adjacent_graph = torch.from_numpy(adjacent_graph).to(ag.Arguments.device)
        self.all_data = self.__sgc_precompute(adjacent_graph, self.args.k_order)

        create_cls_params = nw.NetworkParams(self.args)
        self.create_cls_factory = HyLaNetFactory(create_cls_params)

    def __sgc_precompute(self, adjacent_graph, degree) -> [[float]]:
        nonzero_perc = []
        features = self.all_data
        if degree == 0:
            number_nonzero = torch.Tensor(features != 0).sum().item()
            percentage = (number_nonzero * 1.0 / features.size(0) / features.size(1) * 100.0)
            nonzero_perc.append("%.2f" % percentage)
            print("input order 0, return raw feature")
            return features, nonzero_perc
        for i in range(degree):
            features = torch.mm(adjacent_graph, features)
            number_nonzero = torch.Tensor(features != 0).sum().item()
            percentage = (number_nonzero * 1.0 / features.size(0) / features.size(1) * 100.0)
            nonzero_perc.append("%.2f" % percentage)

        self.logger.info(f"nonzero_perc during adjacency matrix pre-computations: {nonzero_perc}%")
        return features


class HyLaNetFactory(nw.NetworkFactory):
    def __init__(self, create_params: nw.NetworkParams):
        super().__init__(create_params, "")
        self.create_params = create_params

    def create_network(self) -> (nw.HyLaNet, t.Iterable[to.Optimizer], t.Iterable[to.lr_scheduler.MultiStepLR]):
        hy_ly = nw.HyLa(self.create_params)
        optimizer1 = nw.RiemannianSGD(hy_ly.optim_params(), self.create_params.args.optimizer.learning_rate)
        sgc = nw.SGC(self.create_params)
        optimizer2, scheduler2 = self._create_optimizer_and_scheduler(sgc.parameters(), self.create_params.args)
        network = nw.HyLaNet(hy_ly, sgc)
        return network, [optimizer1, optimizer2], [scheduler2] if scheduler2 is not None else []
