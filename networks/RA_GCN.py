import torch
import torch.nn as nn
import networks as nw
import arguments as ag
import utils.BDGraphUtils as bu


class RA_GCN(nw.GCN, nw.AdversarialNetwork):
    """
    @article{ghorbani2022ra,
      title={RA-GCN: Graph convolutional network for disease prediction problems with imbalanced data},
      author={Ghorbani, Mahsa and Kazi, Anees and Baghshah, Mahdieh Soleymani and Rabiee, Hamid R and Navab, Nassir},
      journal={Medical Image Analysis},
      volume={75},
      pages={102272},
      year={2022},
      publisher={Elsevier}
    }
    """
    def __init__(self, params: nw.GCNParams):
        super().__init__(params)

    def adversarial_loss(self, scores: [[float]], labels: [int], weight_ratios: [[float]]) -> torch.Tensor:
        one_hot_labels = nn.functional.one_hot(labels)

        probabilities = torch.softmax(scores, dim=1)
        log_probs1 = torch.log(probabilities)
        loss_values = one_hot_labels * log_probs1

        weight_probabilities = torch.softmax(weight_ratios, dim=1)

        loss_values *= weight_probabilities

        loss_value = -torch.sum(loss_values)
        return loss_value


class ClassWeightGenerator(nw.GCN):
    def __init__(self, params: nw.GCNParams, masked_label: int):
        super().__init__(params)
        self.masked_label: int = masked_label
        self.masked_adjacency_graph: [[float]] = None

    def _construct_adjacency_graph(self, nodes: [[float]]) -> torch.Tensor:
        adjacency_graph = super()._construct_adjacency_graph(nodes)

        if not self.args.learning_mode:
            samples = self.all_samples[self.current_batch_indices]
            graph_masks = bu.build_masked_graph(samples, self.masked_label)
            graph_masks = torch.from_numpy(graph_masks).to(adjacency_graph.device)
            masked_adjacency_graph = graph_masks * adjacency_graph
        else:
            if self.masked_adjacency_graph is None:
                samples = self.all_samples
                graph_masks = bu.build_masked_graph(samples, self.masked_label)
                graph_masks = torch.from_numpy(graph_masks).to(adjacency_graph.device)
                self.masked_adjacency_graph = graph_masks * adjacency_graph
            masked_adjacency_graph = self.masked_adjacency_graph
        return masked_adjacency_graph


class RA_GCNWeightGenerator(nw.AdversarialNetwork):
    def __init__(self, params: nw.GCNParams):
        super().__init__(params)

        assert isinstance(params.args, ag.RA_GCNArgs)
        self.args = params.args
        self.class_num = params.args.class_num

        self.globalGCNs = nn.ModuleList()

        for i in range(self.class_num):
            globalGCN = ClassWeightGenerator(params, i)
            self.globalGCNs.append(globalGCN)

    def forward(self, x: [[float]]) -> [[float]]:
        num = self.current_batch_indices.shape[0] if not self.args.learning_mode else self.current_indices.shape[0]
        hidden_output2 = torch.zeros((self.class_num, num, self.class_num), dtype=torch.float).to(ag.Arguments.device)

        for i, globalGCN in enumerate(self.globalGCNs):
            assert isinstance(globalGCN, ClassWeightGenerator)
            globalGCN.current_indices = self.current_indices
            if not self.params.args.learning_mode:
                globalGCN.batch_index = self.batch_index

            hidden_output1 = globalGCN.forward(x)
            hidden_output2[i] = hidden_output1

        weight_ratios = torch.sum(hidden_output2, dim=0)
        return weight_ratios

    def adversarial_loss(self, weight_ratios: [[float]], labels: [int], scores: [[float]]) -> torch.Tensor:
        one_hot_labels = nn.functional.one_hot(labels)

        probabilities = torch.softmax(scores, dim=1)
        log_probs1 = torch.log(probabilities)
        loss_values = one_hot_labels * log_probs1

        weight_probabilities = torch.softmax(weight_ratios, dim=1)

        loss_values *= weight_probabilities

        loss_value = (-torch.sum(loss_values) - self.args.free_coefficient
                      * torch.sum(weight_probabilities * torch.log(weight_probabilities)))
        return loss_value
