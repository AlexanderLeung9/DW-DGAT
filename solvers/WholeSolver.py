import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer as too
import typing as t
import arguments as ag
import networks as nw
import solvers as sol


class WholeSolver(sol.SolverBase):
    """
    For transductive learning.
    """
    train_labels: torch.Tensor

    def __init__(self, params: sol.SolverBaseParams):
        super().__init__(params)
        self.train_labels: torch.Tensor
        self.all_data = params.all_data

    def _one_fold_train(self, fold: int, train_indices: [int], validate_indices: [int] = None) -> bool:
        # RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
        self.train_labels = torch.LongTensor(self.params.all_labels[train_indices]).to(ag.Arguments.device)
        return super()._one_fold_train(fold, train_indices, validate_indices)

    def _train(self, classifier: nn.Module, cls_optimizers: [too.Optimizer], generator: nn.Module, gen_optimizers: [too.Optimizer],
               epoch_No: int) -> (t.Optional[float], t.Optional[float]):
        classifier.train()
        for cls_optimizer in cls_optimizers:
            cls_optimizer.zero_grad()

        if self.args.cls_gen_train_ratio > 0 and isinstance(classifier, nw.AdversarialNetwork) and isinstance(generator, nw.AdversarialNetwork) and epoch_No % self.args.cls_gen_train_ratio == 0:
            for gen_optimizer in gen_optimizers:
                gen_optimizer.zero_grad()

            gen_scores = generator.forward(self.all_data)
            cls_scores = classifier.forward(self.all_data)

            gen_loss = generator.adversarial_loss(gen_scores, self.train_labels, cls_scores.detach())
            gen_loss_value = gen_loss.item()

            if np.isnan(gen_loss_value):
                self.logger.warning(f"Epoch {epoch_No}: gen_loss_value is {gen_loss_value}.")
                return None, None

            gen_loss.backward()
            for gen_optimizer in gen_optimizers:
                gen_optimizer.step()

            gen_scores = generator.forward(self.all_data).detach()
            cls_loss = classifier.adversarial_loss(cls_scores, self.train_labels, gen_scores)
        else:
            gen_loss_value = None
            cls_scores = classifier.forward(self.all_data)
            cls_loss = classifier.loss(cls_scores, self.train_labels)

        cls_loss_value = cls_loss.item()
        if np.isnan(cls_loss_value):
            self.logger.warning(f"Epoch {epoch_No}: cls_loss_value is {cls_loss_value}.")
            return None, None

        cls_loss.backward()
        for cls_optimizer in cls_optimizers:
            cls_optimizer.step()

        return cls_loss_value, gen_loss_value

    def _predict(self, network: nn.Module) -> ([int], [int], [[float]]):
        network.eval()

        ground_truth_labels = self.params.all_labels[network.current_indices]

        with torch.no_grad():
            scores = network.forward(self.all_data)
            test_scores = scores.detach()

        predicted_scores = test_scores.cpu().numpy()
        predicted_labels = np.argmax(predicted_scores, axis=1)

        return ground_truth_labels, predicted_labels, predicted_scores
