import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
import torch.optim.optimizer as too
import torch_geometric.data as tgd
import tqdm
import typing as t
import arguments as ag
import solvers as sol
import networks as nw


class BatchSolver(sol.SolverBase):
    """
    For inductive learning.
    """
    def __init__(self, params: sol.SolverBaseParams):
        super().__init__(params)
        self.dataset = params.dataset

    @staticmethod
    def __custom_collate(batch):
        item = batch[0]
        if isinstance(item, tuple):
            if isinstance(item[1], int):
                data = [item[0] for item in batch]
                if isinstance(item[0], torch.Tensor):
                    data = torch.stack(data)

                labels = [item[1] for item in batch]
                labels = torch.LongTensor(labels)
                return data, labels

        assert False, "Invalid data type!"

    def _train(self, classifier: nn.Module, cls_optimizers: [too.Optimizer], generator: nn.Module, gen_optimizers: [too.Optimizer],
               epoch_No: int) -> (t.Optional[float], t.Optional[float]):
        subset = tud.Subset(self.dataset, classifier.current_indices)
        data_loader = tud.DataLoader(subset, batch_size=self.args.batch_size, shuffle=False, collate_fn=BatchSolver.__custom_collate)
        classifier.train()

        cls_loss_value = 0.0
        gen_loss_value = 0.0
        i = -1
        progress_bar = tqdm.tqdm(enumerate(data_loader))
        for i, (batch_x, batch_y) in progress_bar:
            assert isinstance(batch_y, torch.LongTensor)

            if batch_y.size(0) == 1:
                i -= 1
                continue

            for cls_optimizer in cls_optimizers:
                cls_optimizer.zero_grad()

            if isinstance(classifier, nw.NetworkBase):
                classifier.batch_index = i
            if isinstance(generator, nw.NetworkBase):
                generator.batch_index = i

            if isinstance(batch_x, torch.Tensor):
                batch_x = batch_x.to(ag.Arguments.device)
            elif isinstance(batch_x, list) and isinstance(batch_x[0], tgd.Data):
                batch_x = tgd.Batch.from_data_list(batch_x)
            labels = batch_y.to(ag.Arguments.device)

            if self.args.cls_gen_train_ratio > 0 and isinstance(classifier, nw.AdversarialNetwork) and isinstance(generator, nw.AdversarialNetwork) and epoch_No % self.args.cls_gen_train_ratio == 0:
                for gen_optimizer in gen_optimizers:
                    gen_optimizer.zero_grad()

                gen_scores = generator.forward(batch_x)
                cls_scores = classifier.forward(batch_x)

                gen_loss = generator.adversarial_loss(gen_scores, labels, cls_scores.detach())
                gen_loss_value2 = gen_loss.item()

                if np.isnan(gen_loss_value2):
                    self.logger.warning(f"Epoch {epoch_No}, Iteration {i+1}: gen_loss_value is {gen_loss_value2}.")
                    return None, None

                gen_loss.backward()
                for gen_optimizer in gen_optimizers:
                    gen_optimizer.step()

                gen_scores = generator.forward(batch_x).detach()
                cls_loss = classifier.adversarial_loss(cls_scores, labels, gen_scores)
            else:
                gen_loss_value2 = None
                cls_scores = classifier.forward(batch_x)
                cls_loss = classifier.loss(cls_scores, labels)

            loss_value = cls_loss.item()
            if np.isnan(loss_value):
                self.logger.warning(f"Epoch {epoch_No}, Iteration {i+1}: loss_value is {loss_value}.")
                return None, None

            cls_loss_value += loss_value
            if gen_loss_value2 is None:
                gen_loss_value = None
            else:
                gen_loss_value += gen_loss_value2

            cls_loss.backward()
            for cls_optimizer in cls_optimizers:
                cls_optimizer.step()

            if self.args.print_every_iterations > 0 and (i+1) % self.args.print_every_iterations == 0:
                msg = f"Epoch {epoch_No}, Iteration {i+1}: loss_value={loss_value:.5f}"
                self.logger.info(msg)
            else:
                progress_bar.set_postfix({"loss_value": loss_value})

        cls_loss_value /= (i + 1)
        if gen_loss_value is not None:
            gen_loss_value /= (i + 1)
        return cls_loss_value, gen_loss_value

    def _predict(self, network: nn.Module) -> ([int], [int], [[float]]):
        subset = tud.Subset(self.dataset, network.current_indices)
        data_loader = tud.DataLoader(subset, batch_size=self.args.batch_size, shuffle=False, collate_fn=BatchSolver.__custom_collate)
        network.eval()

        predicted_scores = []
        ground_truth_labels = []
        # total_time = 0

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                if isinstance(network, nw.NetworkBase):
                    network.batch_index = i

                if isinstance(batch_x, torch.Tensor):
                    batch_x = batch_x.to(ag.Arguments.device)
                elif isinstance(batch_x, list) and isinstance(batch_x[0], tgd.Data):
                    batch_x = tgd.Batch.from_data_list(batch_x)

                # start_time = time.time()
                scores = network.forward(batch_x)
                # end_time = time.time()
                # total_time += end_time - start_time

                scores2 = scores.cpu().numpy()
                predicted_scores.append(scores2)

                labels = batch_y.numpy()
                ground_truth_labels.append(labels)

        ground_truth_labels = np.hstack(ground_truth_labels)
        predicted_scores = np.vstack(predicted_scores)
        predicted_labels = np.argmax(predicted_scores, axis=1)

        # avg_time = total_time / (i+1)
        # self.logger.info(f"average inference time={avg_time}s")

        return ground_truth_labels, predicted_labels, predicted_scores

    @staticmethod
    def __has_batchNorm(model: nn.Module) -> bool:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return True
        return False
