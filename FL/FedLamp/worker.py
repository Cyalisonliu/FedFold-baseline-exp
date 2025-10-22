import time
import torch
import numpy as np

import torch


class Worker:
    def __init__(self, worker_id, device_type, model, data_loader, val_loader, resource_limits,
                 loss_fn, computing_params, bandwidth_params,
                 learning_rate, momentum, weight_decay, learning_rate_decay, device):
        self.id = worker_id
        self.device_type = device_type
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.val_loader = val_loader
        # Calculate the number of samples from the data loader
        self.num_samples = len(self.data_loader.dataset)

        # Estimated computing resource consumption c, bandwidth resource consumption b
        self.c, self.b = resource_limits
        self.mean_compute_time, self.stddev_compute_time = computing_params
        self.mean_communicate_time, self.stddev_communicate_time = bandwidth_params

        self.train_accuracies = []  # Store training accuracy for each epoch
        self.train_losses = []  # Store training loss for each epoch

        # Error compensation vector
        self.compression_error = {name: torch.zeros_like(param) for name, param in self.model.state_dict().items()}

        # Optimizer settings
        self.loss_fn = loss_fn
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
        #                                  weight_decay=weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=learning_rate_decay)

    def simulate_time(self):
        # return (np.random.normal(self.mean_compute_time, self.stddev_compute_time),
        #         np.random.normal(self.mean_communicate_time, self.stddev_communicate_time) * 1e6 / 8)
        return self.mean_compute_time, self.mean_communicate_time * 1e6 / 8

    def local_update(self, tau_h):
        total_loss = 0.0
        correct = 0
        total = 0
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                    weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

        for epoch in range(tau_h):
            for data, label in self.data_loader:
                data, label = data.to(self.device), label.to(self.device)
                self.model.train()
                logits = self.model(data)
                loss = self.loss_fn(logits, label)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)

            self.train_losses.append(total_loss / len(self.data_loader))
            self.train_accuracies.append(100. * correct / total)

            torch.cuda.empty_cache()

        # Update scheduler after each local update
        # scheduler.step()

        return self.train_accuracies, self.train_losses

    def compress_model_top_k(self, k_ratio):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Skip None or empty parameters
                if param is not None and param.numel() > 0:
                    # Flatten the parameter tensor to a 1D vector for easier manipulation
                    param_flat = param.view(-1)
                    original_param = param_flat.clone()

                    # Add the accumulated error back to the parameter before compression Alg1 20, 21
                    param_flat += self.compression_error[name].view(-1)

                    # Calculate the number of elements to retain based on the k_ratio
                    k = int(len(param_flat) * k_ratio)
                    if k == 0:
                        continue

                    # Get the indices of the top-K elements based on absolute values
                    _, topk_indices = torch.topk(param_flat.abs(), k, sorted=False)

                    # Retain the top-K values
                    compressed_param = torch.zeros_like(param_flat)
                    compressed_param[topk_indices] = param_flat[topk_indices]
                    # Update the parameter with the compressed values, and reshape to the original shape
                    param.copy_(compressed_param.view_as(param))

                    # Keep track of accumulated errors in memory
                    self.compression_error[name] += (original_param - compressed_param).view_as(param)

    # def validate(self):
    #     """Validation step on the validation set."""
    #     self.model.eval()  # Set the model to evaluation mode
    #     val_loss = 0.0
    #     correct = 0
    #     total = 0
    #
    #     with torch.no_grad():  # Disable gradient calculation for validation
    #         for data, label in self.val_loader:
    #             data, label = data.to(self.device), label.to(self.device)
    #             logits = self.model(data)
    #             loss = self.loss_fn(logits, label)
    #             val_loss += loss.item()
    #
    #             _, predicted = torch.max(logits, 1)
    #             correct += (predicted == label).sum().item()
    #             total += label.size(0)
    #
    #             del data, label, logits, loss
    #             torch.cuda.empty_cache()
    #
    #     avg_val_loss = val_loss / len(self.val_loader)
    #     accuracy = 100. * correct / total
    #     return avg_val_loss, accuracy
