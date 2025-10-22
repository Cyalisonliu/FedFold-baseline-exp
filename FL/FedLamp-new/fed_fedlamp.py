import copy
import torch
from utils import Compressor, Utils

import numpy as np
import random

cfg_fedlamp = {
    'max_tau': 5,
    'min_tau': 3,
    'max_gamma': 1.0,
    'min_gamma': 0.2,
    'weak_bw': 100,
    'strong_bw': 10,
}


def calculate_beta(d_type, model_size_in_bytes, _random=False):
    min_bandwidth_bytes = cfg_fedlamp['strong_bw'] * 1e6 / 8  # Convert to bytes per second
    max_bandwidth_bytes = cfg_fedlamp['weak_bw'] * 1e6 / 8  # Convert to bytes per second

    if _random:
        bw = np.random.uniform(min_bandwidth_bytes, max_bandwidth_bytes)
        return model_size_in_bytes / bw
    else:
        if d_type == 'S':
            return model_size_in_bytes / min_bandwidth_bytes
        else:
            return model_size_in_bytes / max_bandwidth_bytes


class Federation:
    def __init__(self, global_params, device_type, strong_comp, weak_comp, _random=False, n_device=100, n_selected=10):

        self.global_params = []
        self.accum_global_params = []
        self.cnt = []

        # store selected clients
        self.model_size = 0
        for param in global_params[0].values():
            self.model_size += param.numel() * param.element_size()
        print("Model size:", self.model_size)

        for i in range(len(global_params)):  # 8 small + 2 large
            # self.global_params.append({k: copy.deepcopy(v) for k, v in global_params[i].items()})
            self.global_params.append({k: v.detach().clone() for k, v in global_params[i].items()})
            self.accum_global_params.append({k: torch.zeros_like(v) for k, v in global_params[i].items()})
            self.cnt.append(0)

        # Device-specific parameters
        self.n_device = n_device
        self.n_selected = n_selected

        # for client selection
        self.strong_worker_ids = []
        self.weak_worker_ids = []
        for w_id, w_type in enumerate(device_type):
            if w_type == 'S':
                self.strong_worker_ids.append(w_id)
            else:
                self.weak_worker_ids.append(w_id)

        self.mu_i_h = []
        self.beta_i_h = []
        for w_id, w_type in enumerate(device_type):
            if w_type == 'S':
                self.mu_i_h.append(strong_comp)
            else:
                self.mu_i_h.append(weak_comp)
            beta = calculate_beta(w_type, self.model_size, _random)
            self.beta_i_h.append(beta)

        self.mu_history = [[] for _ in range(n_device)]
        self.beta_history = [[] for _ in range(n_device)]

        # Initialize tau_i_h based on worker type
        self.tau_i_h = []
        for w_type in device_type:
            if w_type == 'S':
                self.tau_i_h.append(cfg_fedlamp['max_tau'])
            else:
                self.tau_i_h.append(cfg_fedlamp['min_tau'])

        self.gamma_i_h = []
        for w_type in device_type:
            if w_type == 'S':
                self.gamma_i_h.append(cfg_fedlamp['max_gamma'])
            else:
                self.gamma_i_h.append(cfg_fedlamp['min_gamma'])

        total_sqrt_tau = sum([np.sqrt(tau) for tau in self.tau_i_h])
        self.alpha_i_h = [self.n_device * np.sqrt(self.tau_i_h[w_id]) / total_sqrt_tau for w_id in range(self.n_device)]

        # Initialize constants
        self.v_constant = 1 / (cfg_fedlamp['max_tau']+1)

    def download(self, local_params, idx):
        for k, v in self.global_params[idx].items():
            local_params[k] = v.detach().clone()

    def upload(self, local_params, idx, c_time, b_time, w_id):
        for k, v in local_params.items():
            self.accum_global_params[idx][k] += v * self.alpha_i_h[w_id]
        self.cnt[idx] += 1
        self.mu_i_h[w_id] = c_time
        self.beta_i_h[w_id] = b_time

    def aggregate(self):
        for idx in range(len(self.global_params)):
            if self.cnt[idx] != 0:
                print(f"index {idx} has {self.cnt[idx]} elements")
                for k, v in self.accum_global_params[idx].items():
                    self.global_params[idx][k] = v.detach().clone() / self.cnt[idx]
                    self.accum_global_params[idx][k] = torch.zeros_like(v)
                self.cnt[idx] = 0

    def find_fastest_worker_with_largest_tau(self, selected_workers):
        fastest_time = np.inf
        fastest_worker_id = -1  # Variable to track the worker ID with the fastest time

        for w_id in selected_workers:
            # Estimated completion time for this worker using its tau
            estimated_time = self.tau_i_h[w_id] * (self.mu_i_h[w_id] + self.v_constant * self.beta_i_h[w_id])

            # Check if the current worker has the largest tau
            if estimated_time <= fastest_time:
                fastest_time = estimated_time
                fastest_worker_id = w_id

        print(f"Fastest worker ID: {fastest_worker_id}")
        return fastest_worker_id

    def update_alpha(self):
        sqrt_taus = [np.sqrt(tau) for tau in self.tau_i_h]
        total_sqrt_tau = sum(sqrt_taus)
        self.alpha_i_h = [self.n_device * sqrt_tau / total_sqrt_tau for sqrt_tau in sqrt_taus]

    def select_workers(self, num_select_workers, w_ratio=0.8):
        num_weak = int(num_select_workers * w_ratio)
        num_strong = num_select_workers - num_weak

        # Randomly select strong and weak workers
        selected_strong_workers = random.sample(self.strong_worker_ids, num_strong)
        selected_weak_workers = random.sample(self.weak_worker_ids, num_weak)

        # Combine the selected strong and weak workers
        selected_workers = selected_strong_workers + selected_weak_workers

        # Estimate mu and beta with moving average
        alpha = 0.5  # Smoothing factor for EMA
        for w_id in selected_workers:
            if len(self.mu_history[w_id]) > 0:
                prev_mu = self.mu_history[w_id][-1]
                prev_beta = self.beta_history[w_id][-1]
                self.mu_i_h[w_id] = alpha * self.mu_i_h[w_id] + (1 - alpha) * prev_mu
                self.beta_i_h[w_id] = alpha * self.beta_i_h[w_id] + (1 - alpha) * prev_beta

            self.mu_history[w_id].append(self.mu_i_h[w_id])
            self.beta_history[w_id].append(self.beta_i_h[w_id])

        # Select worker finish the fastest and adjust tau
        l, tau_l_h = self.find_fastest_worker_with_largest_tau(selected_workers), cfg_fedlamp['max_tau']

        # Update tau and calculate gamma for each worker
        selected_taus = []
        selected_gammas = []
        for w_id in selected_workers:
            est_tau = np.floor(tau_l_h * (self.mu_i_h[l] + self.v_constant * self.beta_i_h[l]) /
                               (self.mu_i_h[w_id] + self.v_constant * self.beta_i_h[w_id]))
            tau = int(max(min(est_tau, cfg_fedlamp['max_tau']), cfg_fedlamp['min_tau']))
            self.tau_i_h[w_id] = tau
            selected_taus.append(tau)
            gamma = self.v_constant * self.tau_i_h[w_id]
            self.gamma_i_h[w_id] = gamma
            selected_gammas.append(gamma)
        self.update_alpha()

        return selected_workers, selected_taus, selected_gammas
