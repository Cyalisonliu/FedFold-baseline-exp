import torch
import numpy as np
import random


class Server:
    def __init__(self, global_model, workers, resource_limits, tau_e, tau_s, max_g, min_g, mu_s, mu_w, beta_s, beta_w):
        self.global_model = global_model
        self.workers = workers
        self.strong_workers = [worker for worker in workers if worker.device_type == 'S']
        self.weak_workers = [worker for worker in workers if worker.device_type == 'W']

        self.num_workers = len(workers)
        self.num_samples = [worker.num_samples for worker in workers]
        self.C_h = 0
        self.B_h = 0
        self.T_h = 0
        self.selected_workers = None

        # Initialize list to store updates from clients
        self.accum_global_params_list = []

        self.C, self.B = resource_limits
        self.tau_e = tau_e
        self.tau_s = tau_s
        self.max_gamma = max_g
        self.min_gamma = min_g

        # Device-specific parameters
        device_params = {
            'S': {'tau': int(tau_e), 'gamma': max_g},
            'W': {'tau': int(tau_s), 'gamma': min_g}
        }

        # Initialize tau_i_h based on worker type
        self.tau_i_h = []
        self.gamma_i_h = []
        for worker in self.workers:
            device_type = worker.device_type
            self.tau_i_h.append(device_params[device_type]['tau'])
            self.gamma_i_h.append(device_params[device_type]['gamma'])

        total_sqrt_tau = sum([np.sqrt(tau) for tau in self.tau_i_h])
        self.alpha_i_h = [self.num_workers * np.sqrt(self.tau_i_h[worker.id]) / total_sqrt_tau for worker in self.workers]

        # Initialize constants
        self.v_constant = 1 / self.tau_e
        self.mu_i_h = []
        self.beta_i_h = []

        time_params = {
            'S': {'mu': mu_s, 'beta': beta_s},
            'W': {'mu': mu_w, 'beta': beta_w}
        }
        for worker in self.workers:
            device_type = worker.device_type
            self.mu_i_h.append(time_params[device_type]['mu'])
            self.beta_i_h.append(time_params[device_type]['beta'])

        # Histories for tracking across rounds
        self.mu_history = [[] for _ in range(self.num_workers)]
        self.beta_history = [[] for _ in range(self.num_workers)]
        self.tau_history = [[] for _ in range(self.num_workers)]
        self.gamma_history = [[] for _ in range(self.num_workers)]
        self.compressed_weights = [{} for _ in range(self.num_workers)]
        self.stop_flag = False

    def download(self, worker):
        worker.model.load_state_dict({k: v.clone().detach() for k, v in self.global_model.state_dict().items()})
        print(
            f"Sending global model to worker {worker.id}: tau_h={self.tau_i_h[worker.id]}, gamma_h={self.gamma_i_h[worker.id]}")
        return self.tau_i_h[worker.id], self.gamma_i_h[worker.id]

    def upload(self, model_params, c_time, b_time, idx):
        update = {k: v.clone().detach() for k, v in model_params.items()}

        # Store the update
        self.accum_global_params_list.append(update)
        self.mu_i_h[idx] = c_time
        self.beta_i_h[idx] = b_time

    def aggregate(self):
        total_count = len(self.accum_global_params_list)
        n_selected = len(self.selected_workers)
        if total_count != n_selected:
            print("Not collect all updates from clients yet.")
            return

        delta = {name: torch.zeros_like(param) for name, param in self.global_model.state_dict().items()}

        for worker, update in zip(self.selected_workers, self.accum_global_params_list):
            weight = self.alpha_i_h[worker.id] / n_selected
            for k, v in update.items():
                delta[k] += weight * (v - self.global_model.state_dict()[k])

        # Update the global model
        for k, v in delta.items():
            self.global_model.state_dict()[k].add_(v)

        # Reset the list of accumulated updates
        self.accum_global_params_list = []

    def select_workers(self, num_select_workers, w_ratio=0.8):
        num_weak = int(num_select_workers * w_ratio)
        num_strong = num_select_workers - num_weak

        # Randomly select strong and weak workers
        selected_strong_workers = random.sample(self.strong_workers, num_strong)
        selected_weak_workers = random.sample(self.weak_workers, num_weak)

        # Combine the selected strong and weak workers
        self.selected_workers = selected_strong_workers + selected_weak_workers
        print([worker.device_type for worker in self.selected_workers])

        return self.selected_workers

    def update_alpha(self):
        sqrt_taus = [np.sqrt(tau) for tau in self.tau_i_h]
        total_sqrt_tau = sum(sqrt_taus)
        self.alpha_i_h = [self.num_workers * sqrt_tau / total_sqrt_tau for sqrt_tau in sqrt_taus]

    def find_fastest_worker_with_largest_tau(self):
        """
        Find the fastest worker with the largest tau value.

        Returns:
        - fastest_worker_id: The ID of the worker with the largest tau value.
        - largest_tau: The value of the largest tau among the workers.
        """
        fastest_time = np.inf
        fastest_worker_id = -1  # Variable to track the worker ID with the fastest time

        for worker in self.selected_workers:
            # Calculate the estimated completion time for the current worker based on tau and its history
            w_id = worker.id
            mu_h = self.mu_i_h[w_id]  # Estimated computation time
            beta_h = self.beta_i_h[w_id]  # Estimated communication time
            print(mu_h, beta_h)

            # Estimated completion time for this worker using its tau
            estimated_time = self.tau_i_h[w_id] * (mu_h + self.v_constant * beta_h)

            # Check if the current worker has the largest tau
            if estimated_time <= fastest_time:
                fastest_time = estimated_time
                fastest_worker_id = w_id

        print(f"Fastest worker ID: {fastest_worker_id}")
        return fastest_worker_id

    def run(self, h):
        print(f"Round {h} - Aggregating model updates and calculating resource consumption")

        # Update the global model based on workers' updates
        for worker in self.selected_workers:
            self.C_h += worker.c * self.tau_i_h[worker.id]  # Aggregate computing resource consumption
            self.B_h += worker.b * self.gamma_i_h[worker.id]  # Aggregate bandwidth resource consumption
        print(f"Computation budget: {self.C}, Current computation usage: {self.C_h} "
              f"Communication budget: {self.B}, Current communication usage: {self.B_h}")

        # Check if resource limits are exceeded
        if self.C_h > self.C or self.B_h > self.B:
            self.stop_flag = True  # Stop if resource limits are exceeded
        else:
            self.aggregate()

            # Calculate resource consumption
            t_h = max([self.tau_i_h[worker.id] * (self.mu_i_h[worker.id] + self.v_constant * self.beta_i_h[worker.id])
                       for worker in self.selected_workers])
            self.T_h += t_h

            # if h > 1:
            # Estimate mu and beta with moving average
            alpha = 0.5  # Smoothing factor for EMA
            for worker in self.selected_workers:
                w_id = worker.id
                # Update moving averages based on history
                if len(self.mu_history[w_id]) > 0:
                    prev_mu = self.mu_history[w_id][-1]
                    prev_beta = self.beta_history[w_id][-1]
                    self.mu_i_h[w_id] = alpha * self.mu_i_h[w_id] + (1 - alpha) * prev_mu
                    self.beta_i_h[w_id] = alpha * self.beta_i_h[w_id] + (1 - alpha) * prev_beta

                self.tau_history[w_id].append(self.tau_i_h[w_id])
                self.gamma_history[w_id].append(self.gamma_i_h[w_id])
                self.mu_history[w_id].append(self.mu_i_h[w_id])
                self.beta_history[w_id].append(self.beta_i_h[w_id])

            # Select worker finish the fastest and adjust tau
            l, tau_l_h = self.find_fastest_worker_with_largest_tau(), self.tau_e
            print("best tau", tau_l_h)

            # Update tau and calculate gamma for each worker
            for worker in self.workers:
                w_id = worker.id
                est_tau = np.floor(tau_l_h * (self.mu_i_h[l] + self.v_constant * self.beta_i_h[l]) /
                                  (self.mu_i_h[w_id] + self.v_constant * self.beta_i_h[w_id]))
                self.tau_i_h[w_id] = max(min(est_tau, self.tau_e), self.tau_s)

                self.gamma_i_h[w_id] = self.v_constant * self.tau_i_h[w_id]
            self.update_alpha()

        return self.stop_flag
