import copy
import torch
from utils import Compressor, Utils

import numpy as np
import pulp


class Federation:
    def __init__(self, global_params, client_data_sizes, n_device=100, n_selected=10):
        self.global_params = []
        self.accum_global_params = []
        self.cnt = []

        self.participation_counts = {i: 0 for i in range(n_device)}
        self.client_gradients = {i: None for i in range(n_device)}  # Initialize gradient history
        self.client_data_sizes = client_data_sizes

        # store selected clients
        self.n_device = n_device
        self.n_selected = n_selected
        self.model_size = 0
        for param in global_params[0].values():
            self.model_size += param.numel() * param.element_size()
        print("Model size:", self.model_size)

        for i in range(len(global_params)):  # 8 small + 2 large
            # self.global_params.append({k: copy.deepcopy(v) for k, v in global_params[i].items()})
            self.global_params.append({k: v.detach().clone() for k, v in global_params[i].items()})
            self.accum_global_params.append({k: torch.zeros_like(v) for k, v in global_params[i].items()})
            self.cnt.append(0)

    def download(self, local_params, idx):
        for k, v in self.global_params[idx].items():
            # local_params[k] = copy.deepcopy(v)
            local_params[k] = v.detach().clone()

    def upload(self, local_params, idx):
        # if is_svd:
        #     local_params = Compressor.reconstruct_svd(local_params)
        # hidden_size = [
        #     [4, 8, 16, 32],
        #     [4, 8, 16, 32],
        #     [8, 16, 32, 64],
        #     [16, 32, 64, 128],
        #     [32, 64, 128, 256],
        # ]

        for k, v in local_params.items():
            self.accum_global_params[idx][k] += v
            # print(v)
        self.cnt[idx] += 1

    def aggregate(self):
        for idx in range(len(self.global_params)):
            if self.cnt[idx] != 0:
                print(f"index {idx} has {self.cnt[idx]} elements")
                for k, v in self.accum_global_params[idx].items():
                    # self.global_params[idx][k] = copy.deepcopy(v) / self.cnt[idx]
                    self.global_params[idx][k] = v.detach().clone() / self.cnt[idx]
                    self.accum_global_params[idx][k] = torch.zeros_like(v)
                self.cnt[idx] = 0

    def select_clients_submodular(self):
        """
        Select clients using a submodular maximization approach based on participation history,
        data size, and actual gradients stored from previous rounds.
        """
        clients_gains = []
        selected_idx = []
        print(self.participation_counts)

        # Check if gradients are available for all clients
        all_gradients_available = all(self.client_gradients[client] is not None for client in range(self.n_device))

        for _ in range(self.n_selected):
            best_gain = -float('inf')
            best_client = None

            for client in range(self.n_device):
                if client not in selected_idx:
                    if all_gradients_available:
                        gradient_distance = sum(
                            torch.norm(self.client_gradients[client][k]) for k in self.client_gradients[client].keys())
                        candidate_gain = gradient_distance * self.client_data_sizes[client] / (self.participation_counts[client] + 1)
                    else:
                        # Fallback to data size and participation count heuristic if no gradient is available
                        candidate_gain = self.client_data_sizes[client] / (self.participation_counts[client] + 1)

                    gain = candidate_gain

                    if gain > best_gain:
                        best_gain = gain
                        best_client = client

            if best_client is not None:
                selected_idx.append(best_client)
                clients_gains.append(best_gain)
                self.participation_counts[best_client] += 1
            else:
                # Fallback: randomly add any unselected client if no best client was found (rare)
                unselected_clients = set(range(self.n_device)) - set(selected_idx)
                random_id = unselected_clients.pop()
                selected_idx.append(random_id)
                clients_gains.append(0)
                self.participation_counts[random_id] += 1

        return selected_idx, clients_gains

    def solve_linear_programming(self, selected_idx, clients_gains, client_capabilities, time_budget):
        lp_problem = pulp.LpProblem("Compression_Ratio_Optimization", pulp.LpMinimize)

        # Extract relevant capabilities for selected clients
        T_k_cmp = np.array([client_capabilities[c]['computation_time'] for c in selected_idx])
        bandwidths = np.array([client_capabilities[c]['outbound_bandwidth'] for c in selected_idx])

        # Create variables for theta_k (compression ratios)
        theta_k = [pulp.LpVariable(f'theta_{i}', lowBound=0, upBound=1) for i in range(self.n_selected)]

        # Estimate compression errors for the selected clients
        compression_errors = []
        for i in range(self.n_selected):
            # gradient_norm = np.sum([torch.norm(grad).item() for grad in client_gains[selected_clients[i]]])
            gradient_norm = clients_gains[i]
            # Check for NaN or Inf in gradient_norm
            if np.isnan(gradient_norm) or np.isinf(gradient_norm):
                raise ValueError(
                    f"NaN or Inf value detected in gradient_norm for client {selected_idx[i]}. Please check your data.")

            compression_errors.append(gradient_norm)

        compression_errors = np.array(compression_errors)

        # Define the objective function to minimize the sum of compression errors
        H = 5
        lp_problem += pulp.lpSum((1 - theta_k[i]) * H ** 2 * compression_errors[i] ** 2 for i in range(self.n_selected)), "Total_Compression_Error"

        # Add constraints for each client
        for i in range(self.n_selected):
            lp_problem += T_k_cmp[i] + (theta_k[i] * self.model_size) / bandwidths[i] <= time_budget, f"Time_Constraint_{i}"

        # Solve the problem
        solver = pulp.PULP_CBC_CMD()
        lp_problem.solve(solver)

        theta_k_values = [pulp.value(theta_k[i]) for i in range(self.n_selected)]
        print(theta_k_values)

        return theta_k_values

    def joint_optimization_sub(self, client_capabilities, time_budget):
        """
        Perform joint optimization for client selection and compression ratios.
        """
        # Step 1: Select clients based on their gradients
        selected_idx, clients_gains = self.select_clients_submodular()
        print(f"Selected clients: {selected_idx}")

        # Step 2: Determine compression rates for the selected clients
        compression_rates = self.solve_linear_programming(selected_idx, clients_gains, client_capabilities, time_budget)

        return selected_idx, compression_rates
