import numpy as np
import torch
import pulp


class FedCG:
    def __init__(self, client_data_sizes, global_model, n_selected, n_device):
        # global model and params
        self.global_model = global_model
        self.accum_global_params = []
        self.cnt = []

        total_parameters = sum(p.numel() for p in self.global_model.parameters())
        bytes_per_parameter = 4
        self.model_size = total_parameters * bytes_per_parameter
        print(f"Model size: {self.model_size} bytes")

        # info for join optimization
        self.client_data_sizes = client_data_sizes
        self.participation_counts = {i: 0 for i in range(n_device)}
        self.client_gradients = {i: None for i in range(n_device)}  # Initialize gradient history

        # store selected clients
        self.n_device = n_device
        self.n_selected = n_selected
        self.selected_idx = []
        self.clients_gains = []

    def download(self, client_model):
        # for k, v in self.global_model.state_dict().items():
        #     # local_params[k] = copy.deepcopy(v)
        #     local_params[k] = v.detach().clone()
        client_model.load_state_dict({k: v.clone().detach() for k, v in self.global_model.state_dict().items()})

    def upload(self, model_params, client_id):
        update = {k: v.clone().detach() for k, v in model_params.items()}
        # Calculate the gradient norm or parameter delta (proxy for gradient)
        gradients = {k: update[k] - self.global_model.state_dict()[k] for k in update.keys()}

        # Store the actual gradients for each client
        self.client_gradients[client_id] = gradients

        self.accum_global_params.append(update)

    def aggregate(self):
        if len(self.accum_global_params) != self.n_selected:
            print("Not collect all updates from clients yet.")
            return

        delta = {name: torch.zeros_like(param) for name, param in self.global_model.state_dict().items()}

        for update in self.accum_global_params:
            for k, v in update.items():
                # delta[k] += (v - self.global_model.state_dict()[k])
                delta[k] += v

        self.global_model.load_state_dict({k: v / self.n_selected for k, v in delta.items()})

        # reset for next round
        self.accum_global_params = []
        self.selected_idx = []

    # def select_clients_submodular(self):
    #     """
    #     Select M clients using a submodular maximization approach based on participation history and data size.
    #     Clients with larger data sizes and lower participation will be prioritized.
    #
    #     Returns:
    #     - selected_clients: List of selected client indices.
    #     """
        # print(self.participation_counts)
        # for _ in range(self.n_selected):
        #     best_gain = -float('inf')
        #     best_client = None
        #
        #     for client in range(self.n_device):
        #         if client not in self.selected_idx:
        #             # Calculate a heuristic for marginal gain based on data size and participation history
        #             gain = self.client_data_sizes[client] / (self.participation_counts[client] + 1)
        #             if gain > best_gain:
        #                 best_gain = gain
        #                 best_client = client
        #
        #     if best_client is not None:
        #         self.selected_idx.append(best_client)
        #         self.clients_gains.append(best_gain)
        #         self.participation_counts[best_client] += 1
        #     else:
        #         # Fallback: randomly add any unselected client if no best client was found (rare)
        #         unselected_clients = set(range(self.n_device)) - set(self.selected_idx)
        #         random_id = unselected_clients.pop()
        #         self.selected_idx.append(random_id)
        #         self.clients_gains.append(0)
        #         self.participation_counts[random_id] += 1
        #
        # return self.selected_idx, self.clients_gains
    def select_clients_submodular(self):
        """
        Select clients using a submodular maximization approach based on participation history,
        data size, and actual gradients stored from previous rounds.
        """
        print(self.participation_counts)
        for _ in range(self.n_selected):
            best_gain = -float('inf')
            best_client = None

            for client in range(self.n_device):
                if client not in self.selected_idx:
                    # Calculate a heuristic based on data size and gradient difference
                    if self.client_gradients[client] is not None:
                        gradient_distance = sum(
                            torch.norm(self.client_gradients[client][k]) for k in self.client_gradients[client].keys())
                    else:
                        gradient_distance = 1  # Default to 1 if no gradient is available

                    gain = (self.client_data_sizes[client] / (
                                self.participation_counts[client] + 1)) * gradient_distance

                    if gain > best_gain:
                        best_gain = gain
                        best_client = client

            if best_client is not None:
                self.selected_idx.append(best_client)
                self.clients_gains.append(best_gain)
                self.participation_counts[best_client] += 1
            else:
                # Fallback: randomly add any unselected client if no best client was found (rare)
                unselected_clients = set(range(self.n_device)) - set(self.selected_idx)
                random_id = unselected_clients.pop()
                self.selected_idx.append(random_id)
                self.clients_gains.append(0)
                self.participation_counts[random_id] += 1

        return self.selected_idx, self.clients_gains

    def solve_linear_programming(self, client_capabilities, time_budget):
        # Create the linear programming problem
        lp_problem = pulp.LpProblem("Compression_Ratio_Optimization", pulp.LpMinimize)

        # Extract relevant capabilities for selected clients
        T_k_cmp = np.array([client_capabilities[c]['computation_time'] for c in self.selected_idx])
        bandwidths = np.array([client_capabilities[c]['outbound_bandwidth'] for c in self.selected_idx])

        # Create variables for theta_k (compression ratios)
        theta_k = [pulp.LpVariable(f'theta_{i}', lowBound=0, upBound=1) for i in range(self.n_selected)]

        # Estimate compression errors for the selected clients
        compression_errors = []
        for i in range(self.n_selected):
            # gradient_norm = np.sum([torch.norm(grad).item() for grad in client_gains[selected_clients[i]]])
            gradient_norm = self.clients_gains[i]
            # Check for NaN or Inf in gradient_norm
            if np.isnan(gradient_norm) or np.isinf(gradient_norm):
                raise ValueError(
                    f"NaN or Inf value detected in gradient_norm for client {self.selected_idx[i]}. Please check your data.")

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
        self.select_clients_submodular()
        print(f"Selected clients: {self.selected_idx}")

        # Step 2: Determine compression rates for the selected clients
        compression_rates = self.solve_linear_programming(client_capabilities, time_budget)

        return self.selected_idx, compression_rates


# # no usage
# def estimate_compression_error(grad, k):
#     """
#     Estimate the compression error for the gradient when applying Top-K sparsification.
#
#     Args:
#     - grad: The gradient tensor.
#     - k: The number of top elements to keep (Top-K).
#
#     Returns:
#     - compression_error: The L2 norm of the difference between the original gradient and the compressed gradient.
#     """
#     # Flatten the gradient
#     grad_flat = grad.flatten()
#
#     # Get the indices of the top-k elements by absolute value
#     topk_indices = torch.topk(torch.abs(grad_flat), k)[1]
#
#     # Create a mask with the top-k elements
#     compressed_grad = torch.zeros_like(grad_flat)
#     compressed_grad[topk_indices] = grad_flat[topk_indices]
#
#     # Compute the compression error as the L2 norm of the difference
#     compression_error = torch.norm(grad_flat - compressed_grad, p=2).item()
#
#     return compression_error
#
#
# def calculate_marginal_gain(selected_clients, candidate_client, client_gradients):
#     """
#     Calculate the marginal gain of adding the candidate client to the selected_clients set.
#     The marginal gain is calculated based on the norm difference between the gradients.
#     """
#     # Flatten and concatenate the gradients of the candidate client
#     candidate_gradient = np.concatenate([grad.flatten().cpu().numpy() for grad in client_gradients[candidate_client]])
#     # If no clients have been selected, return the norm of the candidate's gradient
#
#     if not selected_clients:
#         return np.linalg.norm(candidate_gradient)
#
#     # Compute the gain as the sum of norm differences with each selected client's gradient
#     gain = 0
#     for client in selected_clients:
#         client_gradient = np.concatenate([grad.flatten().cpu().numpy() for grad in client_gradients[client]])
#         gain += np.linalg.norm(candidate_gradient - client_gradient)
#
#     return gain
#
#
# def select_clients(client_gradients, M):
#     """
#     Select M clients from the client_gradients list using a submodular maximization approach.
#     """
#     selected_clients = []
#     for _ in range(M):
#         best_gain = -float('inf')
#         best_client = None
#
#         for client in range(len(client_gradients)):
#             if client not in selected_clients:
#                 gain = calculate_marginal_gain(selected_clients, client, client_gradients)
#                 if gain > best_gain:
#                     best_gain = gain
#                     best_client = client
#
#         # Ensure that the best client is added
#         if best_client is not None:
#             selected_clients.append(best_client)
#         else:
#             # Fallback: add the first unselected client if no best client was found (very rare case)
#             unselected_clients = set(range(len(client_gradients))) - set(selected_clients)
#             if unselected_clients:
#                 selected_clients.append(unselected_clients.pop())
#
#     return selected_clients
#
#
# def joint_optimization(client_gradients, client_capabilities, M, time_budget, model_size):
#     """
#     Perform joint optimization for client selection and compression ratios.
#     """
#     # Step 1: Select clients based on their gradients
#     selected_clients = select_clients(client_gradients, M)
#     print(f"Selected clients: {selected_clients}")
#
#     # Step 2: Determine compression rates for the selected clients
#     compression_rates = solve_linear_programming(selected_clients, client_capabilities, time_budget,
#                                                  client_gradients, model_size)
#
#     return selected_clients, compression_rates