import torch
from model import ResNet
from utils import Utils 

checkpoint_path_large = './output/CIFAR100_non-iid-20_train-ratio-16-1_ResNet/ResNet152_fedfold_q1/S2-W8_2024-11-09 21:29:13/4_8.pth'
checkpoint_path_small = './output/CIFAR100_non-iid-20_train-ratio-16-1_ResNet/ResNet152_fedfold_q1/S1-W9_2024-11-09 21:29:13/1_1.pth'  
state_dict_large = torch.load(checkpoint_path_large)
state_dict_small = torch.load(checkpoint_path_small)

# Initialize the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_size = {
            '16': [64, 128, 256, 512],
            '8': [32, 64, 128, 256],
            '4': [16, 32, 64, 128],
            '2': [8, 16, 32, 64],
            '1': [4, 8, 16, 32]
        }
model_fn = ResNet
model_large = model_fn(hidden_size['8']).to(device)
model_small = model_fn(hidden_size['1']).to(device)

# Load the state dict into the model
model_large.load_state_dict(state_dict_large)
model_small.load_state_dict(state_dict_small)
hidden_size = Utils.get_hidden_size(1)

models = Utils.split_resnet_params(model_large.state_dict(), hidden_size, 1)

#large compare to small
total_diff = 0.0
total_mse = 0.0
correlations = []
for (key, param) in model_small.state_dict().items():      
    param_0 = models[0][key]
    tensor0_flat = param_0.view(-1)
    tensor_flat = param.view(-1)
    
    print(f'Layer {key}')
    # print(param.size())
    # print(param_0.size())
    norm_diff = torch.norm(param - param_0)
    mse = torch.mean((param - param_0) ** 2)                
    correlation = torch.corrcoef(torch.stack([tensor0_flat, tensor_flat]))[0, 1]

    total_diff += norm_diff.item()
    total_mse += mse
    correlations.append(correlation)
    print(f"correlation: {correlation}")
    print(f"Mean Squared Error: {mse}")
    print(f"L2 Norm difference for model compared to models[0]: {norm_diff}")



#large model compare to its self portion
# for m in models:
#     total_diff = 0.0
#     total_mse = 0.0
#     correlations = []
#     for (key, param) in m.items():      
#         param_0 = models[0][key]
#         tensor0_flat = param_0.view(-1)
#         tensor_flat = param.view(-1)
        
#         norm_diff = torch.norm(param - param_0)
#         mse = torch.mean((param - param_0) ** 2)                
#         correlation = torch.corrcoef(torch.stack([tensor0_flat, tensor_flat]))[0, 1]

#         total_diff += norm_diff.item()
#         total_mse += mse
#         correlations.append(correlation)
#         print(f"correlation in layer {key}: {correlation}")

#     print(f"L2 Norm difference for model compared to models[0]: {total_diff}")
#     print(f"Mean Squared Error: {total_mse}")
    # print(f"Correlation: {sum(correlations)/len(correlations)}")

# Print the model parameters
# log_file = open("./model_param/fedfold_resnet152_2_2.txt", "w")
# for name, param in model_large.named_parameters():
#     log_file.write(f"Parameter name: {name}\n")
#     log_file.write(f"{param.data}\n")
# log_file.close()