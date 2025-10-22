import torch
import numpy as np
import copy
import math


class Compressor:
    def apply_top_k(model, k_ratio):
        """
        Apply Top-K sparsification on the model params.
        :param model: The model whose params will be sparsified.
        :param k_ratio: The ratio of params to keep (K).
        """
        with torch.no_grad():
            for param in model.parameters():
                if param is not None:
                    # Flatten the parameter tensor and compute the number of values to keep
                    param_flat = param.view(-1)
                    k = int(len(param_flat) * k_ratio)
                    if k == 0:
                        continue

                    _, idxs = torch.topk(param_flat.abs(), k, sorted=False)
                    compressed_param = torch.zeros_like(param_flat)
                    compressed_param[idxs] = param_flat[idxs]
                    param.copy_(compressed_param.view_as(param))
        total_bytes_top_k = 0
        for param in model.parameters():
            total_bytes_top_k += param.data.nelement() * param.data.element_size()
        print(f"Total bytes after Top-K: {total_bytes_top_k}")


class Utils:
    # Retrieve model parameters for the specified size
    def get_model_params(idx, model_list):
        model = model_list[idx]
        return {name: param.data for name, param in model.named_parameters()}
    
    # align all model to the left upper corner, all params are divided by the same ammount
    def accum_model(model_idx, model_list):
        # model order in model_idx: [1,1,2,4,8]
        local_model_params = {}
        accum_model_params = {}
        cnt = len(model_idx)
        reversed_model_idx = list(reversed(model_idx)) #[1,1,2,4,8] [4,3,2,1,0]
        # weight = [1,1,2,4,8]
        weight = [1,1,1,1,1]
        for idx in reversed_model_idx:
            model_params = Utils.get_model_params(idx, model_list)
            # print(f"Model params for idx {idx}: {model_params['layer1.1.conv2.weight'][0][0]}")
            if not accum_model_params:
                accum_model_params = {k: v.clone()*weight[idx] for k, v in model_params.items()}
                # print(f"Initial accum_model_params: {accum_model_params['layer1.1.conv2.weight'][0][0]}")
            else:
                for k, v in model_params.items():
                    if accum_model_params[k].shape != v.shape:
                        diff = [accum_model_params[k].size(dim) - v.size(dim) for dim in range(v.dim())]
                        if all(d >= 0 for d in diff):
                            padding = []
                            for d in reversed(diff):
                                padding.extend((0, d))  # Padding format (pad_left, pad_right)
                            padded_v = torch.nn.functional.pad(v, padding)
                            accum_model_params[k] += padded_v*weight[idx]
                        else:
                            raise ValueError(f"Cannot pad tensor {k}, model_params is larger in some dimensions")
                    else:
                        accum_model_params[k] += v.clone()*weight[idx]  
                # print(f"Accumulated: {accum_model_params['layer1.1.conv2.weight'][0][0]}")          
                
        # for k in accum_model_params.keys():
        #     accum_model_params[k] /= cnt
        # print(f"Averaged accum_model_params: {accum_model_params['layer1.1.conv2.weight'][0][0]}")

        local_model_params = Utils.get_model_params(4, model_list)
        for k, v in accum_model_params.items():
            local_model_params[k] = v.clone()
        # print(f"Final local_model_params: {local_model_params['layer1.1.conv2.weight'][0][0]}")
        return local_model_params
    
    # split the aggregate model, no split linear part
    def split_resnet_params(global_params, hidden_sizes, moving_spitting):
        models = [{} for _ in hidden_sizes]
        n_class = 10

        for k, concat_param in global_params.items():
            start_idx1 = 0
            start_idx2 = 0
            # print(f"Global model param shape for {k}: {concat_param.shape}")
            # Split the parameters based on each model's hidden size
            for i, hidden_size in enumerate(hidden_sizes):
                if 'layer1' in k:
                    param_size1 = hidden_size[0]
                    param_size2 = hidden_size[0]
                elif 'layer2' in k:
                    param_size1 = hidden_size[1]
                    if 'layer2.0.conv1' in k or 'layer2.0.shortcut.weight' in k:
                        param_size2 = hidden_size[0]
                    else:
                        param_size2 = hidden_size[1]
                elif 'layer3' in k:
                    param_size1 = hidden_size[2]
                    if 'layer3.0.conv1' in k or 'layer3.0.shortcut.weight' in k:
                        param_size2 = hidden_size[1]
                    else:
                        param_size2 = hidden_size[2]
                elif 'layer4' in k:
                    param_size1 = hidden_size[3]
                    if 'layer4.0.conv1' in k or 'layer4.0.shortcut.weight' in k:
                        param_size2 = hidden_size[2]
                    else:
                        param_size2 = hidden_size[3]
                elif 'conv1' in k:
                    param_size1 = hidden_size[0]
                    param_size2 = hidden_size[0]
                elif 'linear.weight' in k:
                    param_size1 = n_class
                    param_size2 = hidden_size[3]
                elif 'linear.bias' in k:
                    param_size1 = n_class
                    param_size2 = None
                else:
                    raise ValueError(f"Unknown layer key: {k}")               

                if moving_spitting:
                    if concat_param.ndim == 1:  # Biases (1D tensors)
                        models[i][k] = concat_param[:param_size1].clone()
                    else:  # 2D+ tensors
                        models[i][k] = concat_param[start_idx1:start_idx1 + param_size1, start_idx2:start_idx2+param_size2, ...].clone()
                    if 'linear.bias' in k:
                        continue
                    elif k=='conv1.weight':
                        start_idx1 += param_size1
                    elif 'linear.weight' in k:
                        start_idx2 += param_size2
                    else:
                        start_idx2 += param_size2
                        start_idx1 += param_size1
                else:
                    if concat_param.ndim == 1:  # Biases (1D tensors)
                        models[i][k] = concat_param[:param_size1].clone()
                    else:  # 2D+ tensors
                        models[i][k] = concat_param[start_idx1:start_idx1 + param_size1, :param_size2, ...].clone()
                
                # print(f"Split model {i} param shape for {k}: {models[i][k].shape}")
                # if using accum_model, please comment this line
                # start_idx1 += param_size1 
                
        return models
    
    def split_cnn_params(global_params, hidden_sizes, moving_splitting):
        models = [{} for _ in hidden_sizes]
        for i, hidden_size in enumerate(hidden_sizes):
            for k, concat_param in global_params.items():
                start_idx1 = 0
                start_idx2 = 0
                # print(f"Global model param shape for {k}: {concat_param.shape}")
                if '0' in k:
                    param_size1 = hidden_size[0]
                    if 'bias' in k:
                        param_size2 = None
                    else:
                        param_size2 = 3
                elif '13.weight' in k:
                    param_size1 = 10
                    param_size2 = hidden_size[3]
                elif '13.bias' in k:
                    param_size1 = 10
                    param_size2 = None
                elif '3' in k:
                    param_size1 = hidden_size[1]
                    if 'bias' in k:
                        param_size2 = None
                    else:
                        param_size2 = hidden_size[0]
                elif '6' in k:
                    param_size1 = hidden_size[2]
                    if 'bias' in k:
                        param_size2 = None
                    else:
                        param_size2 = hidden_size[1]
                elif '9' in k:
                    param_size1 = hidden_size[3]
                    if 'bias' in k:
                        param_size2 = None
                    else:
                        param_size2 = hidden_size[2]
                
                if moving_splitting:
                    if concat_param.ndim == 1:  # Biases (1D tensors)
                        models[i][k] = concat_param[:param_size1].clone()
                    else:  # 2D+ tensors
                        models[i][k] = concat_param[start_idx1:start_idx1 + param_size1, start_idx2:start_idx2+param_size2, ...].clone()
                    if k=='blocks.13.bias':
                        continue
                    elif k=='blocks.0.weight' or concat_param.ndim == 1:
                        start_idx1 += param_size1
                    elif 'blocks.13.weight' in k:
                        start_idx2 += param_size2
                    else:
                        start_idx2 += param_size2
                        start_idx1 += param_size1
                else:
                    if concat_param.ndim == 1:  # Biases (1D tensors)
                        models[i][k] = concat_param[:param_size1].clone()
                    else:  # 2D+ tensors
                        models[i][k] = concat_param[:param_size1, :param_size2, ...].clone()
                # print(f"Split model {i} param shape for {k}: {models[i][k].shape}")
        return models

    def split_and_merge(model_idx, model_list, model_type):
        models = []
        for idx in model_idx:
            hidden_size = Utils.get_hidden_size(idx)
            # print(f"length of hidden_size={len(hidden_size)}")
            large_model = Utils.get_model_params(idx, model_list)
            if model_type == 'ResNet':
                models.extend(Utils.split_resnet_params(large_model, hidden_size, 1))
            elif model_type =='Conv':
                models.extend(Utils.split_cnn_params(large_model, hidden_size, 1))
        # print(len(models))
        local_model_params = {}
        accum_model_params = {}

        for idx, model_params in enumerate(models):
            if not accum_model_params:
                accum_model_params = {k: v.clone()  for k, v in model_params.items()}
            else:
                for k, v in model_params.items():
                    accum_model_params[k] += v.clone()
        local_model_params = {k: v.clone() for k, v in accum_model_params.items()}
        # print(f"Local: {local_model_params['layer1.1.conv2.weight'][0][0][0][0]}")
        return local_model_params
    
    def concat_resnet(model, hidden_sizes):    
        concatenated_params = []
        model_tmp = copy.deepcopy(model)
        for hidden_size in hidden_sizes:
            if hidden_size[0] == 4:
                concatenated_params.append(model_tmp)
            else:         
                concat_model = copy.deepcopy(model_tmp)
                for _ in range(int(math.log(hidden_size[0],2)-2)):
                    model = copy.deepcopy(concat_model)                                  
                    for k, v in model.items():                                        
                        if 'linear.bias' in k:
                            continue
                        elif 'linear' in k:
                            concat_model[k] = torch.cat((concat_model[k], v), dim=1)
                        elif k=='conv1.weight':
                            concat_model[k] = torch.cat((concat_model[k], v), dim=0)
                        else:
                            concat_model[k] = torch.cat((concat_model[k], v), dim=0)
                            diff = [concat_model[k].size(dim) - v.size(dim) for dim in range(v.dim())]
                            if any(d > 0 for d in diff):
                                padding = []
                                for d in reversed(diff):
                                    padding.extend((0, d))
                                padded_v = torch.nn.functional.pad(v, padding)
                            concat_model[k] = torch.cat((concat_model[k], padded_v), dim=1)
                            # print(f"Shape of concat_model[{k}]: {concat_model[k].shape}")                                                                                                                
                concatenated_params.append(concat_model)
        # print(len(concatenated_params))
        return concatenated_params
    
    def concat_cnn(model, hidden_sizes):    
        concatenated_params = []
        model_tmp = copy.deepcopy(model)
        for hidden_size in hidden_sizes:
            if hidden_size[0] == 4:
                concatenated_params.append(model)
            else:         
                concat_model = copy.deepcopy(model_tmp)
                for _ in range(int(math.log(hidden_size[0],2)-2)):
                    model = copy.deepcopy(concat_model)                                  
                    for k, v in model.items():                                        
                        if 'blocks.13.bias' in k:
                            continue
                        elif 'blocks.13.weight' in k:
                            concat_model[k] = torch.cat((concat_model[k], v), dim=1)
                        elif k=='blocks.0.weight' or v.ndim==1 :
                            concat_model[k] = torch.cat((concat_model[k], v), dim=0)
                        else:
                            concat_model[k] = torch.cat((concat_model[k], v), dim=0)
                            diff = [concat_model[k].size(dim) - v.size(dim) for dim in range(v.dim())]
                            if any(d > 0 for d in diff):
                                padding = []
                                for d in reversed(diff):
                                    padding.extend((0, d))
                                padded_v = torch.nn.functional.pad(v, padding)
                            concat_model[k] = torch.cat((concat_model[k], padded_v), dim=1)
                            # print(f"Shape of concat_model[{k}]: {concat_model[k].shape}")                                                                                                                
                concatenated_params.append(concat_model)
        # print(len(concatenated_params))
        return concatenated_params


    def get_hidden_size(idx):
        hidden_size = [
            [4, 8, 16, 32],
            [4, 8, 16, 32],
            [4, 8, 16, 32],
            [4, 8, 16, 32],
            [4, 8, 16, 32],
            [4, 8, 16, 32],
            [4, 8, 16, 32],
            [4, 8, 16, 32],
        ]
        
        if idx==0 or idx==1:
            return hidden_size[:1]
        elif idx==2:
            return hidden_size[:2]
        elif idx==3:
            return hidden_size[:4]
        return hidden_size


