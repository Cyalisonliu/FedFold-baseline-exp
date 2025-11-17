import torch
from collections import defaultdict
from typing import List, Dict, Any, Union

def get_trained_classes(n_device: int) -> Union[Dict[int, List[int]], None]:
    """
    Reads the device_classes_log.txt file to determine the classes each device was trained on.
    Returns a dictionary mapping class ID to a list of device indices that trained on it.
    """
    class_to_devices = defaultdict(list)
    try:
        with open("device_classes_log.txt", "r") as f:
            for i, line in enumerate(f):
                if i >= n_device:
                    break
                    
                # Example line format: "Classes [1, 5, 8]\n"
                # Extract the list part: [1, 5, 8]
                try:
                    class_str = line.split('[')[1].split(']')[0]
                    # Filter for empty strings just in case the log format has trailing commas
                    classes = [int(c.strip()) for c in class_str.split(',') if c.strip()]
                except IndexError:
                    print(f"Warning: Could not parse line {i+1} in log file.")
                    continue
                
                for c in classes:
                    class_to_devices[c].append(i)
    except FileNotFoundError:
        print("Warning: device_classes_log.txt not found. Cannot calculate per-class metrics.")
        return None
        
    return dict(class_to_devices)


def test_per_class_performance(cfg: Dict[str, Any], labels: Dict[str, List[int]], 
                               prediction: torch.Tensor, device: str) -> float:
    """
    Calculates the average accuracy across all classes, using the final ensemble prediction.
    It prints detailed accuracy for each class and returns the overall average.

    :param cfg: The configuration dictionary (requires 'n_device', 'n_class').
    :param labels: The labels dictionary (requires 'test' labels).
    :param prediction: The final ensemble prediction tensor (logits or softmax).
    :param device: The computational device ('cuda' or 'cpu').
    :return: The average per-class accuracy.
    """
    
    class_to_devices = get_trained_classes(cfg['n_device'])
    if class_to_devices is None:
        return 0.0
        
    n_class = cfg['n_class']
    final_prediction_logits = prediction
    test_labels = torch.tensor(labels['test']).to(device)

    class_accuracies = {}
    
    # Calculate Accuracy for each class
    for c in range(n_class):
        # Find indices in the test set belonging to class 'c'
        class_indices = (test_labels == c).nonzero(as_tuple=True)[0]
        
        if class_indices.numel() == 0:
            continue
            
        # Filter the predictions and true labels for only this class
        # Ensure we use argmax since prediction might be probabilities
        class_preds = final_prediction_logits.argmax(dim=1).index_select(0, class_indices)
        class_true_labels = test_labels.index_select(0, class_indices)
        
        # Calculate accuracy for this class
        acc = (class_preds == class_true_labels).float().mean().item()
        
        class_accuracies[c] = acc

    # Calculate the Final Average Class Accuracy
    if not class_accuracies:
        return 0.0

    avg_class_acc = sum(class_accuracies.values()) / len(class_accuracies)
    
    # Log the results
    print("\n--- Per-Class Ensemble Accuracy ---")
    for c, acc in class_accuracies.items():
        # Report the number of devices trained on this class for context
        print(f"Class {c:03d} Acc: {acc:.4f} (Trained by {len(class_to_devices.get(c, []))} devices in the pool)")
        
    print(f"Average Per-Class Ensemble Accuracy: {avg_class_acc:.4f}")
    
    return avg_class_acc