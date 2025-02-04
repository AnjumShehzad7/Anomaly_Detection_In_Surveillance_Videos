import torch  # PyTorch library for deep learning
import torch.nn.functional as F  # Functional API for loss functions and activations

def MIL(y_pred, batch_size, is_transformer=0):
    """
    Multiple Instance Learning (MIL) loss function.
    
    Args:
        y_pred (Tensor): Predicted scores from the model.
        batch_size (int): Number of samples in the batch.
        is_transformer (int): Flag to indicate whether the model is a transformer (1) or not (0).
    
    Returns:
        Tensor: Computed MIL loss.
    """
    loss = torch.tensor(0.).cuda()  # Initialize total loss
    loss_intra = torch.tensor(0.).cuda()  # Unused intra-class loss placeholder
    sparsity = torch.tensor(0.).cuda()  # Sparsity regularization term
    smooth = torch.tensor(0.).cuda()  # Smoothness regularization term
    
    # Reshape predictions if not using a transformer
    if is_transformer == 0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)  # Apply sigmoid activation for transformer-based models
    
    for i in range(batch_size):
        # Generate random permutations for anomaly and normal samples
        anomaly_index = torch.randperm(30).cuda()
        normal_index = torch.randperm(30).cuda()
        
        # Select top 32 anomaly and normal predictions using randomized indices
        y_anomaly = y_pred[i, :32][anomaly_index]
        y_normal = y_pred[i, 32:][normal_index]
        
        # Compute max and min values for anomaly and normal samples
        y_anomaly_max = torch.max(y_anomaly)  # Maximum value among anomaly samples
        y_anomaly_min = torch.min(y_anomaly)  # Minimum value among anomaly samples
        
        y_normal_max = torch.max(y_normal)  # Maximum value among normal samples
        y_normal_min = torch.min(y_normal)  # Minimum value among normal samples
        
        # Compute hinge loss to enforce anomaly score > normal score
        loss += F.relu(1. - y_anomaly_max + y_normal_max)
        
        # Compute sparsity regularization to encourage small anomaly scores
        sparsity += torch.sum(y_anomaly) * 0.00008
        
        # Compute smoothness regularization to enforce temporal consistency
        smooth += torch.sum((y_pred[i, :31] - y_pred[i, 1:32])**2) * 0.00008
    
    # Average loss across batch
    loss = (loss + sparsity + smooth) / batch_size
    
    return loss
