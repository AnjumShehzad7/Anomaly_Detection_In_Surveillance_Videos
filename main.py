from torch.utils.data import DataLoader  # DataLoader for batch processing
from learner import Learner  # Import the model
from loss import *  # Import custom loss functions
from dataset import *  # Import dataset loaders
import os  # OS module for system operations
from sklearn import metrics  # Metrics for evaluation

# Load datasets (Normal and Anomaly) for training and testing
normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)

anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)

# Create DataLoaders for batch processing
normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

# Set device for computation (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize model, optimizer, scheduler, and loss function
model = Learner(input_dim=2048, drop_p=0.0).to(device)  # Model with input dimension 2048
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.0010000000474974513)  # Adagrad optimizer
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])  # Learning rate scheduler
criterion = MIL  # MIL loss function

def train(epoch):
    """
    Training loop for a single epoch.
    
    Args:
        epoch (int): Current epoch number.
    """
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        # Concatenate anomaly and normal inputs along the feature dimension
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]  # Get batch size
        inputs = inputs.view(-1, inputs.size(-1)).to(device)  # Reshape and move to device
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print('Loss = {:.6f}'.format(train_loss / len(normal_train_loader)))
    scheduler.step()  # Update learning rate scheduler

def test_abnormal(epoch):
    """
    Evaluation function to compute AUC on abnormal test data.
    
    Args:
        epoch (int): Current epoch number.
    """
    model.eval()
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            # Process anomaly test data
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0] // 16, 33))  # Define step intervals

            for j in range(32):  # Assign model scores to frame intervals
                score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):  # Assign ground truth labels to respective frames
                s = gts[k * 2]
                e = min(gts[k * 2 + 1], frames)
                gt_list[s - 1:e] = 1

            # Process normal test data
            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
            
            for kk in range(32):  # Assign model scores for normal data
                score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
            
            gt_list2 = np.zeros(frames2[0])
            
            # Concatenate anomaly and normal scores and ground truth labels
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
            
            # Compute AUC using ROC curve
            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
        
        print('AUC = {:.6f}'.format(auc / 140))

# Training and testing loop for multiple epochs
for epoch in range(75):
    train(epoch)
    test_abnormal(epoch)
