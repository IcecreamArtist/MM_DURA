import torch
import torch.nn as nn

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_tasks):
        super(MultiTaskLossWrapper, self).__init__()
        self.num_tasks = num_tasks
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, preds, targets):
        # preds and targets should be the same size and have the following shape: [batch_size, num_tasks]
        assert preds.shape == targets.shape, "Predictions and targets must have the same shape."

        total_loss = 0
        for i in range(self.num_tasks):
            task_pred = preds[:, i]
            task_target = targets[:, i]
            total_loss += self.mse_loss(task_pred, task_target)

        return total_loss / self.num_tasks
    


def new_MSE(y_true, y_pred):
    # Create a mask where y_true is not equal to -5
    mask = y_true != -5
    mask = mask.type_as(y_true)  # Convert mask to the same dtype as y_true
    # Apply the mask to y_true and y_pred
    y_true_masked = mask * y_true
    y_pred_masked = mask * y_pred
    # Compute the squared error
    squared_errors = torch.square(y_pred_masked - y_true_masked)
    # Compute MSE: sum of squared errors divided by the sum of the mask (to only consider masked elements)
    mse = torch.sum(squared_errors) / (torch.sum(mask) + 1e-16)
    return mse


def masked_MAE(y_true, y_pred):
    # Squeeze the input tensors to remove dimensions of size 1
    y_true = torch.squeeze(y_true)
    y_pred = torch.squeeze(y_pred)
    # Create a mask that is False where y_true is -5.0 and True elsewhere
    mask = y_true != -5.0
    mask = mask.type_as(y_true)  # Ensure the mask is the same type as y_true
    # Apply the mask
    y_true = y_true * mask
    y_pred = y_pred * mask
    # Calculate the absolute errors only where mask is True, then sum and normalize
    absolute_errors = torch.abs(y_true - y_pred)
    mae = torch.sum(absolute_errors) / (torch.sum(mask) + 1e-16)
    return mae
