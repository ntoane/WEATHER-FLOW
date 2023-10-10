import torch

    
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss 
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    # print(mask)
    loss = (y_pred - y_true)*(y_pred-y_true)
    loss = loss 
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mape_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)/y_true
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.abs().mean()

def masked_smape_loss(y_pred, y_true):
    # mask = (y_true != 0).float()
    # mask /= mask.mean()
    # print(mask)
    # print("PRedicted")
    # print(y_pred[0])
    # print("TRue")
    # print(y_true)
    numerator = torch.abs(y_pred - y_true)
    # print(numerator)
    denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2.0
    
    loss = (numerator / denominator) * 100.0
    # print(loss.mean())
    loss = loss
    # loss = loss  * mask

    loss[loss != loss] = 0
    
    return loss.mean()
