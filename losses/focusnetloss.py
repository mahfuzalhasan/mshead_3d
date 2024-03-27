from .tversky import tversky_loss
from monai.losses import TverskyLoss, FocalLoss
from .structure_loss import total_structure_loss_focusnet

def FocusNetLoss(preds, mask):
    # print([pred.size() for pred in preds])
    # outputs, lateral_map_2, lateral_map_1 = preds
    outputs = preds
    # print(outputs.size())
    tversky_loss = TverskyLoss(to_onehot_y=False, softmax=False, sigmoid=True)
    focal_loss = FocalLoss(to_onehot_y=False, softmax=False, sigmoid=True)
    t_l = tversky_loss(outputs, mask)
    f_l = focal_loss(outputs, mask)
    # t_l = tversky_loss(mask, outputs)
    # print(t_l)
    # s_l = total_structure_loss_focusnet(mask, (lateral_map_2, lateral_map_1))
    return t_l + f_l