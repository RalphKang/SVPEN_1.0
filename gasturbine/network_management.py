from network_alter.mlp import *
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F


def load_state_estimator(input_size,output_size):
    """
    This module is used to load the trained/untrained state estimator
    """
    state_estimator = MLP_pred(input_size=input_size, action_size=output_size) # model A, final output 5 items
    state_estimator.to("cuda")
    return state_estimator


def optimization_network_config(state_estimator,LR_state=1.5e-4, LR_error=1.5e-4,input_size=2,action_size=11,num_classes=1):

    error_net_1 = MLP_error_net(input_size=input_size, action_size=action_size, num_classes=num_classes)
    error_net_2 = MLP_error_net(input_size=input_size, action_size=action_size, num_classes=num_classes)
    error_net_1.to('cuda')
    error_net_2.to('cuda')

    optimizer_est = AdamW(state_estimator.parameters(), lr=LR_state, weight_decay=0.)
    # optimizer_RAM = RMSprop(RAM_net.parameters(), lr=LR_RAM, weight_decay=0.)
    lr_scd_action = CosineAnnealingWarmRestarts(optimizer_est,T_0=20,T_mult=1)  # to avoid local minima

    optimizer_error1 = AdamW(error_net_1.parameters(), lr=LR_error, weight_decay=0.)
    optimizer_error2 = AdamW(error_net_2.parameters(), lr=LR_error, weight_decay=0.)
    loss_error_net = nn.MSELoss(reduction="mean")
    return error_net_1, error_net_2, optimizer_est, optimizer_error1, optimizer_error2, lr_scd_action, loss_error_net

def online_update(error_net_1, error_net_2, state_net, obverstion_for_net, state_set,error_set, data_norm_store,
                  error_net_optimizer,error_net_2_optimizer, state_net_optimizer, loss_error_net, delay, current_epoch):
    error_net_1.train()
    error_net_2.train()
    state_net.train()
    state_set = state_set.squeeze().to("cuda")
    error_set = error_set.to("cuda")
    # norm_label_pred = RAM_net(test_sample)
    observation_set = obverstion_for_net.repeat(state_set.shape[0], 1)
    for i in range(delay):
        error_net_pred_1 = error_net_1(observation_set, state_set)
        loss_diff = loss_error_net(error_net_pred_1, error_set)
        if loss_diff > 1e-4:
            error_net_optimizer.zero_grad()
            loss_diff.backward()
            error_net_optimizer.step()

        error_net_2_pred = error_net_2(observation_set,state_set)
        loss_diff_2 = loss_error_net(error_net_2_pred, error_set)
        if loss_diff_2 > 1e-4:
            error_net_2_optimizer.zero_grad()
            loss_diff_2.backward()
            error_net_2_optimizer.step()
    current_state = state_net(obverstion_for_net)
    estimated_error = torch.maximum(error_net_1(obverstion_for_net,current_state),error_net_2(obverstion_for_net,current_state))
    loss_boundary = torch.mean(F.relu(current_state - torch.ones_like(current_state)) + \
                               F.relu(torch.zeros_like(current_state) - current_state), dim=1)

    # label_state = torch.FloatTensor([[0, 0, 0, ]]).to("cuda")
    # loss_state = torch.maximum(loss_error_net(error_net_1(RAM_net(seed_for_RAM)), label_state),
    #                            loss_error_net(error_net_2(RAM_net(seed_for_RAM)), label_state))
    # use relu to avoid negative loss
    loss_state=estimated_error+loss_boundary*0.1
    state_net_optimizer.zero_grad()
    loss_state.backward()
    state_net_optimizer.step()
    return torch.max(loss_diff, loss_diff_2), loss_state, loss_state
