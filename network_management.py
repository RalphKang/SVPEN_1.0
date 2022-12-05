from network_alter.vgg import *
from torch.optim import AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F


def load_state_estimator():
    """
    This module is used to load the trained/untrained state estimator
    """
    state_estimator = VGG(make_layers(cfg['A'], batch_norm=False), 2)  # model A, final output 5 items
    state_estimator.to("cuda")
    model_save_dir = 'ml_model/vgg_A.pt'
    state_estimator.load_state_dict(torch.load(model_save_dir))
    return state_estimator


def optimization_network_config(state_estimator,LR_state=1.5e-4, LR_error=1.5e-4):

    error_net_1 = error_net(make_layers(cfg['A'], batch_norm=False),1)
    error_net_2 = error_net(make_layers(cfg['A'], batch_norm=False),1)
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
    observation_set = obverstion_for_net.repeat(state_set.shape[0], 1, 1)
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
    # label_state = torch.FloatTensor([[0, 0, 0, ]]).to("cuda")
    # loss_state = torch.maximum(loss_error_net(error_net_1(RAM_net(seed_for_RAM)), label_state),
    #                            loss_error_net(error_net_2(RAM_net(seed_for_RAM)), label_state))
    # use relu to avoid negative loss
    normed_feasible_domain_tc = torch.tensor(data_norm_store.feasible_domain, dtype=torch.float32).to('cuda')
    loss_boundary_1=F.relu(current_state[0,0]-normed_feasible_domain_tc[1,0])+F.relu(normed_feasible_domain_tc[0,0]-current_state[0,0])
    loss_boundary_2=F.relu(current_state[0,1]-normed_feasible_domain_tc[1,1])+F.relu(normed_feasible_domain_tc[0,1]-current_state[0,1])
    loss_state=estimated_error+loss_boundary_1+loss_boundary_2
    state_net_optimizer.zero_grad()
    loss_state.backward()
    state_net_optimizer.step()
    return torch.max(loss_diff, loss_diff_2), loss_state, loss_state
