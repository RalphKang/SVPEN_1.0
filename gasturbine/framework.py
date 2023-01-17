from data_operation import *
from network_management import load_state_estimator, optimization_network_config, online_update
from physical_model import gasturbine_forward_model, physics_evaluation_module
import argparse
import os

"""
SVPEN1.1 :
This version is reorganized old V1.0
"""


def main(index_test):
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iteration', type=int, default=334, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--error_threshold', type=float, default=0.05, help='error threshold')
    parser.add_argument('--delay_update', type=int, default=40, help='delay')
    parser.add_argument('--error_zoomer', type=float, default=1.0, help='error zoomer')
    parser.add_argument('--error_estimator_number', type=int, default=4, help='number of error estimator')
    parser.add_argument('--feasible_domain_state', type=np.ndarray,
                        default=np.array([[5.0, 1.3, 1.2, 8.0, 1300., 0.85, 0.82, 0.84, 0.95, 0.86, 0.87],
                                          [6.0, 2.5, 2.0, 15.0, 1800., 0.95, 0.92, 0.94, 0.995, 0.96, 0.97]]),
                        help='feasible domain')
    parser.add_argument('--ground_truth_state', type=np.ndarray,
                        default=np.array([5.313, 1.636, 2.84, 9.0, 1624., 0.864, 0.87, 0.915, 0.85, 0.985, 0.985]),
                        help='ground truth')
    parser.add_argument('--state_meaning', type=list,
                        default=['BPR', 'PR_fan', 'PR_LC', 'PR_HC', 'T4', 'eta_fan', 'eta_LC', 'eta_HC', 'eta_B',
                                 'eta_HT',
                                 'eta_LT'], help='state meaning')
    parser.add_argument('--record_path', type=str, default='output/best_state_recording.csv', help='record path')
    parser.add_argument('--observation_meaning', type=list, default=['thrust', 'TSFC'])
    args = parser.parse_args()
    global_search_step = int(args.batch_size)  # the number of initial global search steps

    optimization_recording = pandas_recording(
        column=args.state_meaning + args.observation_meaning + ['error'],
        save_dir='output/best_state_' + str(index_test) + '_recording.csv')
    state_size = len(args.ground_truth_state)
    # %% Obtain the test sample
    forward_physical_model = gasturbine_forward_model(h=0., mach=0.0, mf=361., eta_inlet=0.98, pi_burner=0.99,
                                                      eta_nozzle=0.985)
    actual_observation = forward_physical_model.generate_sample(args.ground_truth_state)  # generate test observation
    data_norm_store = normalization_data_store(feasible4state=args.feasible_domain_state,
                                               observation=actual_observation,
                                               state_size=state_size,
                                               error_zoomer=args.error_zoomer)  # store all normalization boundaries, feasible boundaries

    """
    ===========================================================================================
                         1 INVERSE FUNCTION MODE
    ===========================================================================================
    """
    state_estimator = load_state_estimator(input_size=len(args.observation_meaning),output_size=state_size)  # load state estimator
    actual_observation4net = data_norm_store.obse_trans_phy2state_est(actual_observation)  # observation transform
    actual_observation4net=actual_observation4net.squeeze(dim=1).to('cuda')
    state_est_first_norm_tc = state_estimator(actual_observation4net.to('cuda'))  # observation-->state estimation
    state_est_first_norm = state_est_first_norm_tc.detach().cpu().squeeze().numpy()  # state estimation: torch-->numpy
    observation_rebuilt_first, state_est_first, error_first = physics_evaluation_module(
        state_norm=state_est_first_norm,
        data_norm_store=data_norm_store,
        forward_physical_model=forward_physical_model)
    best_error = error_first
    best_state_norm = state_est_first_norm
    best_state = state_est_first
    best_observation = observation_rebuilt_first
    optimizaton_provide_mini = "Inverse function mode"
    i = 0
    optimization_recording_row_list = np.hstack(
        (best_state.flatten(), observation_rebuilt_first.flatten(), best_error)).tolist()
    optimization_recording.add_values(added_data=optimization_recording_row_list)
    if best_error < args.error_threshold:
        print("Optimization is done, best state: {}, best error:{}, the result is provided by {}".
              format(best_state, best_error, optimizaton_provide_mini))

    """
    ========================================================================================================================
                         2 OPTIMIZATION MODE
    ========================================================================================================================
    """
    if error_first > args.error_threshold:
        print("Inverse function mode does not work well, state estimation:{},error:{}".format(state_est_first,
                                                                                              error_first))

        """
        --------------- 2.1 Initialize networks, buffers, recordings and optimizer----------------------------------------------
        """
        error_net_1, error_net_2, optimizer_state, optimizer_error1, optimizer_error2, lr_scd_action, loss_error_net = \
            optimization_network_config(state_estimator=state_estimator, LR_state=5.e-5, LR_error=5.e-5,
                                        input_size=len(args.observation_meaning),action_size=state_size,num_classes=1)

        random_walk_buffer = ReplayBuffer()
        in_situ_buffer = ReplayBuffer()

        gradient_recording = pandas_recording(
            column=args.state_meaning + ['error', 'est_error', 'state_loss', 'error_loss'],
            save_dir='output/network_recording_'+str(index_test)+'.csv')
        # %%
        for i in range(0, args.num_iteration):
            # ---------------------------2.2.1 collect data for buffers----------------------------
            # -------------------------------estimation from network--------------------------------
            est_state_norm_tc = state_estimator(actual_observation4net)  # observation-->state estimation
            est_state_norm = est_state_norm_tc.detach().cpu().squeeze().numpy()  # state estimation: torch-->numpy
            rebuild_observation, est_state, error_sum = physics_evaluation_module(
                state_norm=est_state_norm,
                data_norm_store=data_norm_store,
                forward_physical_model=forward_physical_model)

            # -------------------random walk buffer---------------------------------
            if i == 0:
                for index in range(0, global_search_step):
                    random_state_norm = np.random.rand(data_norm_store.state_size)
                    random_observation, random_state, random_error = physics_evaluation_module(
                        state_norm=random_state_norm,
                        data_norm_store=data_norm_store,
                        forward_physical_model=forward_physical_model)
                    random_walk_buffer.put((random_observation, random_state_norm, random_error))
            else:
                random_state_norm = np.random.rand(data_norm_store.state_size)
                random_observation, random_state,  random_error = physics_evaluation_module(
                    state_norm=random_state_norm,
                    data_norm_store=data_norm_store,
                    forward_physical_model=forward_physical_model)
                random_walk_buffer.put((random_observation, random_state_norm, random_error))
                # random_walk_buffer.put((random_observation, random_state_norm, random_error))
            # ----------------in-situ buffer---------------------------------------
            noise = np.clip(np.random.normal(0, 0.2, state_size), -0.2, +0.2)
            noise_state_norm = est_state_norm + noise
            noise_observation, noise_state,  noise_error = physics_evaluation_module(
                state_norm=noise_state_norm,
                data_norm_store=data_norm_store,
                forward_physical_model=forward_physical_model)
            in_situ_buffer.put((noise_observation, noise_state_norm, noise_error))
            # -------------------------2.2.2 sample and update----------------------------------------------
            _, state_set, error_set = sample_from_buffer_v5(i=i, in_situ_memory=in_situ_buffer,
                                                            global_search_memory=random_walk_buffer,
                                                            current_observation=rebuild_observation,
                                                            current_norm_state=est_state_norm,
                                                            current_error=error_sum,
                                                            batch_size=args.batch_size,
                                                            partition=0.5)

            loss_error, loss_state, estimated_error = online_update(error_net_1=error_net_1, error_net_2=error_net_2,
                                                                    obverstion_for_net=actual_observation4net,
                                                                    state_net=state_estimator,
                                                                    state_set=state_set,
                                                                    error_set=error_set,
                                                                    data_norm_store=data_norm_store,
                                                                    error_net_optimizer=optimizer_error1,
                                                                    error_net_2_optimizer=optimizer_error2,
                                                                    state_net_optimizer=optimizer_state,
                                                                    loss_error_net=loss_error_net,
                                                                    delay=5, current_epoch=i)

            lr_scd_action.step(i)
            gradient_recording_row_list = np.hstack((est_state.flatten(), error_sum, estimated_error.cpu().item(),
                                                     loss_state.cpu().item(), loss_error.cpu().item())).tolist()
            gradient_recording.add_values(gradient_recording_row_list)
            # -----------------------validation-------------------------------------
            if error_sum < best_error:
                best_state_norm = est_state_norm
                best_state = data_norm_store.state_unnorm_feasible(best_state_norm)
                best_observation = rebuild_observation
                best_error = error_sum
                optimizaton_provide_mini = "gradient_optimization"

            optimization_recording_row_list = np.hstack((best_state.flatten(), observation_rebuilt_first.flatten(), best_error)).tolist()
            optimization_recording.add_values(optimization_recording_row_list)

            print(
                "current epoch: {},current estimation: {}, current actual error: {}, current_pred_erro:{}, current best error: {}".
                    format(i, est_state, error_sum, estimated_error.cpu().item(), best_error))
            if best_error < args.error_threshold or i == args.num_iteration - 1:
                print("Optimization is done, best state: {}, best error:{}, the result is provided by {}".
                      format(best_state, best_error, optimizaton_provide_mini))
                break


if __name__ == '__main__':
    for index_test in range(0, 100):
        print("The {}th experiment is running".format(index_test))
        main(index_test)
