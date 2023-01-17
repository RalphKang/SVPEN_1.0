import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction

from data_operation import *
from network_management import load_state_estimator, optimization_network_config, online_update
from physical_model import forward_model_radis, physics_evaluation_module
import argparse
import os

def main(feasible_domain, ground_truth_state, test_sample_index):
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iteration', type=int, default=500, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--random_walk_prestore', type=bool, default=False, help='whether to use global prestore state')
    parser.add_argument('--prestore_dir', type=str, default="buffer_storage/global_search_buffer.pkl",
                        help='prestore dir')
    parser.add_argument('--error_threshold', type=float, default=0.10, help='error threshold')
    parser.add_argument('--delay_update', type=int, default=4, help='delay')
    parser.add_argument('--error_zoomer', type=float, default=0.4, help='error zoomer')
    parser.add_argument('--feasible_domain_state', type=np.ndarray, default=np.array([[100.0, 0.01], [1000, 0.1]]),
                        help='feasible domain')
    parser.add_argument('--ground_truth_state', type=np.ndarray, default=np.array([300, 0.03]), help='ground truth')
    parser.add_argument('--state_meaning', type=list, default=['Temperature', 'mole fraction'], help='state meaning')
    parser.add_argument('--record_path', type=str, default='output/best_state_recording.csv', help='record path')
    args = parser.parse_args()

    args.feasible_domain_state = feasible_domain
    args.ground_truth_state = ground_truth_state
    global_search_step = int(args.batch_size)  # the number of initial global search steps

    optimization_recording = pandas_recording(column=args.state_meaning + ['error'],
                                              save_dir='output/best_state_recording_' + str(test_sample_index) + '.csv')
    wave_need = np.arange(2375.0, 2395.0 + 0.1, 0.1)
    state_size = len(args.ground_truth_state)
    # %% Obtain the test sample
    forward_physical_model = forward_model_radis(molecule="CO2", waveneed=wave_need,
                                                 lightpath=10,
                                                 databank='hitemp',
                                                 spec_type='absorptivity')  # initialize physical model
    actual_observation = forward_physical_model.generate_sample(args.ground_truth_state)  # generate test observation
    data_norm_store = normalization_data_store(norm_dir4state="buffer_storage/normalization_data/label_standarder.csv",
                                               norm_dir4observation="buffer_storage/normalization_data/spectrum_standarder.csv",
                                               feasible4state=args.feasible_domain_state,
                                               observation=actual_observation,
                                               state_size=state_size,
                                               error_zoomer=args.error_zoomer)  # store all normalization boundaries, feasible boundaries
    normed_feasible_domain = data_norm_store.state_normalization(args.feasible_domain_state)
    data_norm_store.feasible_domain = normed_feasible_domain
    # %%
    """
    ===========================================================================================
                         1 INVERSE FUNCTION MODE
    ===========================================================================================
    """
    state_estimator = load_state_estimator()  # load state estimator
    actual_observation4net = data_norm_store.obse_trans_phy2state_est(actual_observation)  # observation transform
    actual_observation4net = actual_observation4net.to('cuda')
    state_est_first_norm_tc = state_estimator(actual_observation4net)  # observation-->state estimation
    state_est_first_norm = state_est_first_norm_tc.detach().cpu().squeeze().numpy()  # state estimation: torch-->numpy
    observation_rebuilt_first, state_est_first, error_group_first, error_first = physics_evaluation_module(
        state_norm=state_est_first_norm,
        data_norm_store=data_norm_store,
        forward_physical_model=forward_physical_model)
    best_error = error_first
    best_state_norm = state_est_first_norm
    best_state = state_est_first

    optimization_recording_row_list = np.hstack(
        (best_state.flatten(), best_error)).tolist()
    optimization_recording.add_values(added_data=optimization_recording_row_list)
    i = 0
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
            optimization_network_config(state_estimator=state_estimator, LR_state=5.e-5, LR_error=5.e-5)

        random_walk_buffer = ReplayBuffer(prestore=args.random_walk_prestore, buffer_dir=args.prestore_dir)
        in_situ_buffer = ReplayBuffer()

        gradient_recording = pandas_recording(
            column=args.state_meaning + ['error', 'est_error', 'state_loss', 'error_loss'],
            save_dir='output/network_recording_'+str(test_sample_index)+'.csv')
        # %%
        for i in range(0, args.num_iteration):
            # ---------------------------2.2.1 collect data for buffers----------------------------
            # -------------------------------estimation from network--------------------------------
            est_state_norm_tc = state_estimator(actual_observation4net)  # observation-->state estimation
            est_state_norm = est_state_norm_tc.detach().cpu().squeeze().numpy()  # state estimation: torch-->numpy
            rebuild_observation, est_state, physical_error_est, error_sum = physics_evaluation_module(
                state_norm=est_state_norm,
                data_norm_store=data_norm_store,
                forward_physical_model=forward_physical_model)

            # -------------------random walk buffer---------------------------------
            if i == 0 and args.random_walk_prestore == False:
                for index in range(0, global_search_step):
                    random_state_norm = np.random.rand(data_norm_store.state_size)
                    random_observation, random_state, random_physcial_error, random_error = physics_evaluation_module(
                        state_norm=random_state_norm,
                        data_norm_store=data_norm_store,
                        forward_physical_model=forward_physical_model)
                    random_walk_buffer.put((random_observation, random_state_norm, random_physcial_error))
            else:
                random_state_norm = np.random.rand(data_norm_store.state_size)
                random_observation, random_state, random_physcial_error, random_error = physics_evaluation_module(
                    state_norm=random_state_norm,
                    data_norm_store=data_norm_store,
                    forward_physical_model=forward_physical_model)
                random_walk_buffer.put((random_observation, random_state_norm, random_physcial_error))
                # random_walk_buffer.put((random_observation, random_state_norm, random_error))
            # ----------------in-situ buffer---------------------------------------
            noise = np.clip(np.random.normal(0, 0.2, 2), -0.2, +0.2)
            noise_state_norm = est_state_norm + noise
            noise_observation, noise_state, noise_physical_error, noise_error = physics_evaluation_module(
                state_norm=noise_state_norm,
                data_norm_store=data_norm_store,
                forward_physical_model=forward_physical_model)
            in_situ_buffer.put((noise_observation, noise_state_norm, noise_physical_error))
            # -------------------------2.2.2 sample and update----------------------------------------------
            _, state_set, error_set = sample_from_buffer_v5(i=i, in_situ_memory=in_situ_buffer,
                                                            global_search_memory=random_walk_buffer,
                                                            current_observation=rebuild_observation,
                                                            current_norm_state=est_state_norm,
                                                            current_error=physical_error_est,
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
                best_state = data_norm_store.state_unnorm(best_state_norm)
                best_observation = rebuild_observation
                best_error = error_sum
                optimizaton_provide_mini = "gradient_optimization"

            optimization_recording_row_list = np.hstack((best_state.flatten(), best_error)).tolist()
            optimization_recording.add_values(optimization_recording_row_list)

            print(
                "current epoch: {},current estimation: {}, current actual error: {}, current_pred_erro:{}, current best error: {}".
                    format(i, est_state, error_sum, estimated_error.cpu().item(), best_error))
            if best_error < args.error_threshold or i == args.num_iteration - 1:
                print("Optimization is done, best state: {}, best error:{}, the result is provided by {}".
                      format(best_state, best_error, optimizaton_provide_mini))
                break
    return i


if __name__ == '__main__':
    # read feasible_domain
    test_sample_file_dir = 'feasible_domain_ground_truth_state.csv'
    test_sample_file_pd = pd.read_csv(test_sample_file_dir)
    test_sample_file = test_sample_file_pd.values
    recording = pd.DataFrame(columns=['need_iteration'])
    for i in range(len(test_sample_file)):
        feasible_domain = np.array(
            [[test_sample_file[i, 2], test_sample_file[i, 4]], [test_sample_file[i, 1], test_sample_file[i, 3]]])
        test_state = np.array([test_sample_file[i, 5], test_sample_file[i, 6]])
        need_iteration = main(feasible_domain, test_state, i)
        recording.loc[i] = [need_iteration]
        recording.to_csv('need_iteration.csv')
