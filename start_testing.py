import copy
import sys
import timeit

import cplex_methods as cpx
import kmeans_processing as kmp
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import Load_Reordered_Testing_Model as lrtm
import os
import utils as ut
import knapsack_methods as knp
import rl_metods as rlm
import numpy as np

#####HERE GOES THE MAIN LOOP OF THE CODE

# info to be able to load the right model
training_steps = 5500000
model_name = 'A2C'
number_of_loops = 30
sol_loops = 1000

kmean_all_obj = []
cplex_all_obj = []
rl_all_obj = []
rl_only_all_obj = []

kmean_all_time = []
cplex_all_time = []
rl_all_time = []
rl_only_all_time = []

kmean_all_gaps = []
rl_all_gaps = []
rl_only_all_gaps = []
cplex_all_gaps = []

kmean_all_correct_items = []
rl_all_correct_items = []
rl_only_all_correct_items = []
cplex_all_correct_items = []

default_prediction_perc = []
default_prediction_perc_a1 = []
default_prediction_perc_a2 = []

rl_all_gaps_a1 = []
rl_all_time_a1 = []
rl_all_obj_a1 = []
rl_all_correct_items_a1 = []

rl_all_gaps_a2 = []
rl_all_time_a2 = []
rl_all_obj_a2 = []
rl_all_correct_items_a2 = []

p1_85_all_gaps = []
p1_85_all_correct_items = []
p1_85_all_obj = []
p1_85_all_time = []
default_prediction_perc_p1 = []

p1_95_all_gaps = []
p1_95_all_correct_items = []
p1_95_all_obj = []
p1_95_all_time = []
default_prediction_perc_p2 = []

log_dir = "model_and_env/"
fn = 'testing_2d_distinct_knn_best' + "_" + str(model_name) + "_10072022.txt"
instance_loc = "G:/My Drive/Projects/07 - RL for MKP/Test Instances/generated ins/"

sys.stdout = open(fn, 'w')


def __main__():
    curr_best_gap_rl = -1
    curr_best_rl_sol = []
    curr_best_obj_rl = -1
    cnt = 0
    #    best_sol_from_rl.clear()
    #    print("Looop Nr: ", instance)

    # testing instaces
    size_arr = [10, 20, 30]

    for i in range(3):
        number_of_items = size_arr[i] ** 2
        num_of_const = number_of_items
        number_of_clusters = int(number_of_items / 25)
        run_id = str(model_name) + '_' + str(100) + '_' + str(training_steps)
        stats_path = os.path.join(log_dir, str(run_id) + "_vec_normalize_multiInstance.pkl")

        # GENERATE INSTANCE - cost values, constraints and rhs
        # cost_values, constraints, rhs = knp.instance_generator(number_of_items, num_of_const)

        # read saved instances and solution stats - if any is saved
        cost_values, constraints, rhs, ind = knp.read_instances(instance_loc, number_of_items, 1, 100)

        cpx_obj, cpx_sol, cpx_gap = cpx.load_cpx_sols(instance_loc, number_of_items, 1, 100)
        # SOLVE USING CPLEX
        start_cplex_timer = timeit.default_timer()
        # cpx_obj, cpx_sol, cpx_gap = ut.apply_standard_cplex(constraints,cost_values,number_of_items,num_of_const)
        # print("objective using CPLEX: ", cpx_obj)
        # print("solution using CPLEX: ", cpx_sol)

        end_cplex_timer = timeit.default_timer()
        cplex_all_gaps.append(cpx_gap)
        cplex_all_obj.append(cpx_obj)
        cplex_all_time.append(end_cplex_timer - start_cplex_timer)

        sol_kmean, obj_kmean, time_kmean = kmp.kmeans_start(constraints, cost_values, number_of_items, num_of_const,
                                                            number_of_loops, number_of_clusters, sol_loops, cpx_obj, cpx_sol, cpx_gap)

        cnt_kmean = 0
        for i in range(0, len(sol_kmean)):
            if sol_kmean[i] == cpx_sol[i]:
                cnt_kmean += 1
        print("Number of common items for KNN is : ", cnt_kmean)
        kmean_all_correct_items.append(cnt_kmean)
        kmean_all_obj.append(obj_kmean)
        kmean_all_time.append(time_kmean)
        kmean_gap = ((obj_kmean - cpx_obj) / cpx_obj) * 100
        kmean_all_gaps.append(kmean_gap)

        full_instance = knp.combine_cost_and_constraints(cost_values, constraints)
        ind = knp.get_index_matrix(cost_values, constraints, rhs, number_of_items, num_of_const)
        # Load the saved statistics
        reordered_testing = lrtm.Load_Reordered_Testing_Model(number_of_items, constraints, cost_values, ind,
                                                              full_instance, training_steps, run_id, rhs, sol_kmean,
                                                              obj_kmean, stats_path)
        testing_env = reordered_testing.create_initial_model()
        reordered_model = reordered_testing.load_trained_model(model_name, str(run_id), log_dir)
        #    model = A2C.load(log_dir + str(algo) +"knp_env_multiInstance")
        env = VecNormalize.load(stats_path, testing_env)
        # Test the trained agent

        cnt_rl, curr_best_gap_rl, curr_best_obj_rl, best_sol_from_rl, time_solving, best_sol_from_rl_inverted = rlm.prediction_loop(
            sol_loops, reordered_model, env, cpx_obj, cpx_sol)
        rl_only_all_correct_items.append(copy.deepcopy(cnt_rl))
        rl_only_all_gaps.append(copy.deepcopy(curr_best_gap_rl))
        rl_only_all_obj.append(copy.deepcopy(curr_best_obj_rl))
        rl_only_all_time.append(time_solving)

        # assessing pred percentages
        # 1
        pred_perc, sel_1, nsel_1 = rlm.assert_default_prediction(best_sol_from_rl_inverted, ind)
        default_prediction_perc.append(pred_perc)

        cnt_1, rl_obj_1, per_change_1, time_solving_1 = rlm.get_partial_pred_sol(best_sol_from_rl, cpx_obj, cpx_sol,
                                                                                 constraints, cost_values,
                                                                                 number_of_items, num_of_const, sel_1,
                                                                                 nsel_1)
        rl_all_correct_items.append(cnt_1)
        rl_all_obj.append(rl_obj_1)
        rl_all_gaps.append(per_change_1)
        rl_all_time.append(time_solving_1)

        # 2
        pred_perc_a1, sel_2, nsel_2 = rlm.assert_default_prediction(best_sol_from_rl_inverted, ind)
        default_prediction_perc_a1.append(pred_perc_a1)

        cnt_2, rl_obj_2, per_change_2, time_solving_2 = rlm.get_partial_pred_sol(best_sol_from_rl, cpx_obj, cpx_sol,
                                                                                 constraints, cost_values,
                                                                                 number_of_items, num_of_const, sel_2,
                                                                                 nsel_2)
        rl_all_correct_items_a1.append(cnt_2)
        rl_all_obj_a1.append(rl_obj_2)
        rl_all_gaps_a1.append(per_change_2)
        rl_all_time_a1.append(time_solving_2)

        # Here we define 85% predictions
        pred_perc_85_perc, sel_3, nsel_3 = rlm.assert_default_prediction_85_perc(best_sol_from_rl_inverted, ind)
        default_prediction_perc_p1.append(pred_perc_85_perc)
        cnt_3, rl_obj_3, per_change_3, time_solving_3 = rlm.get_partial_pred_sol(best_sol_from_rl, cpx_obj, cpx_sol,
                                                                                 constraints, cost_values,
                                                                                 number_of_items, num_of_const, sel_3,
                                                                                 nsel_3)
        p1_85_all_correct_items.append(cnt_3)
        p1_85_all_obj.append(rl_obj_3)
        p1_85_all_gaps.append(per_change_3)
        p1_85_all_time.append(time_solving_3)

        # Here se define 95% predictions
        pred_perc_95_perc, sel_4, nsel_4 = rlm.assert_default_prediction_95_perc(best_sol_from_rl_inverted, ind)
        default_prediction_perc_p2.append(pred_perc_95_perc)
        cnt_4, rl_obj_4, per_change_4, time_solving_4 = rlm.get_partial_pred_sol(best_sol_from_rl, cpx_obj, cpx_sol,
                                                                                 constraints, cost_values,
                                                                                 number_of_items, num_of_const, sel_4,
                                                                                 nsel_4)
        p1_95_all_correct_items.append(cnt_4)
        p1_95_all_obj.append(rl_obj_4)
        p1_95_all_gaps.append(per_change_4)
        p1_95_all_time.append(time_solving_4)

        pred_perc_a2, sel_5, nsel_5 = rlm.assert_default_prediction(best_sol_from_rl_inverted, ind)
        default_prediction_perc_a2.append(pred_perc_a2)

        cnt_5, rl_obj_5, per_change_5, time_solving_5 = rlm.get_partial_pred_sol(best_sol_from_rl, cpx_obj, cpx_sol,
                                                                                 constraints, cost_values,
                                                                                 number_of_items, num_of_const, sel_5,
                                                                                 nsel_5)
        rl_all_correct_items_a2.append(cnt_5)
        rl_all_obj_a2.append(rl_obj_5)
        rl_all_gaps_a2.append(per_change_5)
        rl_all_time_a2.append(time_solving_5)

    print("Cplex Objectives: ", cplex_all_obj)
    print("Kmeans Objectives: ", kmean_all_obj)
    print("Rl Objectives: ", rl_all_obj)
    print("Cplex Time: ", cplex_all_time)
    print("Kmeans Time: ", kmean_all_time)
    print("Rl Time: ", rl_all_time)
    print("Kmeans Gaps: ", kmean_all_gaps)
    print("Rl Gaps: ", rl_all_gaps)
    print("RL Only Time: ", rl_only_all_time)
    print("Rl Only Gaps: ", rl_only_all_gaps)

    print("=============================================================")

    print("mean cplex gap", np.mean(cplex_all_gaps))
    print("mean Kmeans gap", np.mean(kmean_all_gaps))
    print("mean rl gap", np.mean(rl_all_gaps))
    print("mean rl only gap", np.mean(rl_only_all_gaps))
    print("=============================================================")

    print("mean cplex time", np.mean(cplex_all_time))
    print("mean Kmeans time", np.mean(kmean_all_time))
    print("mean rl time", np.mean(rl_all_time))
    print("mean rl only time", np.mean(rl_only_all_time))
    print("=============================================================")
    # print("mean cplex items", np.mean(cplex_all_time))
    print("mean Kmeans items", np.mean(kmean_all_correct_items))
    print("mean rl items", np.mean(rl_all_correct_items))
    print("mean rl only items", np.mean(rl_only_all_correct_items))

    print("=============================================================")
    print("mean prediction percentage for default: ", np.mean(default_prediction_perc))
    print("mean prediction percentage for default a1: ", np.mean(default_prediction_perc_a1))
    print("mean prediction percentage for default a2: ", np.mean(default_prediction_perc_a2))

    print("mean prediction percentage for 85: ", np.mean(default_prediction_perc_p1))
    print("mean prediction percentage for 95: ", np.mean(default_prediction_perc_p2))

    print("=============================================================")

    print("Average objective value cpx: ", np.mean(cplex_all_obj))
    print("Average objective value knn: ", np.mean(kmean_all_obj))
    print("Average objective value RL: ", np.mean(rl_all_obj))
    print("Average objective value RL_Only: ", np.mean(rl_only_all_obj))

    print("=======================Default (RL) Prediction======================================")

    print("Average objective value default: ", np.mean(rl_all_obj))
    print("Average predicted items default: ", np.mean(rl_all_correct_items))
    print("Average gap default: ", np.mean(rl_all_gaps))
    print("Average time default: ", np.mean(rl_all_time))

    print("=======================Default (RL) Prediction Allowance = 1 ======================================")

    print("Average objective value default: ", np.mean(rl_all_obj_a1))
    print("Average predicted items default: ", np.mean(rl_all_correct_items_a1))
    print("Average gap default: ", np.mean(rl_all_gaps_a1))
    print("Average time default: ", np.mean(rl_all_time_a1))

    print("=======================Default (RL) Prediction Allowance = 2 ======================================")

    print("Average objective value default: ", np.mean(rl_all_obj_a2))
    print("Average predicted items default: ", np.mean(rl_all_correct_items_a2))
    print("Average gap default: ", np.mean(rl_all_gaps_a2))
    print("Average time default: ", np.mean(rl_all_time_a2))

    print("=======================85% Prediction======================================")

    print("Average objective value 85%: ", np.mean(p1_85_all_obj))
    print("Average predicted items 85%: ", np.mean(p1_85_all_correct_items))
    print("Average gap 85%: ", np.mean(p1_85_all_gaps))
    print("Average time 85%: ", np.mean(p1_85_all_time))

    print("=======================95% Prediction======================================")
    print("Average objective value 95%: ", np.mean(p1_95_all_obj))
    print("Average predicted items 95%: ", np.mean(p1_95_all_correct_items))
    print("Average gap 95%: ", np.mean(p1_95_all_gaps))
    print("Average time 95%: ", np.mean(p1_95_all_time))


__main__()
