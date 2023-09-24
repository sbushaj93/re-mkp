import sys
import copy
import utils as ut
import Create_Reordered_Training_Model as crtm
# import cplex_methods as cpx
import knapsack_methods as knp
import os
import kmeans_processing as kmp

training_steps = 3500000
algorithm = 'A2C'

# location of saved img:
mkp_loc = os.path.dirname(os.path.abspath(__file__)) + '\\mkp_data\\runs'

# image array for vid
images = []
# keeps global indices
curr_ind_for_ins = []

best_sol_from_rl = []
best_obj_from_rl = []

# solutions and objectives
sol_rl = []
sol_ = []
obj_rl = 0
sol_full = []
obj_full = 0

ins_run_number = 0
number_of_loops = 30
optimality_gap = 0.001
opt_gap_m1 = 0.01
algo = "A2C_knn_10072022_100"
sol_loops = 1000

# fn = 'training_2d_distinct_knn_best'+"_"+str(algo)+"_"+str(rl_loops)+"_"+ str(number_of_items)+'_'+str(num_of_const)+'_'+str(number_of_clusters)+'_'+str(number_of_loops)+'_'+str(opt_gap_m1)+'.txt'
fn = 'training_2d_distinct_knn_best' + "_" + str(algo) + "_10072022.txt"
sys.stdout = open(fn, 'w')


def __main__():
    size_arr = [20, 30]

    for i in range(len(size_arr)):
        number_of_items = size_arr[i] ** 2
        num_of_const = number_of_items
        number_of_clusters = int((number_of_items) / 25)
        run_id = str(algorithm) + '_' + str(number_of_items) + '_' + str(training_steps)

        # GENERATE INSTANCE - cost values, constraints and rhs
        cost_values, constraints, rhs = knp.instance_generator(number_of_items, num_of_const)
        # print(constraints)

        # SOLVE USING CPLEX
        cpx_obj, cpx_sol, gap = ut.apply_standard_cplex(constraints, cost_values, number_of_items, num_of_const)
        print("objective using CPLEX: ", cpx_obj)
        print("solution using CPLEX: ", cpx_sol)
        print("gap using CPLEX: ", gap)

        sol_kmean, obj_kmean, time_solving = kmp.kmeans_start(constraints, cost_values, number_of_items, num_of_const,
                                                number_of_loops, number_of_clusters, sol_loops, cpx_obj, cpx_sol, gap)

        full_instance = knp.combine_cost_and_constraints(cost_values, constraints)
        ind = knp.get_index_matrix(cost_values, constraints, rhs, number_of_items, num_of_const)
        reordered_training = crtm.Create_Reordered_Training_Model(number_of_items, constraints, cost_values, ind,
                                                                  full_instance, training_steps, run_id, rhs, sol_kmean,
                                                                  obj_kmean)
        reordered_env = reordered_training.create_initial_env()
        reordered_model = reordered_training.create_model()
        reordered_training.train_reordered_model(reordered_model)

        reordered_training.save_trained_model(reordered_model)



__main__()
