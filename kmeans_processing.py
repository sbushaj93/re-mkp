import copy
import timeit

import cplex_methods as cpx
import kmeans_internal as kmi
import knapsack_methods as knp
import pandas as pd
import numpy as np


def kmeans_start(const_list, cost_values, number_of_items, num_of_const, number_of_loops, number_of_clusters, sol_loops,obj_full, sol_full, gap_full):
    in_problem = []

    #obj_full, sol_full, gap_full = cpx.solve_using_cplex(const_list,cost_values,number_of_items,num_of_const)
    #print("objective using CPLEX: ", obj_full)


    cons_inst = copy.deepcopy(const_list)

    #this maybe needs to change
    arr = pd.DataFrame(cons_inst).apply(pd.Series)
    arr = arr.rename(columns=lambda x : 'item_'+str(x))

    #list to remain constant to check for feasibility
    static_list_of_const = copy.deepcopy(arr.to_numpy().tolist())

    #number_of_clusters, data_to_cluster, num_of_iter, num_of_items
    res0 = kmi.knn_constraint_out(number_of_clusters, arr, number_of_loops, number_of_items)


    r=[]
    for it in range (1, number_of_clusters):
     #   a = np.array(list(list(res0[1])[i][1]["Constraint"]))
        a = res0[5][it]
      #  if a not in r:
        idx_max = np.argmax(a)
        idx_min = np.argmin(a)
      #  print(res0[4][i][0])
        r.append(res0[4][it][idx_max].tolist())
        r.append(res0[4][it][idx_min].tolist())


    cons_list_3 = []
    for it in range(0, len(r)):
     #   if(const_list[i] not in in_problem):
         in_problem.append(r[it])



    startRecursiveSolving = timeit.default_timer()
    lp = 0
    feasible = False
    best_obj_knn = 0
    best_sol_knn = [0]*number_of_items
    prev_gap = -100
    violated_const = []
    violation_index=[]
    violation_amount = []
    satisfied_const = []
    satisfied_index=[]
    satisfied_amount = []

    while(feasible != True and lp < sol_loops):
        #print("LOOP Number ", lp)
        lp+=1

        newObjective, newSolution = cpx.solve_using_cplex_reduced_knn(in_problem,cost_values,number_of_items,len(in_problem))
        #print("objective using CPLEX reduced: ", newObjective)

        new_per_change = ((newObjective-obj_full)/obj_full)*100

        #print("Percentage change is: ", new_per_change)


        cnt_l=0
        for item in range(0,len(sol_full)):
            if abs(newSolution[item]) == abs(sol_full[item]):
                cnt_l+=1
        #print("Number of common items is : ", cnt_l)
          # if (best_obj_knn == 0):
          #     best_obj_knn = newObjective
          #     best_sol_knn = newSolution
          # else:
          #     if(prev_gap < 0):
          #         if(prev_gap > new_per_change):
          #             best_obj_knn = newObjective
          #             best_sol_knn = newSolution
          #         else:
          #             if(prev_gap<new_per_change):
          #                 best_obj_knn = newObjective
          #                 best_sol_knn = newSolution

         # if (prev_gap < 0):
         #     if (prev_gap<new_per_change):
         #         prev_gap=new_per_change
         # else:
         #     if (prev_gap>new_per_change):
         #         prev_gap=new_per_change

           #  if (new_per_change > -0.0001):
           #      feasible = True
           #      break
         #here we check for violated constraints and feasibility
        violated_const.clear()
        violation_index.clear()
        violation_amount.clear()
        satisfied_const.clear()
        satisfied_index.clear()
        satisfied_amount.clear()
        con_vio_idx = 0
        # con_sat_idx = 0
        for item in range(len(static_list_of_const)):
            diff = 3/4*sum(static_list_of_const[item]) - sum([a*b for a,b in zip(static_list_of_const[item],newSolution)])
            if (diff>0):
                #print("Constraint ", i, " is not satisfied")
                violated_const.append(static_list_of_const[item])
                #the higher the difference in the second parameter the larger the violation
                violation_index.append(con_vio_idx)
                violation_amount.append(diff)
                con_vio_idx+=1
         #       else:
         #           satisfied_const.append(static_list_of_const[item])
         #           #the higher the difference in the second parameter the larger the violation
         #           satisfied_index.append(con_sat_idx)
         #           satisfied_amount.append(diff)
         #           con_sat_idx+=1
         #this considers all constraints of the original problem
        for item_in_problem in range(len(static_list_of_const)):
            dif = 3/4*sum(static_list_of_const[item_in_problem]) - sum([a*b for a,b in zip(static_list_of_const[item_in_problem],newSolution)])
            if (dif <= 0):
                satisfied_const.append(static_list_of_const[item_in_problem])
                #the higher the difference in the second parameter the larger the violation
                satisfied_index.append(item_in_problem)
                satisfied_amount.append(diff)
          #  con_sat_idx+=1

          #this is only considering const in problem
        # for item_in_problem in range(len(in_problem)):
        #     diff = 3/4*sum(in_problem[item_in_problem]) - sum([a*b for a,b in zip(in_problem[item_in_problem],newSolution)])
        #     satisfied_const.append(in_problem[item_in_problem])
        #     #the higher the difference in the second parameter the larger the violation
        #     satisfied_index.append(item_in_problem)
        #     satisfied_amount.append(diff)
        #   #  con_sat_idx+=1

         #remove the most loose constraints in the new model
       #     for j in range(3):
       #         min_sat = np.argmin(satisfied_amount)
       #         idx_sat = satisfied_index[min_sat]
       #         not_in_problem.append(copy.deepcopy(satisfied_const[idx_sat]))
       #         del in_problem[idx_sat]
       #         del satisfied_const[idx_sat]
       #         del satisfied_amount[min_sat]
              #  del in_problem[in_problem.index(satisfied_const[idx_sat])]

        #print(violation_index)
        #print(violation_amount)
        #print("Violated Constraints: ", len(violated_const))
        #print("Satisfied Constraints: ", len(satisfied_const))

        """
         In this part we can get the constraint which was violated the most or the least
         --TODO : we can also remove 1 constraint when we add new const   !!!
        """
        if (len(violated_const)==0):
             feasible = True
             last_sol_knn = newSolution
             last_obj_knn = newObjective
             break
        elif (len(violated_const) > 20):
            #print("The list with differences is: ")
            #print(violation_amount)
            for cns in range(6):
                #each loop add the most violated constraint
                #rhs - lhs is +, so the larger the more violated
                max_vio = np.argmax(violation_amount)
                idx_vio = violation_index[max_vio]
                in_problem.append(copy.deepcopy(violated_const[idx_vio]))
                #print("added constraint with violation: ", violation_amount[max_vio])
                del violated_const[idx_vio]
                del violation_amount[max_vio]
           #     del violation_index[max_vio]
              #  del static_list_of_const[violated_const[idx_vio]]
        else:
            for x in range(len(violated_const)):
                in_problem.append(copy.deepcopy(violated_const[x]))
                #print("Adding All Const : ",violated_const[x] )


    endRecursiveSolving = timeit.default_timer()
    time_solving = endRecursiveSolving - startRecursiveSolving

    #print("Best objective is: ", best_obj_knn)
    #print("Best solution is: ", best_sol_knn)
    #print("LAST objective is: ", last_obj_knn)
    #print("LAST solution is: ", last_sol_knn)

    #    print("LAST objective is: ", obj_knn)
    #    print("LAST solution is: ", sol_knn)


    sol_knn = copy.deepcopy(last_sol_knn)
    obj_knn = copy.deepcopy(last_obj_knn)

    cnt_new=0
    for item in range(0,len(sol_full)):
        if abs(last_sol_knn[item]) == abs(sol_full[item]):
            cnt_new+=1

    #print("Number of common items is : ", cnt_new)


    #print("Lowest Gap: ", prev_gap)
    #print("Time clustering: ", res0[3])
    #print("Time solving: ", timeSolving)
    return  sol_knn, obj_knn, time_solving
