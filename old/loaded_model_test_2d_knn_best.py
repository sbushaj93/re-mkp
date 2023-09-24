# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:43:29 2021

@author: sb2386
"""

import random
import copy
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, SGD
import cplex
import math
import random
from timeit import default_timer as timer
import time
import sys
import gym
from gym import spaces
import numpy as np
from stable_baselines.common.policies import MlpPolicy,CnnLnLstmPolicy,CnnPolicy,MlpLnLstmPolicy,MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import SAC
from stable_baselines import ACKTR, DQN
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import make_vec_env
import pickle
import os
from math import sqrt  # For returning the square root of a number or calculation
import random  # For the choosing of our random centroids to begin with
import numpy as np  # For various data wrangling tasks
import matplotlib.pyplot as plt  # For plots to be generated
import seaborn as sns  # For plots, their styling and colouring.
import pandas as pd  # For reading of the data and drawing inferences from it
pd.options.display.max_rows = 4000  # To display all the rows of the data frame
import sys
from scipy.spatial.distance import pdist, squareform
import timeit

dist_num = 100
current_instance_number = -1
number_of_loops = 20
number_of_items = 30**2
number_of_clusters = int((number_of_items)/25)
num_of_const = number_of_items
optimality_gap = 0.001
opt_gap_m1 = 0.01
opt_gap_m2 = 0.001
algo = "ACKTR_knn_2d_0.08_100"
rl_loops = 505


best_sol_from_rl = []
best_obj_from_rl = []
best_sol_from_rl_inverted = []

#solutions and objectives
sol_rl = [0]*number_of_items
sol_knn = []
obj_rl = 0
obj_knn = 0
sol_full = [0]*number_of_items
obj_full = 0

#fn = 'd_loaded_model_test_add'+"_"+"_"+str(algo)+"_"+str(rl_loops)+"_"+ str(number_of_items)+'_'+str(num_of_const)+'_'+str(number_of_clusters)+'_'+str(number_of_loops)+'_'+str(opt_gap_m1)+'.txt'
fn = 'final_res'+"_"+ str(dist_num)+"_"+str(algo)+"_"+str(rl_loops)+"_"+ str(number_of_items)+'_'+str(num_of_const)+'_'+str(number_of_clusters)+'_'+str(number_of_loops)+'_'+str(opt_gap_m1)+"_allowence_0_with_reset"+'.txt'

sys.stdout = open(fn, 'w')



log_dir = "model_and_env/"
stats_path = os.path.join(log_dir, str(algo) +"vec_normalize_multiInstance.pkl")

# Load the agent
model = ACKTR.load(log_dir + str(algo) +"knp_env_multiInstance")


#instance generation class
"""
generating instances:
    we generate instances based on the number of items variable
    for ex: 100 items 50 cons, 400 items 200 cons, 900 items 450 cons
"""
class Knp_Ins_Gen():
    def con_generator(number_of_items):
            temp = []
            for x in range(number_of_items):
                tmp = float(random.randint(1,dist_num))
                temp.append(tmp)
            #print(temp)
            return temp
    
    def instance_generator(number_of_items):
        ins = []
        obj_values = []
        #all_con = []
        tst = []
        b = []
        b.append(0)
        for x in range(number_of_items):
            temp = float(random.randint(1,dist_num))
            obj_values.append(temp)
        ins.append(obj_values)
        tst.append(obj_values)
        for y in range(int(num_of_const)):
            for z in range(1):
                #cons.append([])
                con_new = Knp_Ins_Gen.con_generator(number_of_items)
                #all_con.append(con_new)
                rhs = 3/4 * sum(con_new)
                b.append(rhs) 
                #cons[y].append(con_new)
                ins.append(con_new)
                tst.append(con_new)
        return ins,b,tst
        
    
    def sort_based_on_ratios(matrix,rhs):
        cv = matrix[0]
        res = [0]*len(cv)
        ln_mat = len(matrix)
        ln_cv = len(matrix[0])
        
        for i in range(ln_cv):
            # print(i)
            for j in range(1,ln_mat):
            #    print(j)
                res[i] = res[i] + (cv[i]/matrix[j][i]/rhs[j])
                #old
               # res[i] = res[i] + (cv[i]/matrix[j][i])
                
        for i in range(ln_cv):
            res[i] = res[i]/ln_cv
            
        return res
    def saveList(myList,filename):
        # the filename should mention the extension 'npy'
        np.save(filename,myList)
        print("Saved successfully!")
    
    def loadList(filename):
        # the filename should mention the extension 'npy'
        tempNumpyArray=np.load(filename)
        return tempNumpyArray.tolist()
    
    # def sort_based_on_ratios(matrix,rhs):
    #     new_rhs = []
    #     all_array = copy.deepcopy(matrix)
        
    #     cols_sum_before = [sum([all_array[row][i] for row in range(1,len(all_array))]) for i in range(0,len(all_array[0]))]
    
    #     #here we will calculate the ratio of cost value/sum of culumns
    #     cost_to_sm_col_rat = []
    #     all_rat_vals = []
    #     for i in range(0,len(all_array[0])):
    #         cost_to_sm_col_rat.append(cols_sum_before[i]/all_array[0][i])
    #     all_rat_vals.append(cost_to_sm_col_rat)
      
    #     #for each constraint we get the ratio of weight/rhs 
    #     for i in range(1,len(all_array)):
    #         cons_tmp_rat = []
    #         for j in range(0,len(all_array[0])):
    #             cons_tmp_rat.append(all_array[i][j]/rhs[i])
    #         all_rat_vals.append(cons_tmp_rat)
            
    #     cols_sum_after = [sum([all_rat_vals[row][i] for row in range(1,len(all_rat_vals))]) for i in range(0,len(all_rat_vals[0]))]
    
    #     cost_to_sm_col_rat_after = []
    #     for i in range(0,len(all_rat_vals[0])):
    #         cost_to_sm_col_rat_after.append(cols_sum_after[i]/all_rat_vals[0][i])
          
    #     return cost_to_sm_col_rat_after
    
    def Standardize(data):
        return (data - np.mean(data))/np.std(data)

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))


class Cplex_Methods():   
    def solve_using_cplex_reduced_knn(const_list_3,objf,ndec,ncon):
        sn = []
        #sn_old = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
        sn2 = 'G'*len(const_list_3)
        sn.append(sn2)
        #400 items
        #sn = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
    
        lb = []
        ub = []
        var_names = []
        for x in range(number_of_items):
            #temp = float(random.randint(1,100))
            #dec_var.append(temp)
            lb.append(0.0)
            ub.append(1.0)
            nm = "x%d" % (x+1)
            var_names.append(nm)
            #knapsack_gen.write_to_file(str(temp),0)
            
            
        
        con_names = []   
        b_rhs = [] 
        conts_all_with_names = []
        fin_con = []
        #sn = ''
        for y in range(len(const_list_3)):
            #print(y)
            rhs = 3/4 * sum(const_list_3[y])
            #print(rhs)
            b_rhs.append(rhs)
            cn = "c%d"%(y+1)
            con_names.append(cn)
            tmp_con = []
            tmp_con.append(var_names)
            tmp_con.append(const_list_3[y])
            conts_all_with_names.append(tmp_con)
            #fin_con.append(conts_all_with_names)
            #sn = sn + 'G'
            ##cons.append([])
            #for z in range(1):
            #    cons.append([])
            #    cons[y].append(var_names)
            #    #cons[y].append([])
            #    con_new = knapsack_gen.con_generator()
            #    rhs = 3/4 * sum(con_new)
            #    b.append(rhs) 
            #    cons[y].append(con_new)
            #sn = sn + 'G'
        
        #print(b_rhs)
        #number of variables is important, in this case we use hardcoded 4
        cpl  = cplex.Cplex()
        cpl.parameters.timelimit.set(600)
        #set max sense temporarily
        cpl.objective.set_sense(cpl.objective.sense.minimize)
        t= cpl.variables.type
        
        cpl.parameters.mip.tolerances.mipgap.set(float(opt_gap_m1)) 
        
        #decision variables
       
        cpl.variables.add(obj = objf,
                         lb = lb,
                         ub = ub,
                         types = [t.binary]*ndec,
                         names = var_names)
       
        cpl.linear_constraints.add(lin_expr = conts_all_with_names, senses = sn, rhs = b_rhs, names = con_names)
    
     #   cpl.write("model.lp")
        cpl.solve()
       
       
        print(cpl.solution.get_values())
        #for it in cpl.solution.get_values():
        #    sol_val.append(int(it))
        sol_val = cpl.solution.get_values()
        print(cpl.solution.get_objective_value())
        print("Solution status : ", cpl.solution.get_solution_type())
        print("Relative Gap is: ",cpl.solution.MIP.get_mip_relative_gap())
        print("Number of variables : ",cpl.variables.get_num())
        print("Number of binary variables : ",cpl.variables.get_num_binary())
        print("CPLEX GET STATS: ", cpl.get_stats())
     #   print(cpl.rows)
        obj = cpl.solution.get_objective_value()
        print("GAP for solve_using_cplex_reduced_knn: " , cpl.solution.MIP.get_mip_relative_gap())
        return obj,sol_val



    def solve_using_cplex(const_list,objf,ndec,ncon):
        #var_names,con_names, objf, cons, ndec, ncon, b_r, sn, lb, up
        #sn = ['GGGGG']
        #50 constraints
        sn = []
        #sn_old = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
        sn2 = 'G'*int(num_of_const)
        sn.append(sn2)
        #400 items
        #sn = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
    
        lb = []
        ub = []
        var_names = []
        for x in range(number_of_items):
            #temp = float(random.randint(1,100))
            #dec_var.append(temp)
            lb.append(0.0)
            ub.append(1.0)
            nm = "x%d" % (x+1)
            var_names.append(nm)
            #knapsack_gen.write_to_file(str(temp),0)
            
            
        
        con_names = []   
        b_rhs = [] 
        conts_all_with_names = []
        fin_con = []
        #sn = ''
        for y in range(int(num_of_const)):
            #print(y)
            rhs = 3/4 * sum(const_list[y])
            #print(rhs)
            b_rhs.append(rhs)
            cn = "c%d"%(y+1)
            con_names.append(cn)
            tmp_con = []
            tmp_con.append(var_names)
            tmp_con.append(const_list[y])
            conts_all_with_names.append(tmp_con)
            #fin_con.append(conts_all_with_names)
            #sn = sn + 'G'
            ##cons.append([])
            #for z in range(1):
            #    cons.append([])
            #    cons[y].append(var_names)
            #    #cons[y].append([])
            #    con_new = knapsack_gen.con_generator()
            #    rhs = 3/4 * sum(con_new)
            #    b.append(rhs) 
            #    cons[y].append(con_new)
            #sn = sn + 'G'
        
        #print(b_rhs)
        #number of variables is important, in this case we use hardcoded 4
        cpl  = cplex.Cplex()
        cpl.parameters.timelimit.set(7200)
       # cpl.parameters.threads.set(2)
        #set max sense temporarily
        cpl.objective.set_sense(cpl.objective.sense.minimize)
        t= cpl.variables.type
        
      #  cpl.parameters.mip.tolerances.mipgap.set(float(optimality_gap)) 
        
        #decision variables
       
        cpl.variables.add(obj = objf,
                         lb = lb,
                         ub = ub,
                         types = [t.binary]*ndec,
                         names = var_names)
       
        cpl.linear_constraints.add(lin_expr = conts_all_with_names, senses = sn, rhs = b_rhs, names = con_names)
    
      #  cpl.write("model.lp")
        cpl.solve()
       
       
        print(cpl.solution.get_values())
        #for it in cpl.solution.get_values():
        #    sol_val.append(int(it))
        sol_val = cpl.solution.get_values()
        print(cpl.solution.get_objective_value())
        obj = cpl.solution.get_objective_value()
        gap = cpl.solution.MIP.get_mip_relative_gap()
        print("GAP for solve_using_cplex: ",gap)
        
        return obj,sol_val,gap

    def solve_using_cplex_relaxed(const_list,objf,ndec,ncon):
            #var_names,con_names, objf, cons, ndec, ncon, b_r, sn, lb, up
            #sn = ['GGGGG']
            #50 constraints
            sn = []
            #sn_old = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
            sn2 = 'G'*int(num_of_const)
            sn.append(sn2)
            #400 items
            #sn = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
        
            lb = []
            ub = []
            var_names = []
            for x in range(number_of_items):
                #temp = float(random.randint(1,100))
                #dec_var.append(temp)
                lb.append(0.0)
                ub.append(1.0)
                nm = "x%d" % (x+1)
                var_names.append(nm)
                #knapsack_gen.write_to_file(str(temp),0)
                
                
            
            con_names = []   
            b_rhs = [] 
            conts_all_with_names = []
            fin_con = []
            #sn = ''
            for y in range(int(num_of_const)):
                #print(y)
                rhs = 3/4 * sum(const_list[y])
                #print(rhs)
                b_rhs.append(rhs)
                cn = "c%d"%(y+1)
                con_names.append(cn)
                tmp_con = []
                tmp_con.append(var_names)
                tmp_con.append(const_list[y])
                conts_all_with_names.append(tmp_con)
                #fin_con.append(conts_all_with_names)
                #sn = sn + 'G'
                ##cons.append([])
                #for z in range(1):
                #    cons.append([])
                #    cons[y].append(var_names)
                #    #cons[y].append([])
                #    con_new = knapsack_gen.con_generator()
                #    rhs = 3/4 * sum(con_new)
                #    b.append(rhs) 
                #    cons[y].append(con_new)
                #sn = sn + 'G'
            
            #print(b_rhs)
            #number of variables is important, in this case we use hardcoded 4
            cpl  = cplex.Cplex()
           
            #set max sense temporarily
            cpl.objective.set_sense(cpl.objective.sense.minimize)
            t= cpl.variables.type
            
            cpl.parameters.mip.tolerances.mipgap.set(float(opt_gap_m1)) 
            
            #decision variables
           
            cpl.variables.add(obj = objf,
                             lb = lb,
                             ub = ub,
                      #       types = [t.binary]*ndec,
                             names = var_names)
           
            cpl.linear_constraints.add(lin_expr = conts_all_with_names, senses = sn, rhs = b_rhs, names = con_names)
        
         #   cpl.write("model.lp")
            cpl.solve()
           
           
            print(cpl.solution.get_values())
            #for it in cpl.solution.get_values():
            #    sol_val.append(int(it))
            sol_val = cpl.solution.get_values()
            print(cpl.solution.get_objective_value())
            obj = cpl.solution.get_objective_value()
            print(cpl.solution.MIP.get_mip_relative_gap())
            return obj,sol_val


    def solve_using_cplex_reduced_rl(const_list,objf,ndec,ncon,b_sol,curr_ins,sel,nsel,obj_ub, rl_sol):
      #  global current_instance_number
        print("current instance number: ",current_instance_number)
       # self.current_instance_number = current_instance_number
        print("self current instance number: ",curr_ins)
        print("RL SOL: ", rl_sol)
        #var_names,con_names, objf, cons, ndec, ncon, b_r, sn, lb, up
        #sn = ['GGGGG']
        #50 constraints
        sn = []
        #sn_old = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
        sn2 = 'G'*int(num_of_const)
        sn.append(sn2)
        #400 items
        #sn = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
    
        
    
        #here create two arrays which hold indexes of the begining and end of the sorted prediction
        
   #     beg_arr = all_ind[curr_ins] [:int (0.1 * number_of_items)] 
   #     end_arr = all_ind[curr_ins] [int (0.9 * number_of_items):] 
        
        # lb = []
        # ub = []
        # var_names = []
        # for x in range(number_of_items):
        #     #temp = float(random.randint(1,100))
        #     #dec_var.append(temp)
        #     #print(x)
        #     #check if item in beg_arr
        #     if x in beg_arr:
        #         lb.append(float(b_sol[x]))
        #         ub.append(float(b_sol[x]))
            
        #     #check if item in end_arr 
        #     if x in end_arr:
        #         lb.append(float(b_sol[x]))
        #         ub.append(float(b_sol[x]))
            
    
        #     if x not in beg_arr and x not in end_arr:
        #         lb.append(0.0)
        #         ub.append(1.0)
        print("forced items to 1 ", len(sel))
        print("forced items to 0 ", len(nsel))
        
        ###temporary change for the sel and nsel to 10% and 5%
        cnt_s = 0
        cnt_ns = 0
        lb = []
        ub = []
        var_names = []
        for x in range(number_of_items):
            #temp = float(random.randint(1,100))
            #dec_var.append(temp)
            #print(x)
            #check if item in beg_arr
            if x in sel:# and cnt_s <=int(0.1*number_of_items):
                cnt_s =+ 1
#                lb.append(1)
#                ub.append(1)
                lb.append(float(rl_sol[x]))
                ub.append(float(rl_sol[x]))
            
            #check if item in end_arr 
            if x in nsel:# and cnt_ns <=int(0.05*number_of_items):
                cnt_ns = +1
#                lb.append(0)
#                ub.append(0)
                lb.append(float(rl_sol[x]))
                ub.append(float(rl_sol[x]))
            
    
            if x not in sel and x not in nsel:# or cnt_s > int(0.1*number_of_items) or cnt_ns>int(0.05*number_of_items):
                lb.append(0.0)
                ub.append(1.0)   
            nm = "x%d" % (x+1)
            var_names.append(nm)
            #knapsack_gen.write_to_file(str(temp),0)
            
            
        
        con_names = []   
        b_rhs = [] 
        conts_all_with_names = []
        fin_con = []
        #sn = ''
        for y in range(int(num_of_const)):
            #print(y)
            rhs = 3/4 * sum(const_list[y])
            #print(rhs)
            b_rhs.append(rhs)
            cn = "c%d"%(y+1)
            con_names.append(cn)
            tmp_con = []
            tmp_con.append(var_names)
            tmp_con.append(const_list[y])
            conts_all_with_names.append(tmp_con)
            #fin_con.append(conts_all_with_names)
            #sn = sn + 'G'
            ##cons.append([])
            #for z in range(1):
            #    cons.append([])
            #    cons[y].append(var_names)
            #    #cons[y].append([])
            #    con_new = knapsack_gen.con_generator()
            #    rhs = 3/4 * sum(con_new)
            #    b.append(rhs) 
            #    cons[y].append(con_new)
            #sn = sn + 'G'
        
        #print(b_rhs)
        #number of variables is important, in this case we use hardcoded 4
        cpl  = cplex.Cplex()
       
        #set max sense temporarily
        cpl.objective.set_sense(cpl.objective.sense.minimize)
        t= cpl.variables.type
        cpl.parameters.timelimit.set(7200)

        
   #     cpl.parameters.mip.tolerances.mipgap.set(float(opt_gap_m2)) 
        
        #decision variables
       
        cpl.variables.add(obj = objf,
                         lb = lb,
                         ub = ub,
                         types = [t.binary]*ndec,
                         names = var_names)
       
        cpl.linear_constraints.add(lin_expr = conts_all_with_names, senses = sn, rhs = b_rhs, names = con_names)
    
        cpl.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = var_names, val = objf)],
                                   senses=["L"],
                                   rhs = [obj_ub]
                                   )
        
        
   #     cpl.write("model.lp")
        cpl.solve()
       
       
        print(cpl.solution.get_values())
        #for it in cpl.solution.get_values():
        #    sol_val.append(int(it))
        sol_val = cpl.solution.get_values()
        print(cpl.solution.get_objective_value())
        obj = cpl.solution.get_objective_value()
        print("GAP for solve_using_cplex_reduced_rl: ",cpl.solution.MIP.get_mip_relative_gap())
        return obj,sol_val



all_ind = []
sorted_rat_for_train = []
all_ins = []
for rats in range(10):
    name_ind = "ind_"+str(number_of_items)+"_"+str(rats) +"_"+ str(dist_num)+ ".npy"
    name_ins = "ins_"+str(number_of_items)+"_"+str(rats) +"_"+ str(dist_num)+ ".npy"
    ##
    ##      DO THIS PART ONLY THE FIRST TIME TO GENERATE TEST INSTANCES
    ##      THEN LOAD THEM ONLY
#    ins,rhs,tst_train = Knp_Ins_Gen.instance_generator(number_of_items)
#    all_ins.append(ins)
##    ins_n = copy.deepcopy(Knp_Ins_Gen.Standardize(ins))
##    rhs_n = copy.deepcopy(Knp_Ins_Gen.Standardize(rhs))
#    ratio_ins_for_train = Knp_Ins_Gen.sort_based_on_ratios(ins,rhs)
#    ind = sorted(range(len(ratio_ins_for_train)), key=lambda k: ratio_ins_for_train[k],reverse=False)
#    all_ind.append(ind)
#    ratio_ins_for_train.sort()
#    sorted_rat_for_train.append(ratio_ins_for_train)
    #save files to load

#    Knp_Ins_Gen.saveList(ins, name_ins)
#    Knp_Ins_Gen.saveList(ind, name_ind)

    ####
    ###     To load saved instances uncoment these lines
    ###
    all_ind.append(Knp_Ins_Gen.loadList(name_ind))
    all_ins.append(Knp_Ins_Gen.loadList(name_ins))
    



#print("ins:",ins)
#print("rhs:",rhs)
#print("ins_n:",ins_n)
#print("rhs_n:",rhs_n)
#print("sorted_rat_fro_train:", sorted_rat_for_train)
#print("ind",ind)


""" UP TO HERE EVERYTHING IS AS IT SHOULD BE"""

###we have tu run any instance throught the knn first
class KNN_KNP():
    #self.loops = loops
    #self.datat = data
    def distcorr(X, Y):
        #Compute the distance correlation function
        #>>> a = [1,2,3,4,5]
        #>>> b = np.array([1,2,9,4,4])
        #>>> distcorr(a, b)
        #0.762676242417
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        if np.prod(X.shape) == len(X):
            X = X[:, None]
        if np.prod(Y.shape) == len(Y):
            Y = Y[:, None]
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        n = X.shape[0]
        if Y.shape[0] != X.shape[0]:
            raise ValueError('Number of samples must match')
        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
        
        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor
    
    
    def corr_distance(cent, data_points):
        distances_arr = []  # create a list to which all distance calculations could be appended.
        for centroid in cent:
            for datapoint in data_points:
                #this uses the distance correlation
            #    distances_arr.append(distcorr(centroid, datapoint))
                #this uses jaccard similarity
                distances_arr.append(KNN_KNP.jaccard(centroid, datapoint))
        return distances_arr
    
    #define Jaccard Similarity function
    def jaccard(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union
    
    #find Jaccard Similarity between the two sets 
    #jaccard(a, b)
    
    def distance_between(cent, data_points):
        '''This function calculates the euclidean distance between each data point and each centroid.
        It appends all the values to a list and returns this list.'''
    #    print("data_points",data_points)
    #    print("cent", cent)
        distances_arr = []  # create a list to which all distance calculations could be appended.
        for centroid in cent:
            for datapoint in data_points:
              sum_of_all = 0
              for i in range(int(len(centroid))):
                  sum_of_all += (datapoint[i]-centroid[i])**2
              sum_of_all += (sum(datapoint) - sum(centroid) )**2
            #sum_of_all+=(3/4 * sum(datapoint)- 3/4*sum(centroid))**2
              distances_arr.append((sqrt(sum_of_all))/len(centroid))
    #          print("Sum of ALL: ", sum_of_all)
    #          print("Len Centroid",len(centroid))
                    #here we can calculate the distance as the average distance of all the points in the constraints
              
    #    print(distances_arr)
        #how many entries we should have?
    #    print("num of entries: ", len(cent)*len(data_points), len(distances_arr))
        return distances_arr
    
    
    def format_data(arr):
        '''This function reads a csv file with pandas, prints the dataframe and returns
        the two columns in numpy ndarray for processing as well as the country names in
        numpy array needed for cluster matched results'''
        data1 = arr 
        c_names = arr.index.values
        #list_array = data1[[data1.columns[0],data1.columns[1], data1.columns[2],data1.columns[3],data1.columns[4],]].values
        list_array = data1.iloc[:,:].values
    #    print(list_array)
    #    print(country_names)
        return list_array, c_names
    
    def knn_constraint_out(num_of_iter, data_to_cluster,number_of_clusters):    
        startClustering = timeit.default_timer()
        '''This function reads a csv file with pandas, prints the dataframe and returns
        the two columns in numpy ndarray for processing as well as the country names in
        numpy array needed for cluster matched results'''
        #data1 = arr 
        x = KNN_KNP.format_data(data_to_cluster)
    
        k = int(number_of_clusters)
    
        #iterations = int(input("Please enter the number of iterations that the algorithm must run: "))
        iterations = num_of_iter
        # Set the random number of centroids based on the user input value of "k".
      #  centroids = random.sample(x_list, k)
    
    
    
        for iteration in range(0, iterations):
            # Print the iteration number
            print("ITERATION: " + str(iteration+1))
            # assign the function to a variable as it has more than one return value
            assigning = KNN_KNP.assign_to_cluster_mean_centroid(x, k)
        
            # Create the dataframe for vizualisation
        #    cluster_data = pd.DataFrame({'x': x[0][0:, :],
        #                                 'label': assigning[0],
        #                                 'const': x[1]})
            
            cluster_data = pd.DataFrame(x[0][0:, :]) # , assigning[0], x[1]})
            cluster_data['Constraint'] = x[1]
            cluster_data['Label'] = assigning[0]
            
        
            # Create the dataframe and grouping, then print out inferences
        #    group_by_cluster = cluster_data[[
        #        'Country', 'Birth Rate', 'Life Expectancy', 'label']].groupby('label')
            group_by_cls = cluster_data.groupby(['Label'])
            count_clusters = group_by_cls.count()
            # Inference 1
         #   print("CONSTRAINTS PER CLUSTER: \n" + str(count_clusters))
            # Inference 2
         #   print("LIST OF COUNTRIES PER CLUSTER: \n",list(group_by_cls))
            # Inference 3
        #    print("AVERAGES: \n", str(cluster_data.groupby(['label']).mean()))
        
           # Set the variable mean that holds the clusters dict
            mean = assigning[1]
        #    print("mean",mean)
            # create a dict that will hold the distances of between each data point in
            # a particular cluster and its mean. The loop here will create the amount of clusters based
            # on user input.
            means = {}
            for clst in range(0, k):
                means[clst+1] = []
        
            # Create a for loop to calculate the squared distances between each
            # data point and its cluster mean
            for index, data in enumerate(mean):
                array = np.array(mean[data])
                array = np.reshape(array, (len(array),number_of_items ))
                # Set two variables, one for each variable in the data set that
                # holds the calculation of the cluster mean of each variable
                avg=[]
                for j in range(len(array[0])):
                  if(len(array[0:0])!=0):
                    avg.append(sum(array[0:,j])/len(array[0:j]))
                  else:
                    avg.append(0)
        #        birth_rate = sum(array[0:, 0])/len(array[0:, 0])
        #        life_exp = sum(array[0:, 1])/len(array[0:, 1])
                # within this for loop, create another for loop that appends to the means dict
                # the squared distance of between each data point in it's cluster and the cluster mean.
        
                sum_of_all_2 = 0
                for data_point in array:
                  for i in range(len(data_point)):
                    sum_of_all_2 += sqrt((avg[i]-data_point[i])**2)
                   
        
                  if(len(data_point)!=0):
                    means[index+1].append(sum_of_all_2/len(data_point))
                  else:
                    means[index+1].append(0)
            # create a list that will hold all the sums of the means in each of the clusters.
            total_distance = []
            for ind, summed in enumerate(means):
                total_distance.append(sum(means[ind+1]))
        
        #    print("Cluster Data: ", cluster_data)
            print("Total Distance: ", total_distance )
            # print the summed distance
            print("Summed distance of all clusters: " + str(sum(total_distance)))
        #    print(list(group_by_cls)[1][1]["Constraint"])
            
           # cls0 = list(list(group_by_cls)[0][1]["Constraint"])
           # cls1 = list(list(group_by_cls)[1][1]["Constraint"])
           # cls2 = list(list(group_by_cls)[2][1]["Constraint"])
           # print(cls0)
           # print(cls1) 
           # print(cls2)
                
        endClustering = timeit.default_timer()
        
        return cluster_data,group_by_cls, total_distance, endClustering - startClustering, mean, means
        
        
        
    def assign_to_cluster_mean_centroid(x_in, n_user):
        '''This function calls the distance_between() function. It allocates from
        the returned list, each data point to the centroid/cluster that it is the
        closest to in distance. It also rewrites the centroids with the newly calculated
        means. Finally it returns the list with cluster allocations that are 
        in line with the order of the countries. It also returns the clusters dictionary.'''
        x_list = np.ndarray.tolist(x_in[0][0:,0:])
        centroids = random.sample(x_list, n_user)
        
        centroids_in = copy.deepcopy(centroids)
        distances_arr_re = np.reshape(KNN_KNP.distance_between(
            centroids_in, x_in[0]), (len(centroids_in), len(x_in[0])))
        
        #using correlation distance:
    #    distances_arr_re = np.reshape(corr_distance(
    #        centroids_in, x_in[0]), (len(centroids_in), len(x_in[0])))
            
        
    #    print(distances_arr_re)
        datapoint_cen = []
        distances_min = []  # Done if needed
        for value in zip(*distances_arr_re):
            distances_min.append(min(value))
            datapoint_cen.append(np.argmin(value)+1)
        # Create clusters dictionary and add number of clusters according to
        # user input
        clusters = {}
        for no_user in range(0, n_user):
            clusters[no_user+1] = []
        # Allocate each data point to it's closest cluster
        for d_point, cent in zip(x_in[0], datapoint_cen):
            clusters[cent].append(d_point)
    
        # Run a for loop and rewrite the centroids
        # with the newly calculated means
        for i, cluster in enumerate(clusters):
            reshaped = np.reshape(clusters[cluster], (len(clusters[cluster]), len(centroids_in[0])))
            for j in range(len(centroids_in[0])):
    #            print(reshaped[0:,j])
    #            print(reshaped[0:j])
                centroids[i][j] = sum(reshaped[0:,j])/len(reshaped[0:,j])
    #            new_cent[i][j] = sum(reshaped[0:,j])/len(reshaped[0:,j])
    #        centroids[i][0] = sum(reshaped[0:, 0])/len(reshaped[0:, 0])
    #        centroids[i][1] = sum(reshaped[0:, 1])/len(reshaped[0:, 1])
    #    print('Centroids for this iteration are:' + str(centroids))
    #    print(clusters)
        return datapoint_cen, clusters
    
    def knn_main_start(all_instances, number_of_loops,number_of_clusters,curr_ins):
        print("Current Instance KNN: ",curr_ins)        
        tmp_ins = copy.deepcopy(all_instances[curr_ins])
        del tmp_ins[0]
        const_list = copy.deepcopy(tmp_ins)
        
        df = pd.DataFrame(all_instances)
        df2 = pd.DataFrame(df.loc[curr_ins])
        #this maybe needs to change 
        arr = df2[curr_ins].apply(pd.Series)
        arr = arr.rename(columns=lambda x : 'item_'+str(x))
        
        #remove the objective weights
        arr = arr.drop(0)
        
        #list to remain constant to check for feasibility
        static_list_of_const = copy.deepcopy(arr.to_numpy().tolist())
        
        #list containing the constraints not in the new problem
        not_in_problem = copy.deepcopy(arr.to_numpy().tolist())
        #list containing the constraint of the new problem
        in_problem = []
        #data to be used for clustering
        arr
        
        res0 = KNN_KNP.knn_constraint_out(number_of_loops,arr,number_of_clusters)
        r=[] 
        for i in range (1, number_of_clusters):
         #   a = np.array(list(list(res0[1])[i][1]["Constraint"]))
            a = res0[5][i]
          #  if a not in r:
            idx_max = np.argmax(a)
            idx_min = np.argmin(a)
          #  print(res0[4][i][0])
            r.append(res0[4][i][idx_max].tolist())
            r.append(res0[4][i][idx_min].tolist())
        ###KNN MAIN LOOP
        
        
        for i in range(0, len(r)):
         #   if(const_list[i] not in in_problem):
             in_problem.append(r[i])
        
        
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
        while(feasible != True and lp < 1000):
            """
            we run the clustering once and we add 2 const for each cluster and the 
            cluster themself (try two farest from the cluster)
            
            then for each loop, we choose some contraints to add and we also 
            run the clustering again to remove some
            """
            print("LOOP Number ", lp)
            lp+=1
            
            newObjective, newSolution = Cplex_Methods.solve_using_cplex_reduced_knn(in_problem,all_ins[curr_ins][0],number_of_items,len(in_problem))
            print("objective using CPLEX reduced: ", newObjective)
            
            new_per_change = ((newObjective-obj_full)/obj_full)*100
            
            print("Percentage change is: ", new_per_change)
            
            
            cnt_l=0
            for i in range(0,len(sol_full)):
                if newSolution[i] == sol_full[i]:
                    cnt_l+=1
            print("Number of common items is : ", cnt_l)
            
            
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
          
            print(violation_index)
            print(violation_amount)
            print("Violated Constraints: ", len(violated_const))
            print("Satisfied Constraints: ", len(satisfied_const))

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
                print("The list with differences is: ")
                print(violation_amount)
                for i in range(6):
                    #each loop add the most violated constraint
                    #rhs - lhs is +, so the larger the more violated
                    max_vio = np.argmax(violation_amount)
                    idx_vio = violation_index[max_vio]
                    in_problem.append(copy.deepcopy(violated_const[idx_vio]))
                    print("added constraint with violation: ", violation_amount[max_vio])
                    del violated_const[idx_vio]
                    del violation_amount[max_vio]
               #     del violation_index[max_vio]
                  #  del static_list_of_const[violated_const[idx_vio]]
            else:
                for x in range(len(violated_const)):
                    in_problem.append(copy.deepcopy(violated_const[x]))
                    print("Adding All Const : ",violated_const[x] )
        
        
        endRecursiveSolving = timeit.default_timer()
        timeSolving = endRecursiveSolving - startRecursiveSolving
        
        
        print("Best objective is: ", best_obj_knn)
        print("Best solution is: ", best_sol_knn)
        print("LAST objective is: ", last_obj_knn)
        print("LAST solution is: ", last_sol_knn)
        
        return last_obj_knn, last_sol_knn, timeSolving

        
""" UP TO HERE EVERYTHING IS OK""" 




class Env_Knp(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        global current_instance_number 
        global number_of_items
        global sol_knn
        global obj_knn
        print("current instance number: ",current_instance_number)
        self.sol_knn = sol_knn
        self.obj_knn = obj_knn
        self.current_instance_number = current_instance_number
        print("self current instance number: ",self.current_instance_number)
        self.item_num = int(math.sqrt(number_of_items))
        self.avg_rew_ep = 0
        self.cnt=0
        p1 = np.random.randint(0.6*self.item_num,self.item_num-1)
        p2 = np.random.randint(0.6*self.item_num,self.item_num-1)
       #static start each time
        #p1 = int(0.5*number_of_items)
        #p2 = int(0.5*number_of_items)
       #st_point_1 = int(0.7*number_of_items)
        self.agent_pos = (p1,p2)
        self.curr_feas = True
        print("STARTING...")
        self.cons_size = 30
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.cons_size,self.cons_size), dtype=np.int16)
        #self.reward_range(-200,200)
        self.current_episode = 0 
        self.success_episode = []
        self.best_sol = []
        self.best_obj = sys.maxsize
        self.count_state_s = 0
        self.count_state_e = 0
        self.reward_ep = 0
        self.best_obj_to_cmp = sys.maxsize
        self.best_obj_of_episode = 0
        self.iteration_num = 0
        #print("OBSERVATION",self.observation_space)
    
    def populate_world(self):
        res = np.full((self.cons_size,self.cons_size),-1)
        knn_rev = self.normal_to_crazy(self.sol_knn,all_ind[self.current_instance_number])
   #     print("Knn Solution: ")
   #     print(best_sol_knn)
   #     print("Knn reverted sol:")
   #     print(knn_rev)
        s = int(0.8*math.sqrt(number_of_items))
        size = int(math.sqrt(number_of_items))
        knn_2d = np.reshape(knn_rev, (size,size))
        for i in range(size):
            for j in range(size):
                res[i][j] = knn_2d[i][j]
             #   res[i][j] = 1

#        for i in range(s,size):
#            for j in range(s,size):
            #   res[i][j] = knn_2d[i][j]
#                res[i][j] = 0


        # for i in range(4,6):
        #     for j in range(4,6):
        #         res[i][j] = 1
        
        return res
    
    ###CURRENTLY NOT USED
    def populate_world_again(self,wrd):
        #res = np.full((self.cons_size,self.cons_size),-1)
        #print(wrd)
        knn_rev = self.get_reverted_knn_solution(self.sol_knn)
        #print("Knn Solution: ")
        #print(sol_knn)
        #print("Knn reverted sol:")
        #print(knn_rev)
        size = int(math.sqrt(number_of_items))
        knn_2d = np.reshape(knn_rev, (size,size))
        for i in range(0,int(1*size)):
            for j in range(0,int(1*size)):
                #print(i,j)
#                wrd[i][j] = 1
                wrd[i][j] = knn_2d[i][j]

 #       for i in range(side-1,side):
 #           for j in range(side-1,side):
 #               res[i][j] = 0
        
        return wrd

        
    def get_best_sol(self):
        
        return self.best_sol
    
    def get_best_obj(self):
        
        return self.best_obj
    
    def rand_loc(self):
        r1 = np.random.randint(0.3*self.item_num,self.item_num-1)
        r2 = np.random.randint(0.3*self.item_num,self.item_num-1)
        pos  = (r1,r2)
        return pos
        
    
    # def continue_on_the_same_env(self,player):
    #     if self.current_player == 1:
    #         return self.world_p1
    #     else
    #     return
        
    def reset(self):
        #sometimes in a reset, use tunneling to move to another place on the search space, for ex: when state is S for many times or E for many times
        global first_time
        print("RESETTING...")
        self.state='P'
        self.current_step = 0
        self.max_step = 550
        self.best_obj_of_episode = 0
        self.avg_rew_ep = 0
        self.cnt=0
        r1 = np.random.randint(0.7*self.item_num,self.item_num-1)
        r2 = np.random.randint(0.7*self.item_num,self.item_num-1)
        self.curr_feas = True
        #random start each time
     #   r1 = np.random.randint(self.item_num-5,self.item_num-1)
     #   r2 = np.random.randint(self.item_num-5,self.item_num-1)
        #static start each time
        #r1 = int(0.5*number_of_items)
        #r2 = int(0.5*number_of_items)
        self.agent_pos = (r1,r2)
    #    st_point = int(0.7*number_of_items)
   #     self.agent_pos = (st_point, st_point)
        self.world = self.populate_world()
        self.count_state_s = 0
        self.count_state_e = 0
       # self.world = np.ones((self.cons_size,self.cons_size))
        # if first_time == True:
        #     #self.world_p1 = self.original_best_env
        #     #self.world_p2 = self.original_best_env
        #     self.world_p1 = self.populate_world()
        #     self.world_p2 = self.populate_world()
        # else:
        #     self.world_p1 = self.populate_world_again(self.original_best_env)
        #     self.world_p2 = self.populate_world_again(self.original_best_env)
        return self._next_observation()
        
    def _next_observation(self):
        obs = self.world
        #print("OBSERVATION RESET",obs)
        return obs
    
    """    1. lets define only three states
            W :  1
            L : -1
            S : -10

        2. then the conditions become:
            if the problem is feasible and objective is less or = kmeans        - W  - win (Pos)
            if the problem is feasible and objective is larger than kmeans      - G  - go on(pos)
            if the problem is infeasible or objective is larger than kmeans     - L  - loose(neg)
            if we're still moving through existing selections                   - S  - go on(neg)
            otherwise                                                           - O  - Out ( neg)
"""
    def get_state(self, wrd, pre,post):
      #  if self.feas1D(wrd,all_ind[self.current_instance_number],all_ins[self.current_instance_number]) == False:
        if self.curr_feas == False:
            tmp_state = 'L'  #loose, end the search  -100
        if  self.obj_knn > pre:
        #    if self.feas1D(wrd,all_ind[self.current_instance_number],all_ins[self.current_instance_number]) == True:
            if self.curr_feas == True:
                tmp_state = 'W'  #win, end the search +500
        #        else: 
        #            tmp_state = 'B'  #go on, but negative result -3
        if  self.obj_knn <= pre:
          #  if self.feas1D(wrd,all_ind[self.current_instance_number],all_ins[self.current_instance_number]) == True:
            if self.curr_feas == True:
                tmp_state = 'G' #go on, positive result: +10
        #        else: 
        #            tmp_state = 'B' #go on, negative result - 5
        if pre == post:
            tmp_state = 'S' # go on, but negative result:  - 1
        if post < self.obj_knn and self.curr_feas == True: #self.feas1D(wrd,all_ind[self.current_instance_number],all_ins[self.current_instance_number]) == True:
            tmp_state = 'K'
        return tmp_state
    
    def make_move(self, wrd, nextpos):
        if wrd[nextpos] != -1:
            #  world_temp[next_pos] = 0
            if wrd[nextpos] == 1:
                wrd[nextpos] = 0
          #  else:
           #     wrd[nextpos] = 1
        #not allowing adding items
         #   if wrd[nextpos] == 1:
         #      wrd[nextpos] = 0 
        return wrd
            
    def check_legitimacy(self, act):
        next_pos = None
        curr_pos = self.agent_pos
        if act == 0 and curr_pos[0] - 1 >= 0:
            next_pos = (curr_pos[0]-1,curr_pos[1])
        elif act == 1 and curr_pos[1] + 1 <= self.cons_size-1:
            next_pos = (curr_pos[0],curr_pos[1]+1)
        elif act == 2 and curr_pos[0] + 1 <= self.cons_size -1:
            next_pos = (curr_pos[0]+1,curr_pos[1])
        elif act == 3 and curr_pos[1] - 1 >= 0:
            next_pos = (curr_pos[0],curr_pos[1]-1)
        return next_pos
    
    def _take_action(self,action):
        global best_obj_from_rl
        global best_sol_from_rl
        global best_sol_from_rl_inverted 
        post_objective = None
        next_pos = None
    #    current_pos = self.agent_pos
    #    cons_curr_pos = self.agent_pos
        world_temp = copy.deepcopy(self.world)
    #    pre_objective = self.getObjective1D(world_temp,ind_rat_for_train[0], all_ins_for_train[0][0]) #wrd,ind,cost_val
        pre_objective = self.getObjective1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number][0]) #wrd,ind,cost_val
        tmp_world_cons = copy.deepcopy(world_temp)
        print("current pos of agent: ", self.agent_pos)
        print("current action of agent: ", action)
       #then mark that position with -1 or -2, but count the value as 0 (not selected)
        #next_pos = current_pos
        next_pos = self.check_legitimacy(action)

        if next_pos != None: 
            world_temp = self.make_move(world_temp,next_pos)
            self.curr_feas = self.feas1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number])
            post_objective = self.getObjective1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number][0])
            self.state = self.get_state(world_temp, pre_objective,post_objective)
            if world_temp[next_pos] == -1:
                self.state = 'E'
                self.count_state_e += 1
            else:
                self.world = world_temp
                self.agent_pos = next_pos
                if post_objective <= pre_objective and self.curr_feas ==True:
                    self.best_obj_of_episode = post_objective
                    bs = self.TwoD_to_OneD(world_temp,self.item_num)
                    best_sol_from_rl_inverted = copy.deepcopy(bs)
                    self.best_sol = self.crazy_to_normal(bs, all_ind[self.current_instance_number])
                    self.best_obj = post_objective
                    best_sol_from_rl = copy.deepcopy(self.best_sol)
                    best_obj_from_rl = copy.deepcopy(self.best_obj)
        else:    
        #if position is none then it is outside of canvas
            self.state = 'O'
            self.count_state_s +=1
    
   #  def _take_action(self,action):
   #      global best_obj_from_rl
   #      global best_sol_from_rl
   #      global best_sol_from_rl_inverted
   #      post_objective = None
   #      next_pos = None
   #      #world_temp = np.ones((self.cons_size,self.cons_size))
   #      current_pos = self.agent_pos
   #      cons_curr_pos = self.agent_pos
   #      world_temp = self.world
   #      pre_objective = self.getObjective1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number][0]) #wrd,ind,cost_val
   #      tmp_world_cons = copy.deepcopy(world_temp)
   # #     print("current pos of agent: ", current_pos)
   # #     print("current action of agent: ", action)
                
   #      if action == 0:
   #       #   print("action: ", action)
   # #         print("NEXT POS: ",current_pos[0]-1,current_pos[1])
   #          if current_pos[0] - 1 >= 0:# and world_temp[current_pos[0]-1,current_pos[1]] != 0:
   #              next_pos = (current_pos[0]-1,current_pos[1])
   #              world_temp = self.make_move(world_temp,next_pos)
   #              current_pos = next_pos
   #              #if the problem becomes infeaseable the player looses
   #              post_objective = self.getObjective1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number][0])
   #              print("pre obj: ", pre_objective, " post obje: ",post_objective)
   #              self.state = self.get_state(world_temp, pre_objective,post_objective)

   #          else:
   #              #state when choosing actions that take the agent out of the borders
   #              self.state = 'O' #go on, but negative result
   #              self.count_state_s += 1
            
   #          if tmp_world_cons[cons_curr_pos] == -1 and next_pos != None: # and  (world_temp[next_pos] != 1 or world_temp[next_pos] != 0):
   #              #state when going where there is not item
   #              self.state = 'E'
   #              self.count_state_e += 1
                
            
   #      elif action == 1:
   #  #        print("NEXT POS: ",current_pos[0],current_pos[1]+1)
   #          if current_pos[1] + 1 <= self.cons_size-1:# and world_temp[current_pos[0],current_pos[1]+1] != 0:
   #              next_pos = (current_pos[0],current_pos[1]+1)
   #              world_temp = self.make_move(world_temp,next_pos)
   #              current_pos = next_pos
   #              post_objective = self.getObjective1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number][0])
   #              print("pre obj: ", pre_objective, " post obje: ",post_objective)
   #              self.state = self.get_state(world_temp, pre_objective,post_objective)

   #          else:
   #              self.state = 'O'
   #              self.count_state_s += 1

            
   #          if next_pos != None and tmp_world_cons[cons_curr_pos] == -1 :#(world_temp[next_pos] != 1 or world_temp[next_pos] != 0):
   #              self.state = 'E'
   #              self.count_state_e += 1
                
   #      elif action == 2:
   #  #        print("NEXT POS: ",current_pos[0]+1,current_pos[1])
   #          if current_pos[0] + 1 <= self.cons_size -1 :# and world_temp[current_pos[0]+1,current_pos[1]] != 0:
   #              next_pos = (current_pos[0]+1,current_pos[1])
   #              world_temp = self.make_move(world_temp,next_pos)
   #              current_pos = next_pos
   #              post_objective = self.getObjective1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number][0])
   #              print("pre obj: ", pre_objective, " post obje: ",post_objective)
   #              self.state = self.get_state(world_temp, pre_objective,post_objective)

   #          else:
   #              self.state = 'O'
   #              self.count_state_s += 1
            
   #          if next_pos != None and tmp_world_cons[cons_curr_pos] == -1   :# (world_temp[next_pos] != 1 or world_temp[next_pos] != 0):
   #              self.state = 'E'
   #              self.count_state_e +=1

   #      elif action == 3:
   #  #        print("NEXT POS: ",current_pos[0],current_pos[1]-1)
   #          if current_pos[1] - 1 >= 0:# and world_temp[current_pos[0],current_pos[1]-1] != 0:
   #              next_pos = (current_pos[0],current_pos[1]-1)
   #              world_temp = self.make_move(world_temp,next_pos)
   #              current_pos = next_pos
   #              #if the problem becomes infeaseable the player looses
   #              post_objective = self.getObjective1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number][0])
   #              print("pre obj: ", pre_objective, " post obje: ",post_objective)
   #              self.state = self.get_state(world_temp, pre_objective,post_objective)
   #          else:
   #              self.state = 'O'
   #              self.count_state_s += 1
            
   #          if next_pos != None and  tmp_world_cons[cons_curr_pos] == -1 :# (world_temp[next_pos] != 1 or world_temp[next_pos] != 0):
   #              self.state = 'E'
   #              self.count_state_e +=1
        
   #      if next_pos != None:
   #          if pre_objective < post_objective:
   #              self.best_obj_of_episode  = pre_objective
   #          else:
   #              if self.feas1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number])==True:
   #                  self.best_obj_of_episode = post_objective
   #                  self.world = world_temp
   #                  self.agent_pos = current_pos
   #                  bs = self.TwoD_to_OneD(world_temp,self.item_num)
   #                  best_sol_from_rl_inverted = copy.deepcopy(bs)
   #                  self.best_sol = self.crazy_to_normal(bs,all_ind[self.current_instance_number])
   #                  self.best_obj = post_objective
   #                  best_sol_from_rl = copy.deepcopy(self.best_sol)
   #                  best_obj_from_rl = copy.deepcopy(self.best_obj)
#         if next_pos != None:
#             self.best_obj_of_episode = post_objective
#             self.world = world_temp
#             self.agent_pos = current_pos
# #        if self.feas1D(world_temp,ind_rat_for_train[self.ins_num],all_ins_for_train[self.ins_num])==True and post_objective != None and post_objective < self.best_obj:#if the problem is still feasible
#         if self.feas1D(world_temp,all_ind[self.current_instance_number],all_ins[self.current_instance_number])==True and post_objective != None and post_objective <= self.obj_knn+10:  #if the problem is still feasible
#             bs = self.TwoD_to_OneD(world_temp,self.item_num)
#             best_sol_from_rl_inverted = copy.deepcopy(bs)
#             self.best_sol = self.crazy_to_normal(bs, all_ind[self.current_instance_number])
#             self.best_obj = post_objective
#         #    best_sol_from_rl = self.get_reverted_selection(bs, ind_rat_for_train[self.ins_num])
#             best_sol_from_rl = copy.deepcopy(self.best_sol)
#             best_obj_from_rl = copy.deepcopy(self.best_obj)
#    #         print("new best obj ", self.best_obj,  " best obj ", best_obj_from_rl)
#    #         print("new best sol ", self.best_sol)
#    #         print("new best sol ", best_sol_from_rl)
#             tst_obj  = 0
#             for i in range(0,len(self.best_sol)):
#                 tst_obj += all_ins[self.current_instance_number][0][i]*self.best_sol[i]
                
#             tst_obj_2  = 0
#             for i in range(0,len(best_sol_from_rl)):
#                 tst_obj_2 += all_ins[self.current_instance_number][0][i]*best_sol_from_rl[i]
#         print("best objective inside RL: ", self.best_sol)
#         best_sol_from_rl = self.best_sol
   #         print("Comparison: ", self.best_obj, " = ", tst_obj, " = ",tst_obj_2  )
   #         print("Ins NUM: ",self.ins_num)
            
   #     print("the state is : ",self.state)
   #     print("Counter for S: ", self.count_state_s)

     
    def step(self,action):
        print("ITERATION: ", self.iteration_num)
        self.iteration_num+=1
        self._take_action(action)
        self.current_step +=1       
  #      print(self.world_p1)
        if self.state == 'B':
            #here add the conditions below as well
       #     print(f'Player {self.current_player} bad move')
            reward = -3
    #        reward = 500
            done = False
            print("State: ", self.state)
        elif self.state == 'G':
            #here add the conditions below as well
           # print(f'Player {self.current_player} good move')
            reward = +5
         #   reward = 500
            done = False
            print("State: ", self.state)
        elif self.state == 'L':
        #    print(f'Player {self.current_player} lost')
            reward = -100
        #    reward = -500
            done = True
            print("State: ", self.state)
     #       done = True
        elif self.state == 'W':
            reward = +500
        #    reward = +5000
     #       done = True
            done = True
            print("State: ", self.state)
        elif self.state == 'K':
            reward = +500
            done = True
        elif self.state == 'O':
        #    reward = -5000  
            reward = -150
            print("State: ", self.state)
            #done = False
            if self.count_state_s >= 3:
                #use tunneling, do not end the loop.
            #    done = True
                done = True
                #self.agent_pos=(int(0.4*number_of_items),int(0.4*number_of_items))
                self.count_state_s = 0  
            else:
                done = False
        elif self.state == 'E':
            reward = -150
            print("State: ", self.state)
            if self.count_state_e >= 3:
                #use tunneling, do not end the loop.
            #    done = True
                done = True
            #    self.agent_pos=(int(0.4*number_of_items),int(0.4*number_of_items))
                self.count_state_e = 0
            else:
                done = False
        #    done = False
        
        elif self.state == 'S':
            reward = -1
        #    reward = -500
            done = False
            print("State: ", self.state)
        # if self.current_step >= self.max_step:
        #     if self.feas1D(self.world,ind_rat_for_train[0],all_ins_for_train[0]) == False:
        #         done = True
        #     else:
        #         self.current_step = 0
        elif self.feas1D(self.world,all_ind[self.current_instance_number],all_ins[self.current_instance_number]) == False:
            if self.best_obj_of_episode <= self.best_obj_to_cmp:
 #               reward = +5000
                self.best_obj_to_cmp = self.best_obj_of_episode
 #           else:
 #               reward = -5000
            
        self.cnt+=1
        self.avg_rew_ep = (self.avg_rew_ep + reward)/self.cnt
        print("Average Reward: ", self.avg_rew_ep)
        if done or self.feas1D(self.world,all_ind[self.current_instance_number],all_ins[self.current_instance_number]) == False:
            self.render_episode(self.state,self.world,self.avg_rew_ep)
   #         print("Feasibility in Step Function:", self.feas1D(self.world,ind_rat_for_train[0],all_ins_for_train[0]))
            self.current_episode += 1
            done = True
  
#       if done:
#            self.current_episode += 1
#            self.render_episode(self.state)
#            print("Feasibility in Step Function:", self.feas1D(self.world_p1,ind_rat_for_train[0],all_ins_for_train[self.ins_num]), self.feas1D(self.world_p2,ind_rat_for_train[0],all_ins_for_train[self.ins_num]) )
            
        obs = self._next_observation()
       # self.reward_ep = reward

        return obs, self.avg_rew_ep, done, {}           

    
    def TwoD_to_OneD(self,wrd,items):
        res = []
        for i in range(0,items):
            for j in range(0,items):
                res.append(wrd[i][j])
        return res
    
    def crazy_to_normal(self,state, index_arr):
        reverted_form = [0]*len(index_arr)
        for i in range(0,len(index_arr)):
           # reverted_form[i] = state[index_arr[i]]
            reverted_form[index_arr[i]] = state[i]
        return reverted_form

    # def get_reverted_knn_solution(self,knn_sol, index_arr):
    #     reverted_form = [0]*len(index_arr)
    #     for i in range(0,len(index_arr)):
    #         #reverted_form[i] = state[index_arr[i]]
    #         reverted_form[index_arr[i]] = knn_sol[i]
    #            #res[ind[lp]] = testing[lp]
    #     return reverted_form
    
    def normal_to_crazy(self,knn_sol, index_arr):
        reverted_form = [0]*len(index_arr)    
     #   print("KNN SOL:", self.sol_knn)
        for i in range(0,len(index_arr)):
            reverted_form[i] = knn_sol[index_arr[i]]
        return reverted_form

    def getObjective1D(self,wrd,ind,cost_val):
        #state = convert2Dto1D(wrd)
        state = self.TwoD_to_OneD(wrd,self.item_num)
        normal_form = self.crazy_to_normal(state, ind)
        return sum(x*y for x,y in list(zip(normal_form,cost_val)))    
    
    
    def feas1D(self,wrd,ind,mat):
     #   state = convert2Dto1D(wrd)
        state = self.TwoD_to_OneD(wrd,self.item_num)
        rev_state = self.crazy_to_normal(state, ind)
        flag = True
        tmp_state = [1]*len(mat[0])
        # for i in range(1, len(mat)):
        #     rhs.append(0.75*sum([x*y for x,y in zip(tmp_state,mat[i])]))
        #     lhs.append(sum([x*y for x,y in zip(rev_state,mat[i])]))
        
        # for j in range(0,len(rhs)):
        #     if rhs[j] > lhs[j]:
        #         flag = False
                
        for i in range(1,len(mat)):
            if 0.75*sum([x*y for x,y in zip(tmp_state,mat[i])]) > sum([x*y for x,y in zip(rev_state,mat[i])]):
                flag = False
                #     print("Feasibility ", flag)
        
        return flag
    


    def render_episode(self, win_or_lose,wrd,rew):
        global first_time
      #  global best_sol_from_rl_inverted
      #  best_sol_from_rl_inverted = self.TwoD_to_OneD(wrd,self.item_num)
        first_time = False
    #    print("Current Solution: ")
    #    self.showRoute(self.world)
        print("State: ", win_or_lose)
        print("EPISODE FINISHED!!!!!!")
        self.success_episode.append(
        'Success' if win_or_lose == 'W' else 'Failure')
        file = open('res/render_knp.txt', 'a')
        file.write(' — — — — — — — — — — — — — — — — — — — — — -\n')
        file.write('Executing File = {fn}\n')
        file.write(f'Episode number {self.current_episode}\n')
        file.write(f'{self.success_episode[-1]} in {self.current_step} steps\n')
       #file.write(str(wrd))
        file.write(f'Reward is {rew}')
        file.close()
        print("Reward: ", rew)
      #  print("Solution feasibility: ",self.feas1D(wrd,all_ind[self.current_instance_number],all_ins[self.current_instance_number]) == False)
      #  print("Objective value: ", self.getObjective1D(wrd,all_ind[self.current_instance_number], all_ins[self.current_instance_number][0]))        

    def showRoute(self,states):
    # add cliff marked as -1
        #int ROWS = 30
        #int COLS = 30
        for i in range(0, 30):
            print('---------------------------------------------------------------------------------------------------------------------------------------------------')
            out = '| '
            for j in range(0, 30):
                if states[i,j] == 1:
                    token = 'S'
                if states[i,j] == 0:
                    token = 'N'
                if states[i, j] == -1:
                    token = '*'
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    token = 'X'
                out += token + ' | '
            print(out)
        print('---------------------------------------------------------------------------------------------------------------------------------------------------') 

    def close(self):
        pass


#curr_best_gap_rl = -1
#curr_best_rl_sol = []
#curr_best_obj_rl = -2

def sol_eval(curr_best_gap_rl, new_gap, new_sol,newObj):
#    global curr_best_gap_rl
#    global curr_best_rl_sol
    print("SOL_EVAL SOL: ",newObj)
    print("SOL_EVAL OBJ: ",new_sol)
    tmp_obj = -1
    tmp=[]
    if curr_best_gap_rl == -1:
        curr_best_gap_rl = new_gap
        tmp = new_sol
        tmp_obj = newObj
    else:
        if new_gap < curr_best_gap_rl:
            tmp = new_sol
            curr_best_gap_rl = new_gap
            tmp_obj = newObj
    

    print("SOL_EVAL OBJ 2: ",tmp_obj)
    print("SOL_EVAL SOL 2: ",tmp)
    return curr_best_gap_rl, tmp, tmp_obj
    #compare two rl solutions and choose the one with the lowest gap



#####HERE GOES THE MAIN LOOP OF THE CODE

knn_all_obj = []
cplex_all_obj = []
rl_all_obj = []
rl_only_all_obj = []

knn_all_time = []
cplex_all_time = []
rl_all_time = []
rl_only_all_time = []

knn_all_gaps = []
rl_all_gaps = []
rl_only_all_gaps = []
cplex_all_gaps = []

knn_all_correct_items = []
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
p1_85_all_obj =[]
p1_85_all_time =[]
default_prediction_perc_p1 = []


p1_95_all_gaps = []
p1_95_all_correct_items = []
p1_95_all_obj =[]
p1_95_all_time =[]
default_prediction_perc_p2 = []




#for each instance
for instance in range(3):
  #  global curr_best_gap_rl
  #  global curr_best_rl_sol
    
    curr_best_gap_rl = -1
    curr_best_rl_sol = []
    curr_best_obj_rl = -1
    cnt = 0
    
    #global best_sol_from_rl
    best_sol_from_rl.clear()
    print("Looop Nr: ", instance)
#    global sol_knn
    current_instance_number = instance
    ins_t = copy.deepcopy(all_ins[current_instance_number])
    del ins_t[0]
    const_list = copy.deepcopy(ins_t)
    
    start_cplex_timer = timeit.default_timer()
    tmp_full_cons = copy.deepcopy(all_ins[current_instance_number])
    del tmp_full_cons[0]


    name_obj = "cpx_obj_"+str(number_of_items)+"_"+str(instance) +"_"+ str(dist_num) + ".npy"
    name_sol = "cps_sol_"+str(number_of_items)+"_"+str(instance) +"_"+ str(dist_num) + ".npy"
    name_gap = "cps_gap_"+str(number_of_items)+"_"+str(instance) +"_"+ str(dist_num) + ".npy"

#    obj_full, sol_full,cpx_gap = Cplex_Methods.solve_using_cplex(tmp_full_cons,all_ins[current_instance_number][0],number_of_items,num_of_const)
    
 ## uncomment these to use fixed saved solutions for cplex
    obj_full = Knp_Ins_Gen.loadList(name_obj)
    sol_full = Knp_Ins_Gen.loadList(name_sol)
    cpx_gap = Knp_Ins_Gen.loadList(name_gap)
   


    end_cplex_timer = timeit.default_timer()
    
    
    
#    Knp_Ins_Gen.saveList(obj_full, name_obj)
#    Knp_Ins_Gen.saveList(sol_full, name_sol)
#    Knp_Ins_Gen.saveList(cpx_gap, name_gap)

    
    
    print("Full CPLEX Obj: ",obj_full )
    cplex_all_gaps.append(cpx_gap)
    cplex_all_obj.append(obj_full)
    cplex_all_time.append(end_cplex_timer-start_cplex_timer)
    
    
    #solve a relaxation instead of knn
    # start_cplex_rel_timer = timeit.default_timer()
    # ins_rel = copy.deepcopy(all_ins[current_instance_number])
    # obj_rel = copy.deepcopy(ins_rel[0])
    # del ins_rel[0]
    # print("Solve the relaxed problem: ")
    # relaxed_obj,relaxed_sol = Cplex_Methods.solve_using_cplex_relaxed(ins_rel, obj_rel, number_of_items, num_of_const)
    # end_cplex_rel_timer = timeit.default_timer()
    # print(end_cplex_rel_timer-start_cplex_rel_timer)
    
    # rel_change = ((relaxed_obj-obj_full)/relaxed_obj)*100
    # print("Percentage change of relaxed problem is: ", rel_change)
    
    # rel_change_2 = ((obj_full-relaxed_obj)/obj_full)*100
    # print("Percentage change 2 of relaxed problem is: ", rel_change_2)
    
    #run knn algorithm
    
    
    obj_knn, sol_knn, time_knn = KNN_KNP.knn_main_start(all_ins, number_of_loops,number_of_clusters,current_instance_number)

    cnt_knn=0
    for i in range(0,len(sol_full)):
        if sol_knn[i] == sol_full[i]:
            cnt_knn+=1
    print("Number of common items for KNN is : ", cnt_knn)
    knn_all_correct_items.append(copy.deepcopy(cnt_knn))

    knn_all_obj.append(obj_knn)
    knn_all_time.append(time_knn)
    knn_gap = ((obj_knn-obj_full)/obj_full)*100
    knn_all_gaps.append(knn_gap)
    # Load the saved statistics
    env = DummyVecEnv([lambda: Env_Knp()])
    env = VecNormalize.load(stats_path, env)
    # Test the trained agent
    obs = env.reset()
    n_steps = 100
    print("Second Run: ")
    
    start_rl_timer = timeit.default_timer()
    #for step in range(n_steps):
    #    Here do a separate loop for rl only
    while(cnt < rl_loops):
        cnt+=1
        print("Rl step: ", cnt)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done and len(best_sol_from_rl)!=0:
            print("Goal reached!", "reward=", reward)
         #   print("RL SOL:", best_sol_from_rl)
            print("Best obj from rl ", best_obj_from_rl)
            rl_only_change = ((best_obj_from_rl-obj_full)/obj_full)*100
            print("Percentage change is: ", rl_only_change)
     #       curr_best_gap_rl, curr_best_rl_sol, curr_best_obj_rl = sol_eval(curr_best_gap_rl, rl_only_change, best_sol_from_rl,best_obj_from_rl)    
            if curr_best_gap_rl == -1:
                curr_best_gap_rl = rl_only_change
                curr_best_rl_sol = best_sol_from_rl
                curr_best_obj_rl = best_obj_from_rl
            else:
                if rl_only_change < curr_best_gap_rl:
                    curr_best_gap_rl = rl_only_change
                    curr_best_rl_sol = best_sol_from_rl
                    curr_best_obj_rl = best_obj_from_rl
       #     obs = env.reset()
    
    cnt_rl=0
    for i in range(0,len(curr_best_rl_sol)):
        if curr_best_rl_sol[i] == sol_full[i]:
            cnt_rl+=1
    print("Number of common items is : ", cnt_rl)
    rl_only_all_correct_items.append(copy.deepcopy(cnt_rl))
    rl_only_all_gaps.append(copy.deepcopy(curr_best_gap_rl))
    rl_only_all_obj.append(copy.deepcopy(curr_best_obj_rl))
            #here we break out of the loop because we only want one solution
            #break;
     #       cnt = rl_loops
    end_rl_timer = timeit.default_timer()
    timeSolving = end_rl_timer - start_rl_timer
    rl_only_all_time.append(timeSolving)

    
    #Here we define default predictions 
    sel = []
    nsel = []
    cnt_allowance_1 = 2
    cnt_allowance_0 = 2
    #here we generate two lists to hold the partial selected items
    print("inverted sol: ", best_sol_from_rl_inverted)
    for i in range(0,len(best_sol_from_rl_inverted)):
        if best_sol_from_rl_inverted[i] == 1 and cnt_allowance_1 <= 2:
         #   print("i ", i)
         #   print("val ", all_ind[current_instance_number][i])
            sel.append(all_ind[current_instance_number][i])
        elif cnt_allowance_1 < 2: 
            cnt_allowance_1+=1
        else:
            break
    
    for j in range(len(best_sol_from_rl_inverted)-1,-1,-1):
        if best_sol_from_rl_inverted[j] == 0 and cnt_allowance_0 <= 2:
          #  print("j ", j)
          #  print("val ", all_ind[current_instance_number][j])
            nsel.append(all_ind[current_instance_number][j])
        elif cnt_allowance_0 < 2: 
            cnt_allowance_0+=1
        else:
            break
    
    def_prd_perc = (len(nsel)+len(sel))/number_of_items
    
    default_prediction_perc.append(def_prd_perc)

    
    start_rl_timer_2 = timeit.default_timer()
    
    if(len(best_sol_from_rl)!=0):
        rl_obj, rl_sol = Cplex_Methods.solve_using_cplex_reduced_rl(const_list,all_ins[current_instance_number][0],number_of_items,num_of_const,best_sol_from_rl,current_instance_number,sel,nsel,curr_best_obj_rl,curr_best_rl_sol)
        print("objective using RL: ", rl_obj)
        per_change_2 = ((rl_obj-obj_full)/obj_full)*100
        print("Percentage change is: ", per_change_2)
        cnt_2=0
        for i in range(0,len(rl_sol)):
            if rl_sol[i] == sol_full[i]:
                cnt_2+=1
        print("Number of common items is : ", cnt_2)
        rl_all_correct_items.append(copy.deepcopy(cnt_2))
        rl_all_obj.append(copy.deepcopy(rl_obj))
        rl_all_gaps.append(copy.deepcopy(per_change_2))
        end_rl_timer_2 = timeit.default_timer()
        timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
        rl_all_time.append(copy.deepcopy(timeSolving_2))
    else:
        rl_all_obj.append(0)
        rl_all_gaps.append(0)
        end_rl_timer_2 = timeit.default_timer()
        timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
        rl_all_time.append(copy.deepcopy(timeSolving_2))
       # break


    #Here we define default prediction with 1 allowance each side 
    sel = []
    nsel = []
    cnt_allowance_1 = 1
    cnt_allowance_0 = 1
    #here we generate two lists to hold the partial selected items
    print("inverted sol: ", best_sol_from_rl_inverted)
    for i in range(0,len(best_sol_from_rl_inverted)):
        if best_sol_from_rl_inverted[i] == 1 and cnt_allowance_1 <= 2:
         #   print("i ", i)
         #   print("val ", all_ind[current_instance_number][i])
            sel.append(all_ind[current_instance_number][i])
        elif cnt_allowance_1 < 2: 
            cnt_allowance_1+=1
            sel.append(all_ind[current_instance_number][i])
        else:
            break
    
    for j in range(len(best_sol_from_rl_inverted)-1,-1,-1):
        if best_sol_from_rl_inverted[j] == 0 and cnt_allowance_0 <= 2:
          #  print("j ", j)
          #  print("val ", all_ind[current_instance_number][j])
            nsel.append(all_ind[current_instance_number][j])
        elif cnt_allowance_0 < 2: 
            cnt_allowance_0+=1
            nsel.append(all_ind[current_instance_number][j])
        else:
            break
    
    def_prd_perc_a1 = (len(nsel)+len(sel))/number_of_items
    
    default_prediction_perc_a1.append(def_prd_perc_a1)

    
    start_rl_timer_2 = timeit.default_timer()
    
    if(len(best_sol_from_rl)!=0):
        rl_obj, rl_sol = Cplex_Methods.solve_using_cplex_reduced_rl(const_list,all_ins[current_instance_number][0],number_of_items,num_of_const,best_sol_from_rl,current_instance_number,sel,nsel,curr_best_obj_rl,curr_best_rl_sol)
        print("objective using RL: ", rl_obj)
        per_change_2 = ((rl_obj-obj_full)/obj_full)*100
        print("Percentage change is: ", per_change_2)
        cnt_2=0
        for i in range(0,len(rl_sol)):
            if rl_sol[i] == sol_full[i]:
                cnt_2+=1
        print("Number of common items is : ", cnt_2)
        rl_all_correct_items_a1.append(copy.deepcopy(cnt_2))
        rl_all_obj_a1.append(copy.deepcopy(rl_obj))
        rl_all_gaps_a1.append(copy.deepcopy(per_change_2))
        end_rl_timer_2 = timeit.default_timer()
        timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
        rl_all_time_a1.append(copy.deepcopy(timeSolving_2))
    else:
        rl_all_obj_a1.append(-1)
        rl_all_gaps_a1.append(-1)
        end_rl_timer_2 = timeit.default_timer()
        timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
        rl_all_time_a1.append(copy.deepcopy(timeSolving_2))
       # break

    #Here we define default prediction with 2 allowences each side 
    # sel = []
    # nsel = []
    # cnt_allowance_1 = 0
    # cnt_allowance_0 = 0
    # #here we generate two lists to hold the partial selected items
    # print("inverted sol: ", best_sol_from_rl_inverted)
    # for i in range(0,len(best_sol_from_rl_inverted)):
    #     if best_sol_from_rl_inverted[i] == 1 and cnt_allowance_1 <= 2:
    #      #   print("i ", i)
    #      #   print("val ", all_ind[current_instance_number][i])
    #         sel.append(all_ind[current_instance_number][i])
    #     elif cnt_allowance_1 < 2: 
    #         cnt_allowance_1+=1
    #         sel.append(all_ind[current_instance_number][i])
    #     else:
    #         break
    
    # for j in range(len(best_sol_from_rl_inverted)-1,-1,-1):
    #     if best_sol_from_rl_inverted[j] == 0 and cnt_allowance_0 <= 2:
    #       #  print("j ", j)
    #       #  print("val ", all_ind[current_instance_number][j])
    #         sel.append(all_ind[current_instance_number][j])
    #     elif cnt_allowance_0 < 2: 
    #         cnt_allowance_0+=1
    #         sel.append(all_ind[current_instance_number][j])
    #     else:
    #         break
    
    # def_prd_perc_a2 = (len(nsel)+len(sel))/number_of_items
    
    # default_prediction_perc_a2.append(def_prd_perc_a2)

    
    # start_rl_timer_2 = timeit.default_timer()
    
    # if(len(best_sol_from_rl)!=0):
    #     rl_obj, rl_sol = Cplex_Methods.solve_using_cplex_reduced_rl(const_list,all_ins[current_instance_number][0],number_of_items,num_of_const,best_sol_from_rl,current_instance_number,sel,nsel,curr_best_obj_rl,curr_best_rl_sol)
    #     print("objective using RL: ", rl_obj)
    #     per_change_2 = ((rl_obj-obj_full)/obj_full)*100
    #     print("Percentage change is: ", per_change_2)
    #     cnt_2=0
    #     for i in range(0,len(rl_sol)):
    #         if rl_sol[i] == sol_full[i]:
    #             cnt_2+=1
    #     print("Number of common items is : ", cnt_2)
    #     rl_all_correct_items_a2.append(copy.deepcopy(cnt_2))
    #     rl_all_obj_a2.append(copy.deepcopy(rl_obj))
    #     rl_all_gaps_a2.append(copy.deepcopy(per_change_2))
    #     end_rl_timer_2 = timeit.default_timer()
    #     timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
    #     rl_all_time_a2.append(copy.deepcopy(timeSolving_2))
    # else:
    #     rl_all_obj_a2.append(-1)
    #     rl_all_gaps_a2.append(-1)
    #     end_rl_timer_2 = timeit.default_timer()
    #     timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
    #     rl_all_time_a2.append(copy.deepcopy(timeSolving_2))
    #    # break

    #Here we define 85% predictions 
    sel = []
    nsel = []
    #here we generate two lists to hold the partial selected items
    print("inverted sol: ", best_sol_from_rl_inverted)
    cnt_85_1 = 0
    cnt_85_0 = 0
    ##predicting 70% 1 and 15% 0
    for i in range(0,len(best_sol_from_rl_inverted)):
     #   if best_sol_from_rl_inverted[i] == 1:
        if cnt_85_1 <= int(0.72*number_of_items):
         #   print("i ", i)
         #   print("val ", all_ind[current_instance_number][i])
           # if best_sol_from_rl_inverted[i]==1:
            sel.append(all_ind[current_instance_number][i])
            cnt_85_1 +=1
        else:
            break
    
    for j in range(len(best_sol_from_rl_inverted)-1,-1,-1):
#        if best_sol_from_rl_inverted[j] == 0:
        if cnt_85_0 <= int(0.1*number_of_items):
            #  print("j ", j)
          #  print("val ", all_ind[current_instance_number][j])
          #  if best_sol_from_rl_inverted[j]==0:
            nsel.append(all_ind[current_instance_number][j])
            cnt_85_0 +=1
        else:
            break
    
    def_prd_perc = (len(nsel)+len(sel))/number_of_items
    
    default_prediction_perc_p1.append(def_prd_perc)
    start_rl_timer_2 = timeit.default_timer()
    if(len(best_sol_from_rl)!=0):
        rl_obj, rl_sol = Cplex_Methods.solve_using_cplex_reduced_rl(const_list,all_ins[current_instance_number][0],number_of_items,num_of_const,best_sol_from_rl,current_instance_number,sel,nsel,curr_best_obj_rl,curr_best_rl_sol)
        print("objective using RL: ", rl_obj)
        per_change_2 = ((rl_obj-obj_full)/obj_full)*100
        print("Percentage change is: ", per_change_2)
        cnt_2=0
        for i in range(0,len(rl_sol)):
            if rl_sol[i] == sol_full[i]:
                cnt_2+=1
        print("Number of common items is : ", cnt_2)
        p1_85_all_correct_items.append(copy.deepcopy(cnt_2))
        p1_85_all_obj.append(copy.deepcopy(rl_obj))
        p1_85_all_gaps.append(copy.deepcopy(per_change_2))
        end_rl_timer_2 = timeit.default_timer()
        timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
        p1_85_all_time.append(copy.deepcopy(timeSolving_2))
    else:
        p1_85_all_obj.append(0)
        p1_85_all_gaps.append(0)
        end_rl_timer_2 = timeit.default_timer()
        timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
        p1_85_all_time.append(copy.deepcopy(timeSolving_2))
       # break
   
    
    #Here se define 95% predictions
    sel = []
    nsel = []
    #here we generate two lists to hold the partial selected items
    cnt_95_1 = 0
    cnt_95_0 = 0
    print("inverted sol: ", best_sol_from_rl_inverted)
    for i in range(0,len(best_sol_from_rl_inverted)):
      #  if best_sol_from_rl_inverted[i] == 1:
        if cnt_95_1 <= int(0.82*number_of_items):   
         #   print("i ", i)
         #   print("val ", all_ind[current_instance_number][i])
         #   if best_sol_from_rl_inverted[i] == 1:
            sel.append(all_ind[current_instance_number][i])
            cnt_95_1 +=1
        else:
            break
    
    for j in range(len(best_sol_from_rl_inverted)-1,-1,-1):
      #  if best_sol_from_rl_inverted[j] == 0:
        if cnt_95_0 <= int(0.12*number_of_items):
          #  print("j ", j)
          #  print("val ", all_ind[current_instance_number][j])
          #  if best_sol_from_rl_inverted[j]==0:
            nsel.append(all_ind[current_instance_number][j])
            cnt_95_0 +=1
        else:
            break
    
    def_prd_perc = (len(nsel)+len(sel))/number_of_items
    
    default_prediction_perc_p2.append(def_prd_perc)
    start_rl_timer_2 = timeit.default_timer()
    if(len(best_sol_from_rl)!=0):
        rl_obj, rl_sol = Cplex_Methods.solve_using_cplex_reduced_rl(const_list,all_ins[current_instance_number][0],number_of_items,num_of_const,best_sol_from_rl,current_instance_number,sel,nsel,curr_best_obj_rl,curr_best_rl_sol)
        print("objective using RL: ", rl_obj)
        per_change_2 = ((rl_obj-obj_full)/obj_full)*100
        print("Percentage change is: ", per_change_2)
        cnt_2=0
        for i in range(0,len(rl_sol)):
            if rl_sol[i] == sol_full[i]:
                cnt_2+=1
        print("Number of common items is : ", cnt_2)
        p1_95_all_correct_items.append(copy.deepcopy(cnt_2))
        p1_95_all_obj.append(copy.deepcopy(rl_obj))
        p1_95_all_gaps.append(copy.deepcopy(per_change_2))
        end_rl_timer_2 = timeit.default_timer()
        timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
        p1_95_all_time.append(copy.deepcopy(timeSolving_2))
    else:
        p1_95_all_obj.append(0)
        p1_95_all_gaps.append(0)
        end_rl_timer_2 = timeit.default_timer()
        timeSolving_2 = end_rl_timer_2 - start_rl_timer_2
        p1_95_all_time.append(copy.deepcopy(timeSolving_2))
       # break




print("Cplex Objectives: ", cplex_all_obj)
print("Knn Objectives: ", knn_all_obj)
print("Rl Objectives: ", rl_all_obj)
print("Cplex Time: ", cplex_all_time)
print("Knn Time: ", knn_all_time)
print("Rl Time: ", rl_all_time)
print("Knn Gaps: ", knn_all_gaps)
print("Rl Gaps: ", rl_all_gaps)
print("RL Only Time: ", rl_only_all_time)
print("Rl Only Gaps: ", rl_only_all_gaps)

print("=============================================================")

print("mean cplex gap", np.mean(cplex_all_gaps))
print("mean knn gap", np.mean(knn_all_gaps))
print("mean rl gap", np.mean(rl_all_gaps))
print("mean rl only gap", np.mean(rl_only_all_gaps))
print("=============================================================")

print("mean cplex time", np.mean(cplex_all_time))
print("mean knn time", np.mean(knn_all_time))
print("mean rl time", np.mean(rl_all_time))
print("mean rl only time", np.mean(rl_only_all_time))
print("=============================================================")
#print("mean cplex items", np.mean(cplex_all_time))
print("mean knn items", np.mean(knn_all_correct_items))
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
print("Average objective value knn: ", np.mean(knn_all_obj))
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