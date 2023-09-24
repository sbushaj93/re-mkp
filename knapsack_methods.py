import random
from random import *
import numpy as np
import copy





def con_generator(number_of_items):
        temp = []
        for x in range(number_of_items):
            tmp = float(randint(1,10))
            temp.append(tmp)
        #print(temp)
        return temp

def get_index_matrix(cost_values, constraints,rhs, number_of_items, num_of_const):
    ratio_ins_for_train = sort_based_on_ratios(cost_values, constraints,rhs, number_of_items, num_of_const)
    ind = sorted(range(len(ratio_ins_for_train)), key=lambda k: ratio_ins_for_train[k],reverse=False)
    return ind

def combine_cost_and_constraints(cost_values, constraints):
    full_instance = copy.deepcopy(constraints)
    full_instance.insert(0, cost_values)
    return full_instance


def instance_generator(number_of_items,num_of_const, save_data=False, rats=None, dist_num = None):
    constraints = []
    obj_values = []
    #all_con = []
    tst = []
    rhs = []
    #rhs.append(0)
    for x in range(number_of_items):
        temp = float(randint(1,10))
        obj_values.append(temp)
    #ins.append(obj_values)
    tst.append(obj_values)
    for y in range(int(num_of_const)):
        for z in range(1):
            #cons.append([])
            con_new = con_generator(number_of_items)
            #all_con.append(con_new)
            rhs_val = 3/4 * sum(con_new)
            rhs.append(rhs_val)
            #cons[y].append(con_new)
            constraints.append(con_new)
            tst.append(con_new)
            if save_data:
                name_ind = "ind_"+str(number_of_items)+"_"+str(rats) +"_"+ str(dist_num)+ ".npy"
                name_ins = "ins_"+str(number_of_items)+"_"+str(rats) +"_"+ str(dist_num)+ ".npy"
                saveList(combine_cost_and_constraints(obj_values, constraints), name_ins)
                saveList(get_index_matrix(combine_cost_and_constraints(obj_values, constraints), rhs), name_ind)
    return obj_values, constraints, rhs


def sort_based_on_ratios(cost_values, constraints, rhs, number_of_items, number_of_constraints):
    cv = cost_values
    ln_cv = len(cv)
    res = [0]*ln_cv
    ln_mat = len(constraints)


    for i in range(number_of_items):
        # print(i)
        for j in range(number_of_constraints):
        #    print(j)
            res[i] = res[i] + (cost_values[i]/constraints[j][i]/rhs[j])
            #old
           # res[i] = res[i] + (cv[i]/matrix[j][i])

    for i in range(number_of_items):
        res[i] = res[i]/number_of_items

    return res

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

def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")

def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()


def read_instances(loc, number_of_items, rats, dist_num):
    name_ind = loc + "ind_"+str(number_of_items)+"_"+str(rats) +"_"+ str(dist_num)+ ".npy"
    name_ins = loc + "ins_"+str(number_of_items)+"_"+str(rats) +"_"+ str(dist_num)+ ".npy"
    ins = loadList(name_ins)
    ind = loadList(name_ind)
    cost_values = copy.deepcopy(ins[0])
    del ins[0]
    constraints = copy.deepcopy(ins)
    rhs = calculate_rhs(constraints)
    return cost_values, constraints, rhs, ind

def calculate_rhs(cons):
    rhs = []
    for z in cons:
            rhs_val = 3/4 * sum(z)
            rhs.append(rhs_val)
    return rhs


