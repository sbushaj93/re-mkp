

"""
we need:
env,
model,
best_obj_from_rl,
best_obj_from_rl,
obj_cpx,
curr_best_gap_rl,

rl_res = [best_obj_from_rl, best_sol_from_rl, curr_best_gap_rl

"""
import timeit
import cplex_methods as cpx


def prediction_loop(rl_loops, model, env, obj_cpx, sol_cpx):
    obs = env.reset()
    cnt = 0
    curr_best_gap_rl = -1
    curr_best_obj_rl = -1
    curr_best_rl_sol = []
    start_rl_timer = timeit.default_timer()
    while(cnt < rl_loops):
        cnt+=1
        #print("Rl step: ", cnt)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        #print(info)
        best_sol_from_rl = info[0]['best_sol']
        best_obj_from_rl = info[0]['best_obj']
        best_sol_from_rl_inverted = info[0]['best_inv']
        if done and len(best_sol_from_rl)!=0:
            print("Goal reached!", "reward=", reward)
         #   print("RL SOL:", best_sol_from_rl)
            print("Best obj from rl ", best_obj_from_rl)
            rl_only_change = ((best_obj_from_rl-obj_cpx)/obj_cpx)*100
            print("Percentage change is: ", rl_only_change)
            curr_best_gap_rl, curr_best_rl_sol, curr_best_obj_rl = sol_eval(curr_best_gap_rl, rl_only_change, best_sol_from_rl,best_obj_from_rl, curr_best_obj_rl, curr_best_rl_sol)
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
    end_rl_timer = timeit.default_timer()
    timeSolving = end_rl_timer - start_rl_timer
    cnt_rl=0
    for i in range(0,len(curr_best_rl_sol)):
        if curr_best_rl_sol[i] == sol_cpx[i]:
            cnt_rl+=1
    print("Number of common items is : ", cnt_rl)
    return cnt_rl, curr_best_gap_rl, curr_best_obj_rl, curr_best_rl_sol, timeSolving, best_sol_from_rl_inverted


def assert_default_prediction(best_sol_from_rl_inverted, ind):
    #Here we define default predictions
    sel = []
    nsel = []
    cnt_allowance_1 = 2
    cnt_allowance_0 = 2
    #here we generate two lists to hold the partial selected items
    #print("inverted sol: ", best_sol_from_rl_inverted)
    for i in range(0,len(best_sol_from_rl_inverted)):
        if best_sol_from_rl_inverted[i] == 1 and cnt_allowance_1 <= 2:
         #   print("i ", i)
         #   print("val ", all_ind[current_instance_number][i])
            sel.append(ind[i])
        elif cnt_allowance_1 < 2:
            cnt_allowance_1+=1
        else:
            break

    for j in range(len(best_sol_from_rl_inverted)-1,-1,-1):
        if best_sol_from_rl_inverted[j] == 0 and cnt_allowance_0 <= 2:
          #  print("j ", j)
          #  print("val ", all_ind[current_instance_number][j])
            nsel.append(ind[j])
        elif cnt_allowance_0 < 2:
            cnt_allowance_0+=1
        else:
            break

    def_prd_perc = (len(nsel)+len(sel))/len(ind)
    return def_prd_perc, sel, nsel


def assert_default_prediction_85_perc(best_sol_from_rl_inverted, ind):
    sel = []
    nsel = []
    #here we generate two lists to hold the partial selected items
    cnt_85_1 = 0
    cnt_85_0 = 0
    ##predicting 70% 1 and 15% 0
    for i in range(0,len(best_sol_from_rl_inverted)):
     #   if best_sol_from_rl_inverted[i] == 1:
        if cnt_85_1 <= int(0.72*len(ind)):
         #   print("i ", i)
         #   print("val ", all_ind[current_instance_number][i])
           # if best_sol_from_rl_inverted[i]==1:
            sel.append(ind[i])
            cnt_85_1 +=1
        else:
            break

    for j in range(len(best_sol_from_rl_inverted)-1,-1,-1):
#        if best_sol_from_rl_inverted[j] == 0:
        if cnt_85_0 <= int(0.1*len(ind)):
            #  print("j ", j)
          #  print("val ", all_ind[current_instance_number][j])
          #  if best_sol_from_rl_inverted[j]==0:
            nsel.append(ind[j])
            cnt_85_0 +=1
        else:
            break

    def_prd_perc = (len(nsel)+len(sel))/len(ind)
    return def_prd_perc, sel, nsel

def assert_default_prediction_95_perc(best_sol_from_rl_inverted, ind):
       #Here se define 95% predictions
    sel = []
    nsel = []
    #here we generate two lists to hold the partial selected items
    cnt_95_1 = 0
    cnt_95_0 = 0
    #print("inverted sol: ", best_sol_from_rl_inverted)
    for i in range(0,len(best_sol_from_rl_inverted)):
      #  if best_sol_from_rl_inverted[i] == 1:
        if cnt_95_1 <= int(0.82*len(ind)):
         #   print("i ", i)
         #   print("val ", all_ind[current_instance_number][i])
         #   if best_sol_from_rl_inverted[i] == 1:
            sel.append(ind[i])
            cnt_95_1 +=1
        else:
            break

    for j in range(len(best_sol_from_rl_inverted)-1,-1,-1):
      #  if best_sol_from_rl_inverted[j] == 0:
        if cnt_95_0 <= int(0.12*len(ind)):
          #  print("j ", j)
          #  print("val ", all_ind[current_instance_number][j])
          #  if best_sol_from_rl_inverted[j]==0:
            nsel.append(ind[j])
            cnt_95_0 +=1
        else:
            break

    def_prd_perc = (len(nsel)+len(sel))/len(ind)
    return def_prd_perc, sel, nsel



def assert_default_prediction_1_allowance(best_sol_from_rl_inverted, ind):
    #Here we define default prediction with 1 allowance each side
    sel = []
    nsel = []
    cnt_allowance_1 = 1
    cnt_allowance_0 = 1
    #here we generate two lists to hold the partial selected items
    #print("inverted sol: ", best_sol_from_rl_inverted)
    for i in range(0,len(best_sol_from_rl_inverted)):
        if best_sol_from_rl_inverted[i] == 1 and cnt_allowance_1 <= 2:
         #   print("i ", i)
         #   print("val ", all_ind[current_instance_number][i])
            sel.append(ind[i])
        elif cnt_allowance_1 < 2:
            cnt_allowance_1+=1
            sel.append(ind[i])
        else:
            break

    for j in range(len(best_sol_from_rl_inverted)-1,-1,-1):
        if best_sol_from_rl_inverted[j] == 0 and cnt_allowance_0 <= 2:
          #  print("j ", j)
          #  print("val ", all_ind[current_instance_number][j])
            nsel.append(ind[j])
        elif cnt_allowance_0 < 2:
            cnt_allowance_0+=1
            nsel.append(ind[j])
        else:
            break

    def_prd_perc_a1 = (len(nsel)+len(sel))/len(ind)
    return def_prd_perc_a1, sel, nsel






def get_partial_pred_sol(best_sol_from_rl,cpx_obj,cpx_sol, constraints, cost_values, number_of_items, num_of_const,sel,nsel):
    start_rl_timer_2 = timeit.default_timer()
    if(len(best_sol_from_rl)!=0):
        rl_obj, rl_sol = cpx.solve_using_cplex_reduced_rl(constraints,cost_values,number_of_items,num_of_const,best_sol_from_rl,sel,nsel)
        print("objective using RL: ", rl_obj)
        per_change_2 = ((rl_obj-cpx_obj)/cpx_obj)*100
        print("Percentage change is: ", per_change_2)
        cnt_2=0
        for i in range(0,len(rl_sol)):
            if rl_sol[i] == cpx_sol[i]:
                cnt_2+=1
        print("Number of common items is : ", cnt_2)

    else:
        cnt_2 = 0
        rl_obj = 0
        per_change_2 = 0
    end_rl_timer_2 = timeit.default_timer()
    time_solving_2 = end_rl_timer_2 - start_rl_timer_2
    return cnt_2, rl_obj, per_change_2, time_solving_2




def sol_eval(curr_best_gap_rl, new_gap, new_sol, newObj, old_obj, old_sol):
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
        return curr_best_gap_rl, tmp, tmp_obj
    elif new_gap < curr_best_gap_rl:
        tmp = new_sol
        curr_best_gap_rl = new_gap
        tmp_obj = newObj
        return curr_best_gap_rl, tmp, tmp_obj
    else:
        return curr_best_gap_rl, old_sol, old_obj



    #print("SOL_EVAL OBJ 2: ",tmp_obj)
    #print("SOL_EVAL SOL 2: ",tmp)
    #return curr_best_gap_rl, tmp, tmp_obj
    #compare two rl solutions and choose the one with the lowest gap
