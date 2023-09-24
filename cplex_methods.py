

import knapsack_methods as knp
import cplex

optimality_gap = 0.001
opt_gap_m1 =0.01

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
    for x in range(ndec):
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

    cpl.write("model.lp")
    cpl.solve()


    #print(cpl.solution.get_values())
    #for it in cpl.solution.get_values():
    #    sol_val.append(int(it))
    sol_val = cpl.solution.get_values()
    #print(cpl.solution.get_objective_value())
    #print("Solution status : ", cpl.solution.get_solution_type())
    #print("Relative Gap is: ",cpl.solution.MIP.get_mip_relative_gap())
    #print("Number of variables : ",cpl.variables.get_num())
    #print("Number of binary variables : ",cpl.variables.get_num_binary())
    #print("CPLEX GET STATS: ", cpl.get_stats())
 #   print(cpl.rows)
    obj = cpl.solution.get_objective_value()
    #print("GAP for solve_using_cplex_reduced_knn: " , cpl.solution.MIP.get_mip_relative_gap())
    return obj,sol_val



def solve_using_cplex(const_list,objf,ndec,ncon):
    #var_names,con_names, objf, cons, ndec, ncon, b_r, sn, lb, up
    #sn = ['GGGGG']
    #50 constraints
    sn = []
    #sn_old = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
    sn2 = 'G'*int(ncon)
    sn.append(sn2)
    #400 items
    #sn = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']

    lb = []
    ub = []
    var_names = []
    for x in range(ndec):
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
    for y in range(int(ncon)):
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
    cpl.parameters.timelimit.set(600)

    #set max sense temporarily
    cpl.objective.set_sense(cpl.objective.sense.minimize)
    t= cpl.variables.type

    cpl.parameters.mip.tolerances.mipgap.set(float(optimality_gap))

    #decision variables

    cpl.variables.add(obj = objf,
                     lb = lb,
                     ub = ub,
                     types = [t.binary]*ndec,
                     names = var_names)

    cpl.linear_constraints.add(lin_expr = conts_all_with_names, senses = sn, rhs = b_rhs, names = con_names)

    cpl.write("model.lp")
    cpl.solve()


    #print(cpl.solution.get_values())
    #for it in cpl.solution.get_values():
    #    sol_val.append(int(it))
    sol_val = cpl.solution.get_values()
    #print(cpl.solution.get_objective_value())
    obj = cpl.solution.get_objective_value()
    #print("CPLEX GET STATS: ", cpl.get_stats())
    #print("GAP for solve_using_cplex: ",cpl.solution.MIP.get_mip_relative_gap())
    return obj,sol_val, cpl.solution.MIP.get_mip_relative_gap()

def solve_using_cplex_relaxed(const_list,objf,ndec,ncon):
        #var_names,con_names, objf, cons, ndec, ncon, b_r, sn, lb, up
        #sn = ['GGGGG']
        #50 constraints
        sn = []
        #sn_old = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
        sn2 = 'G'*int(ncon)
        sn.append(sn2)
        #400 items
        #sn = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']

        lb = []
        ub = []
        var_names = []
        for x in range(ndec):
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
        for y in range(int(ncon)):
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

        cpl.write("model.lp")
        cpl.solve()


        #print(cpl.solution.get_values())
        #for it in cpl.solution.get_values():
        #    sol_val.append(int(it))
        sol_val = cpl.solution.get_values()
        #print(cpl.solution.get_objective_value())
        obj = cpl.solution.get_objective_value()
        #print("CPLEX GET STATS: ", cpl.get_stats())
        #print(cpl.solution.MIP.get_mip_relative_gap())
        return obj,sol_val


def solve_using_cplex_reduced_rl(const_list,objf,ndec,ncon,b_sol,sel,nsel):
    #var_names,con_names, objf, cons, ndec, ncon, b_r, sn, lb, up
    #sn = ['GGGGG']
    #50 constraints
    sn = []
    #sn_old = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']
    sn2 = 'G'*int(ncon)
    sn.append(sn2)
    #400 items
    #sn = ['GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG']



    #here create two arrays which hold indexes of the begining and end of the sorted prediction

#    beg_arr = all_ind[curr_ins] [:int (0.1 * number_of_items)]
#    end_arr = all_ind[curr_ins] [int (0.9 * number_of_items):]

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

    #     # #we can force some of the variables to be 1 or 0 here
    #     # if(x < int(0.15*number_of_items*number_of_items) or x > int(0.85*number_of_items*number_of_items)):
    #     #     lb.append(float(b_sol[x]))
    #     #     ub.append(float(b_sol[x]))
    #     # else:
    #     #     lb.append(0.0)
    #     #     ub.append(1.0)
    #     nm = "x%d" % (x+1)
    #     var_names.append(nm)
        #knapsack_gen.write_to_file(str(temp),0)
    #print("forced items to 1 ", len(sel))
    #print("forced items to 0 ", len(nsel))
    lb = []
    ub = []
    var_names = []
    for x in range(ndec):
        #temp = float(random.randint(1,100))
        #dec_var.append(temp)
        #print(x)
        #check if item in beg_arr
        if x in sel:
            lb.append(1)
            ub.append(1)

        #check if item in end_arr
        if x in nsel:
            lb.append(0)
            ub.append(0)


        if x not in sel and x not in nsel:
            lb.append(0.0)
            ub.append(1.0)
        nm = "x%d" % (x+1)
        var_names.append(nm)


    con_names = []
    b_rhs = []
    conts_all_with_names = []
    fin_con = []
    #sn = ''
    for y in range(int(ncon)):
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
                     types = [t.binary]*ndec,
                     names = var_names)

    cpl.linear_constraints.add(lin_expr = conts_all_with_names, senses = sn, rhs = b_rhs, names = con_names)

    cpl.write("model.lp")
    cpl.solve()


    #print(cpl.solution.get_values())
    #for it in cpl.solution.get_values():
    #    sol_val.append(int(it))
    sol_val = cpl.solution.get_values()
    #print(cpl.solution.get_objective_value())
    obj = cpl.solution.get_objective_value()
    #print("CPLEX GET STATS: ", cpl.get_stats())
    #print("GAP for solve_using_cplex_reduced_rl: ",cpl.solution.MIP.get_mip_relative_gap())
    return obj,sol_val



def load_cpx_sols(loc, number_of_items, instance, dist_num):

    name_obj = loc + "cpx_obj_"+str(number_of_items)+"_"+str(instance) +"_"+ str(dist_num) + ".npy"
    name_sol = loc + "cps_sol_"+str(number_of_items)+"_"+str(instance) +"_"+ str(dist_num) + ".npy"
    name_gap = loc + "cps_gap_"+str(number_of_items)+"_"+str(instance) +"_"+ str(dist_num) + ".npy"

#    obj_full, sol_full,cpx_gap = Cplex_Methods.solve_using_cplex(tmp_full_cons,all_ins[current_instance_number][0],number_of_items,num_of_const)

 ## uncomment these to use fixed saved solutions for cplex
    return knp.loadList(name_obj), knp.loadList(name_sol), knp.loadList(name_gap)
