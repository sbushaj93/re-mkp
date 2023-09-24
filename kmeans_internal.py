import timeit

import numpy as np
from scipy.spatial.distance import pdist, squareform
from math import sqrt
import pandas as pd
import random
import copy
import cplex_methods as cpx


def __int__(self, number_of_items, cost_values,  total_iter, const, n_clusters):
    self.num_of_items = number_of_items
    self.cost_values = cost_values
    self.num_of_iter = total_iter
    self.number_of_clusters = n_clusters
    self.data_to_cluster = const
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
            distances_arr.append(jaccard(centroid, datapoint))
    return distances_arr

#define Jaccard Similarity function
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


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
    #print(list_array)
    #print(c_names)
    return list_array, c_names

def knn_constraint_out(number_of_clusters, data_to_cluster, num_of_iter, num_of_items):
    startClustering = timeit.default_timer()
    '''This function reads a csv file with pandas, prints the dataframe and returns
    the two columns in numpy ndarray for processing as well as the country names in
    numpy array needed for cluster matched results'''
    #data1 = arr
    x = format_data(data_to_cluster)

    k = int(number_of_clusters)

    for iteration in range(0, num_of_iter):
        # Print the iteration number
        #print("ITERATION: " + str(iteration+1))
        # assign the function to a variable as it has more than one return value
        assigning = assign_to_cluster_mean_centroid(x, k)

        # Create the dataframe for vizualisation
    #    cluster_data = pd.DataFrame({'x': x[0][0:, :],
    #                                 'label': assigning[0],
    #                                 'const': x[1]})

        cluster_data = pd.DataFrame(x[0][0:, :]) # , assigning[0], x[1]})
        #print(cluster_data)
        cluster_data['Constraint'] = x[1]
        cluster_data['Label'] = assigning[0]


        # Create the dataframe and grouping, then print out inferences
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
            array = np.reshape(array, (len(array),num_of_items ))
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
    #    print("Total Distance: ", total_distance )
        # print the summed distance
    #    print("Summed distance of all clusters: " + str(sum(total_distance)))
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
    distances_arr_re = np.reshape(distance_between(
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

def knn_main_start(self):
    #print("Current Instance KNN: ",self.data_to_cluster)
    const_list = copy.deepcopy(self.data_to_cluster)

    #this maybe needs to change
    arr = pd.DataFrame(const_list).apply(pd.Series)
    arr = arr.rename(columns=lambda x : 'item_'+str(x))

    #remove the objective weights
    arr = arr.drop(0)

    #list to remain constant to check for feasibility
    static_list_of_const = copy.deepcopy(arr.to_numpy().tolist())

    #list containing the constraints not in the new problem
    #not_in_problem = copy.deepcopy(arr.to_numpy().tolist())
    #list containing the constraint of the new problem
    in_problem = []
    #data to be used for clustering
    arr

    res0 = self.knn_constraint_out(self.num_of_iter,arr,self.number_of_clusters)
    r=[]
    for i in range (1, self.number_of_clusters):
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
    best_sol_knn = [0]*self.num_of_items
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
    #    print("LOOP Number ", lp)
        lp+=1

        newObjective, newSolution = cpx.solve_using_cplex_reduced_knn(in_problem,self.cost_values,self.num_of_items,len(in_problem))
    #    print("objective using CPLEX reduced: ", newObjective)

        #to compare with full cpx only
        #new_per_change = ((newObjective-obj_full)/obj_full)*100
        #print("Percentage change is: ", new_per_change)
        #cnt_l=0
        #for i in range(0,len(sol_full)):
        #    if newSolution[i] == sol_full[i]:
        #        cnt_l+=1
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

    #    print(violation_index)
    #    print(violation_amount)
    #    print("Violated Constraints: ", len(violated_const))
    #    print("Satisfied Constraints: ", len(satisfied_const))

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
        #    print("The list with differences is: ")
        #    print(violation_amount)
            for i in range(6):
                #each loop add the most violated constraint
                #rhs - lhs is +, so the larger the more violated
                max_vio = np.argmax(violation_amount)
                idx_vio = violation_index[max_vio]
                in_problem.append(copy.deepcopy(violated_const[idx_vio]))
        #        print("added constraint with violation: ", violation_amount[max_vio])
                del violated_const[idx_vio]
                del violation_amount[max_vio]
           #     del violation_index[max_vio]
              #  del static_list_of_const[violated_const[idx_vio]]
        else:
            for x in range(len(violated_const)):
                in_problem.append(copy.deepcopy(violated_const[x]))
        #        print("Adding All Const : ",violated_const[x] )


    endRecursiveSolving = timeit.default_timer()
    timeSolving = endRecursiveSolving - startRecursiveSolving


    #print("Best objective is: ", best_obj_knn)
    #print("Best solution is: ", best_sol_knn)
    #print("LAST objective is: ", last_obj_knn)
    #print("LAST solution is: ", last_sol_knn)

    return last_obj_knn, last_sol_knn, timeSolving
