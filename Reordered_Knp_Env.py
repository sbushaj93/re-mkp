
import copy
import cplex_methods as cpx
import knapsack_methods as knp
import numpy as np
import cv2
import utils as ut
from copy import deepcopy
import math
import gym
from gym import spaces
#from stable_baselines3 import DQN, A2C
#from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.vec_env import DummyVecEnv
#from PIL import Image
import sys


class Reordered_Knp_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, number_of_items,  constraints, cost_values, rev_ind, curr_ins, run_id, rhs, sol_kmean, obj_kmean):
        self.sol_kmean = sol_kmean
        self.obj_kmean = obj_kmean
        self.ccv = cost_values
        self.c_ins = curr_ins
        self.c_ind = rev_ind
        self.c_list = constraints
        self.item_num = int(math.sqrt(number_of_items))
        self.avg_rew_ep = 0
        self.cnt=0
        p1 = np.random.randint(0*self.item_num,self.item_num-1)
        p2 = np.random.randint(0*self.item_num,self.item_num-1)
       #static start each time
        #p1 = int(0.5*number_of_items)
        #p2 = int(0.5*number_of_items)
       #st_point_1 = int(0.7*number_of_items)
        self.agent_pos = (p1,p2)
        #print("STARTING...")
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
        self.curr_feas = True
        self.best_obj_from_rl = -1
        self.best_sol_from_rl = []
        self.best_sol_from_rl_inverted = []

        #print("OBSERVATION",self.observation_space)

    def populate_world(self):
        res = np.full((self.cons_size,self.cons_size),-1)
        knn_rev = self.normal_to_crazy( self.sol_kmean,self.c_ind)
        size = self.item_num
        knn_2d = np.reshape(knn_rev, (size,size))
        for i in range(size):
            for j in range(size):
                res[i][j] = knn_2d[i][j]

        return res

    ###CURRENTLY NOT USED
    def populate_world_again(self,wrd):
        kmean_rev = self.get_reverted_knn_solution(self.sol_kmean)
        size = self.item_num
        knn_2d = np.reshape(kmean_rev, (size,size))
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
    #    self.item_num = int(math.sqrt(number_of_items))
    #    self.ccv =curr_cost_values
    #    self.c_ins = curr_ins
    #    self.c_ind = curr_ind_for_ins
    #    self.c_list = curr_const_list
        print("RESETTING...")
        self.state='P'
        self.current_step = 0
        self.max_step = 450
        self.best_obj_of_episode = 0
        self.avg_rew_ep = 0
        self.cnt=0
    #    r1 = np.random.randint(0.3*self.item_num,self.item_num-1)
    #    r2 = np.random.randint(0.3*self.item_num,self.item_num-1)
        self.curr_feas = True
        #random start each time
        r1 = np.random.randint(self.item_num-5,self.item_num-1)
        r2 = np.random.randint(self.item_num-5,self.item_num-1)
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
     #   tmp_state = 'W'
        if  self.obj_kmean > pre:
           # if self.feas1D(wrd,self.c_ind,self.c_ins) == True:
            if self.curr_feas == True:
                tmp_state = 'W'  #win, end the search +500
        #        else:
        #            tmp_state = 'B'  #go on, but negative result -3
        if  self.obj_kmean <= pre:
           # if self.feas1D(wrd,self.c_ind,self.c_ins) == True:
            if self.curr_feas == True:
                tmp_state = 'G' #go on, positive result: +10
        #        else:
        #            tmp_state = 'B' #go on, negative result - 5
        if pre == post:
            tmp_state = 'S' # go on, but negative result:  - 1
        if post < self.obj_kmean and  self.curr_feas == True: #self.feas1D(wrd,self.c_ind,self.c_ins) == True:
            tmp_state = 'K'
     #   if self.feas1D(wrd,self.c_ind,self.c_ins) == False:
        if self.curr_feas == False:
            tmp_state = 'L'  #loose, end the search  -100
        return tmp_state

    def make_move(self, wrd, nextpos):
        if wrd[nextpos] != -1:
            #  world_temp[next_pos] = 0
            if wrd[nextpos] == 0:
                wrd[nextpos] = 1
            else:
                wrd[nextpos] = 0
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
        post_objective = None
        next_pos = None
    #    current_pos = self.agent_pos
    #    cons_curr_pos = self.agent_pos
        world_temp = copy.deepcopy(self.world)
    #    pre_objective = self.getObjective1D(world_temp,ind_rat_for_train[0], all_ins_for_train[0][0]) #wrd,ind,cost_val
        pre_objective = self.getObjective1D(world_temp,self.c_ind,self.ccv) #wrd,ind,cost_val
        tmp_world_cons = copy.deepcopy(world_temp)
    #    print("current pos of agent: ", self.agent_pos)
    #    print("current action of agent: ", action)
       #then mark that position with -1 or -2, but count the value as 0 (not selected)
        #next_pos = current_pos
        next_pos = self.check_legitimacy(action)

        if next_pos != None:
            world_temp = self.make_move(world_temp,next_pos)
            self.curr_feas = self.feas1D(world_temp,self.c_ind,self.c_ins)
            post_objective = self.getObjective1D(world_temp,self.c_ind,self.ccv)
            self.state = self.get_state(world_temp, pre_objective,post_objective)
            if world_temp[next_pos] == -1:
                self.state = 'E'
                self.count_state_e += 1
            else:
                self.world = world_temp
                self.agent_pos = next_pos
                if post_objective <= pre_objective:# and self.curr_feas ==True:
                    self.best_obj_of_episode = post_objective
                    bs = self.TwoD_to_OneD(world_temp,self.item_num)
                    self.best_sol_from_rl_inverted = copy.deepcopy(bs)
                    self.best_sol = self.crazy_to_normal(bs, self.c_ind)
                    self.best_obj = post_objective
                    self.best_sol_from_rl = copy.deepcopy(self.best_sol)
                    self.best_obj_from_rl = copy.deepcopy(self.best_obj)
        else:
        #if position is none then it is outside of canvas
            self.state = 'O'
            self.count_state_s +=1

    def step(self,action):
        #print("ITERATION: ", self.iteration_num)
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
       #     print("State: ", self.state)
        elif self.state == 'G':
            #here add the conditions below as well
           # print(f'Player {self.current_player} good move')
            reward = +5
         #   reward = 500
            done = False
        #    print("State: ", self.state)
        elif self.state == 'L':
        #    print(f'Player {self.current_player} lost')
            reward = -100
        #    reward = -500
            done = True
        #    print("State: ", self.state)
     #       done = True
        elif self.state == 'W':
            reward = +500
        #    reward = +5000
     #       done = True
            done = True
        #    print("State: ", self.state)
        elif self.state == 'K':
            reward = +500
            done = True
        elif self.state == 'O':
        #    reward = -5000
            reward = -150
        #    print("State: ", self.state)
            #done = False
            if self.count_state_s >= 10:
                #use tunneling, do not end the loop.
            #    done = True
                done = True
                #self.agent_pos=(int(0.4*number_of_items),int(0.4*number_of_items))
                self.count_state_s = 0
            else:
                done = False
        elif self.state == 'E':
            reward = -150
        #    print("State: ", self.state)
            if self.count_state_e >= 10:
                #use tunneling, do not end the loop.
            #    done = True
                done = True
            #    self.agent_pos=(int(0.4*number_of_items),int(0.4*number_of_items))
                self.count_state_e = 0
            else:
                done = False
        #    done = False

        elif self.state == 'S':
            reward = -4
        #    reward = -500
            done = False
        #    print("State: ", self.state)
        # if self.current_step >= self.max_step:
        #     if self.feas1D(self.world,ind_rat_for_train[0],all_ins_for_train[0]) == False:
        #         done = True
        #     else:
        #         self.current_step = 0
       # elif self.feas1D(self.world,self.c_ind, self.c_ins) == False:
        elif self.curr_feas == False:
            if self.best_obj_of_episode <= self.best_obj_to_cmp:
 #               reward = +5000
                self.best_obj_to_cmp = self.best_obj_of_episode
 #           else:
 #               reward = -5000

        self.cnt+=1
        self.avg_rew_ep = (self.avg_rew_ep + reward)/self.cnt
        #print("Average Reward: ", self.avg_rew_ep)
        if done or self.curr_feas == False:
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
        info = {"best_obj":self.best_obj_from_rl, "best_sol":self.best_sol_from_rl, "best_inv":self.best_sol_from_rl_inverted }
        #return obs, self.avg_rew_ep, done, {self.best_obj_from_rl, self.best_sol_from_rl, self.best_sol_from_rl_inverted}
        return obs, self.avg_rew_ep, done, info


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

    def normal_to_crazy(self,sol_kmean, index_arr):
        reverted_form = [0]*len(index_arr)
        for i in range(0,len(index_arr)):
            reverted_form[i] = sol_kmean[index_arr[i]]
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
        print("Current Solution: ")
        self.showRoute(self.world)
        print("State: ", win_or_lose)
        print("EPISODE FINISHED!!!!!!")
        self.success_episode.append(
        'Success' if win_or_lose == 'W' else 'Failure')
        file = open('render_knp'+ str(self.item_num)+'.txt', 'a')
        file.write(' — — — — — — — — — — — — — — — — — — — — — -\n')
        file.write('Executing File = {fn}\n')
        file.write(f'Episode number {self.current_episode}\n')
        file.write(f'{self.success_episode[-1]} in {self.current_step} steps\n')
       #file.write(str(wrd))
        file.write(f'Reward is {rew}')
        file.close()
   #     print("Reward: ", rew)
   #     print("Solution feasibility: ",self.curr_feas)
   #     print("Objective value: ", self.getObjective1D(wrd,self.c_ind, self.ccv))

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
