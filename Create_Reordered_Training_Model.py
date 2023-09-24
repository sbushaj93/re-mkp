# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:18:05 2022

@author: Sabah
"""


import Reordered_Knp_Env as rke
import timeit
import math
from stable_baselines3 import DQN, A2C
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os


class Create_Reordered_Training_Model():
    items = 0
    training_steps = 0

    def __init__(self,number_of_items, constraints, cost_values, ind, full_ins, training_steps, run_id, rhs, sol_kmean, obj_kmean):
        self.run_id = run_id
        self.number_of_items = number_of_items
        self.constraints = constraints
        self.cost_values = cost_values
        self.ind = ind
        self.full_ins = full_ins
        self.training_steps = training_steps
        self.rhs = rhs
        self.sol_kmean = sol_kmean
        self.obj_kmean = obj_kmean

    def create_initial_env(self):
        self.env = DummyVecEnv([lambda: rke.Reordered_Knp_Env(self.number_of_items, self.constraints, self.cost_values, self.ind, self.full_ins, self.run_id, self.rhs, self.sol_kmean, self.obj_kmean)])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10)
        self.size_2d = int(math.sqrt(self.number_of_items))
        return self.env


    def train_reordered_model(self, model):
        startLearning = timeit.default_timer()
        model.learn(self.training_steps)
        endLearning = timeit.default_timer()
        print("Time to learn: ", endLearning - startLearning)
    #    self.save_trained_model(model)

    def save_trained_model(self, model):
        # Don't forget to save the VecNormalize statistics when saving the agent
        log_dir = "model_and_env/"
        model.save(log_dir + str(self.run_id) + "_knp_env_multiInstance")
        stats_path = os.path.join(log_dir, str(self.run_id) + "_vec_normalize_multiInstance.pkl")
        self.env.save(stats_path)
        print("Model Saved!!!")

    def create_model(self):
        model = A2C('MlpPolicy', self.env, learning_rate=0.001,verbose=1)
        return model


    def reporting(self):
        print("here gather info during run")
