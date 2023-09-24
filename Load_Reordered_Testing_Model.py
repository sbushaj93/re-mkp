import Reordered_Knp_Env as rke
import timeit
import math
from stable_baselines3 import DQN, A2C, PPO
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os


class Load_Reordered_Testing_Model():
    items = 0
    training_steps = 0

    def __init__(self,number_of_items, constraints, cost_values, ind, full_ins, training_steps, run_id, rhs, sol_kmean, obj_kmean, stats_path):
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
        self.stats_path = stats_path
        #self.env = DummyVecEnv([lambda: rke.Reordered_Knp_Env(number_of_items, constraints, cost_values,ind, full_ins, run_id, rhs, sol_kmean, obj_kmean)])
        #self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10)
        #self.env = VecNormalize.load(stats_path, self.env)
        self.size_2d = int(math.sqrt(number_of_items))

    def create_initial_model(self):
        self.env = DummyVecEnv([lambda: rke.Reordered_Knp_Env(self.number_of_items, self.constraints, self.cost_values,self.ind, self.full_ins, self.run_id, self.rhs, self.sol_kmean, self.obj_kmean)])
        #self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10)
        self.env = VecNormalize.load(self.stats_path, self.env)
        return self.env



    def load_trained_model(self, model_name, algo, log_dir):
        if model_name == 'DQN':
            return DQN.load(log_dir + str(algo) +"_knp_env_multiInstance")
        elif model_name == 'PPO':
            return PPO.load(log_dir + str(algo) +"_knp_env_multiInstance")
        elif model_name == 'A2C':
            #print(log_dir + str(algo) +"_knp_env_multiInstance")
            return A2C.load(log_dir + str(algo) +"_knp_env_multiInstance")
        else:
            print("provide the correct algorithm")

    def save_trained_model(self, model):
        # Don't forget to save the VecNormalize statistics when saving the agent
        log_dir = "model_and_env/"
        model.save(log_dir + str(self.run_id) + "_knp_env_multiInstance")
        stats_path = os.path.join(log_dir, str(self.run_id) + "_vec_normalize_multiInstance.pkl")
        self.env.save(stats_path)
        print("Model Saved!!!")


