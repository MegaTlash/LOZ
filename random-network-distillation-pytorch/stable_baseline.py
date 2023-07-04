import os
import retro

from stable_baselines3 import PPO_RND   
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import numpy as np

from gym.wrappers import GrayScaleObservation, ResizeObservation

model_name = 'remake_v5_fh_resize-12(1)-boss(1)-1(1)-2(1)-3(1)-4(1)-5(1)-6(1)-7(1)-8(1)-9(1)-10(1)-11(0)'
n_envs = 4
state_name = 'first_dungeon_first_room'
n_steps = 1024

os.makedirs('./monitor_logs/' + model_name, exist_ok=True)
os.makedirs('./checkpoints/' + model_name, exist_ok=True)

def make_env(index=0, state_name='first_dungeon_first_room'):
    def _init():
        env = retro.make(game='LegendOfZelda-Nes', state=state_name, obs_type=retro.Observations.IMAGE, use_render=False,)
        #env = Monitor(env, f'./monitor_logs/{model_name}/{str(index)}')
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, 84)
        return env

    set_random_seed(index)
    return _init


env = SubprocVecEnv([make_env(i, state_name) for i in range(n_envs)])
env = VecFrameStack(env, n_stack=4)


model = PPO_RND("CnnPolicy", env, verbose=1, device='cuda', target_kl=0.03)
model.learn(total_timesteps = 3000)
moodel.save("ppo-rnd_LOZ")


#del model

#model = PPO_RND.load("ppo-rnd_LOZ")


