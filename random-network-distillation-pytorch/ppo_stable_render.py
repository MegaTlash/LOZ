import os
import retro

from stable_baselines3 import PPO_RND
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import GrayScaleObservation, ResizeObservation
from render_browser import render_browser

if __name__ == "__main__":

    model_name = '12_states_decay_vecnorm'

    @render_browser
    def test_agent():
        env = retro.make(game='LegendOfZelda-Nes', state='first_dungeon_full_health_three_keys', obs_type=retro.Observations.IMAGE, use_render=False,)
        # env = GrayScaleObservation(env, keep_dim=True)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, 84)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=4, channels_order='last')
        env = VecNormalize(env)
        #print(env.observation_space.shape)
        #file_name = '9_55_avg_reward'
        model = PPO_RND.load("Legend_of_Zelda_Three_keys_2", env=env, verbose=1, device='cuda')

        done = False
        obs = env.reset()

        while True:

            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            #print(obs.shape)

            yield env.render("rgb_array")

            if done:
                env.reset()

    test_agent()