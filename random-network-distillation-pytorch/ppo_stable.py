import gym

from stable_baselines3 import PPO_RND
import retro
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import GrayScaleObservation, ResizeObservation


n_envs = 128
num_episodes = 3000
model_name = 'first_dungeon_three_keys'
state_name = 'first_dungeon_full_health_three_keys'
n_steps = 1024
def make_env(index=0, state_name='first_dungeon_full_health_three_keys'):
    def _init():
        env = retro.make(game='LegendOfZelda-Nes', state=state_name, obs_type=retro.Observations.IMAGE, use_render=False,)
        env = Monitor(env, f'./monitor_logs/{model_name}/{str(index)}')
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, 84)
        return env

    set_random_seed(index)
    return _init

# Parallel environments
#env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
#Adding a comment
#env = retro.make(game='LegendOfZelda-Nes', state=state_name, obs_type=retro.Observations.IMAGE, use_render=False,)
#env = GrayScaleObservation(env, keep_dim=True)
#env = ResizeObservation(env, 84)
if __name__ == "__main__":
    n_steps = 128

    # Frame-stacking with 4 frames
    env = SubprocVecEnv([make_env(i, state_name) for i in range(4)])
    env = VecFrameStack(env, n_stack=4)
    model = PPO_RND("CnnPolicy", env, verbose=1)
    for i in range(num_episodes):
        model.learn(total_timesteps=1000)
        
        if (i % 100 == 0):
            model.save("Legend_of_Zelda_Three_keys_2")

    #del model # remove to demonstrate saving and loading

    #model = PPO_RND.load("Legend_of_Zelda_Three_keys")
    print("Algorithm Done")