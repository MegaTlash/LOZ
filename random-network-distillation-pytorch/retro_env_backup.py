import gc
import gym
import gzip
import gym.spaces
import json
import numpy as np
import os
import retro
import retro.data
from gym.utils import seeding
import cv2
import math
import time

gym_version = tuple(int(x) for x in gym.__version__.split('.'))

__all__ = ['RetroEnv']

class Enemy:
    def __init__(self):
        self.current_x = None
        self.current_y = None

        self.health = None
        self.exists = False

        self.time_limit = 500
        self.existance_timer = self.time_limit
        
    def has_moved(self, x, y):
        return x != self.current_x or y != self.current_y

    def reset_existance_timer(self):
        self.existance_timer = self.time_limit

class RetroEnv(gym.Env):
    """
    Gym Retro environment class

    Provides a Gym interface to classic video games
    """
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 60.0}

    def __init__(self, game, state=retro.State.DEFAULT, scenario=None, info=None, use_restricted_actions=retro.Actions.FILTERED,
                 record=False, players=1, inttype=retro.data.Integrations.STABLE, obs_type=retro.Observations.IMAGE, use_render=False):
        if not hasattr(self, 'spec'):
            self.spec = None
        self._obs_type = obs_type
        self.img = None
        self.ram = None
        self.viewer = None
        self.gamename = game
        self.statename = state
        self.initial_state = None
        self.players = players
        self.previous_boss_health = None
        self.previous_link_health = None
        self.use_render = use_render
        self.time_limit = 500
        self.time = 0
        self.max_x = 84
        self.max_y = 84
        self.reward_range = [-16, 16]
        self.old_distance = None

        self.enemy_one_x = None
        self.enemy_one_y = None
        self.enemy_two_x = None
        self.enemy_two_y = None
        self.enemy_three_x = None
        self.enemy_three_y = None
        self.enemy_four_x = None
        self.enemy_four_y = None
        self.enemy_five_x = None
        self.enemy_five_y = None
        self.enemy_six_x = None
        self.enemy_six_y = None

        self.enemy_one_last_pos = (0, 0)
        self.enemy_two_last_pos = (0, 0)
        self.enemy_three_last_pos = (0, 0)
        self.enemy_four_last_pos = (0, 0)
        self.enemy_five_last_pos = (0, 0)
        self.enemy_six_last_pos = (0, 0)

        self.enemy_one_exists = False
        self.enemy_two_exists = False
        self.enemy_three_exists = False
        self.enemy_four_exists = False
        self.enemy_five_exists = False
        self.enemy_six_exists = False

        self.enemy_one_killed = False
        self.enemy_two_killed = False
        self.enemy_three_killed = False
        self.enemy_four_killed = False
        self.enemy_five_killed = False
        self.enemy_six_killed = False

        self.enemy_one_timer = self.time_limit
        self.enemy_two_timer = self.time_limit
        self.enemy_three_timer = self.time_limit
        self.enemy_four_timer = self.time_limit
        self.enemy_five_timer = self.time_limit
        self.enemy_six_timer = self.time_limit

        self.enemy_on_screen_count = 0
        self.enemy_killed_count = None
        self.room_location = None
        self.enemy_locations = []
        self.enemy_distances = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        self.map_location = None
        self.link_kill_count = None

        #Added exploration
        self.number_of_rupees = None
        self.number_of_keys = None

        self.check_starting_buffer = True
        self.starting_buffer = 500
        self.max_enemies = 6
        self.enemies = []

        self.enemy_index_dict = {0:"one",
                                 1:"two",
                                 2:"three",
                                 3:"four",
                                 4:"five",
                                 5:"six"}

        self.link_health_dict = {255:3, 
                                 127:2.5,
                                 254:2,
                                 126:1.5,
                                 253:1,
                                 125:0.5,
                                 0:0,}

        self.link_health = None

        metadata = {}
        rom_path = retro.data.get_romfile_path(game, inttype)
        metadata_path = retro.data.get_file_path(game, 'metadata.json', inttype)

        if state == retro.State.NONE:
            self.statename = None
        elif state == retro.State.DEFAULT:
            self.statename = None
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                if 'default_player_state' in metadata and self.players <= len(metadata['default_player_state']):
                    self.statename = metadata['default_player_state'][self.players - 1]
                elif 'default_state' in metadata:
                    self.statename = metadata['default_state']
                else:
                    self.statename = None
            except (IOError, json.JSONDecodeError):
                pass

        if self.statename:
            self.load_state(self.statename, inttype)

        self.data = retro.data.GameData()

        if info is None:
            info = 'data'

        if info.endswith('.json'):
            # assume it's a path
            info_path = info
        else:
            info_path = retro.data.get_file_path(game, info + '.json', inttype)

        if scenario is None:
            scenario = 'scenario'

        if scenario.endswith('.json'):
            # assume it's a path
            scenario_path = scenario
        else:
            scenario_path = retro.data.get_file_path(game, scenario + '.json', inttype)

        self.system = retro.get_romfile_system(rom_path)

        # We can't have more than one emulator per process. Before creating an
        # emulator, ensure that unused ones are garbage-collected
        gc.collect()
        self.em = retro.RetroEmulator(rom_path)
        self.em.configure_data(self.data)
        self.em.step()

        core = retro.get_system_info(self.system)
        self.buttons = core['buttons']
        self.num_buttons = len(self.buttons)

        try:
            assert self.data.load(info_path, scenario_path), 'Failed to load info (%s) or scenario (%s)' % (info_path, scenario_path)
        except Exception:
            del self.em
            raise

        self.button_combos = self.data.valid_actions()
        if use_restricted_actions == retro.Actions.DISCRETE:
            combos = 1
            for combo in self.button_combos:
                combos *= len(combo)
            self.action_space = gym.spaces.Discrete(combos ** players)
        elif use_restricted_actions == retro.Actions.MULTI_DISCRETE:
            self.action_space = gym.spaces.MultiDiscrete([len(combos) if gym_version >= (0, 9, 6) else (0, len(combos) - 1) for combos in self.button_combos] * players)
        else:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons * players)

        kwargs = {}
        if gym_version >= (0, 9, 6):
            kwargs['dtype'] = np.uint8
        
        if self._obs_type == retro.Observations.RAM:
            shape = self.get_ram().shape
        else:
            img = [self.get_screen(p) for p in range(players)]
            shape = img[0].shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, **kwargs)

        self.use_restricted_actions = use_restricted_actions
        self.movie = None
        self.movie_id = 0
        self.movie_path = None
        if record is True:
            self.auto_record()
        elif record is not False:
            self.auto_record(record)
        self.seed()
        if gym_version < (0, 9, 6):
            self._seed = self.seed
            self._step = self.step
            self._reset = self.reset
            self._render = self.render
            self._close = self.close

    def pre_proc(self, X):
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(x, (self.max_x, self.max_y))

        return x

    def _update_obs(self):
        if self._obs_type == retro.Observations.RAM:
            self.ram = self.get_ram()
            return self.ram
        elif self._obs_type == retro.Observations.IMAGE:
            self.img = self.get_screen()
            return self.img
        else:
            raise ValueError('Unrecognized observation type: {}'.format(self._obs_type))

    def action_to_array(self, a):
        actions = []
        for p in range(self.players):
            action = 0
            if self.use_restricted_actions == retro.Actions.DISCRETE:
                for combo in self.button_combos:
                    current = a % len(combo)
                    a //= len(combo)
                    action |= combo[current]
            elif self.use_restricted_actions == retro.Actions.MULTI_DISCRETE:
                ap = a[self.num_buttons * p:self.num_buttons * (p + 1)]
                for i in range(len(ap)):
                    buttons = self.button_combos[i]
                    action |= buttons[ap[i]]
            else:
                ap = a[self.num_buttons * p:self.num_buttons * (p + 1)]
                for i in range(len(ap)):
                    action |= int(ap[i]) << i
                if self.use_restricted_actions == retro.Actions.FILTERED:
                    action = self.data.filter_action(action)
            ap = np.zeros([self.num_buttons], np.uint8)
            for i in range(self.num_buttons):
                ap[i] = (action >> i) & 1
            actions.append(ap)
        return actions

    def step(self, a):
        if self.img is None and self.ram is None:
            raise RuntimeError('Please call env.reset() before env.step()')

        for p, ap in enumerate(self.action_to_array(a)):
            if self.movie:
                for i in range(self.num_buttons):
                    self.movie.set_key(i, ap[i], p)
            self.em.set_button_mask(ap, p)

        if self.movie:
            self.movie.step()
        self.em.step()
        self.data.update_ram()
        ob = self._update_obs()
        rew, done, info = self.compute_step()
        return ob, rew, bool(done), dict(info)

    def reset(self):
        if self.initial_state:
            self.em.set_state(self.initial_state)
        for p in range(self.players):
            self.em.set_button_mask(np.zeros([self.num_buttons], np.uint8), p)
        self.em.step()
        if self.movie_path is not None:
            rel_statename = os.path.splitext(os.path.basename(self.statename))[0]
            self.record_movie(os.path.join(self.movie_path, '%s-%s-%06d.bk2' % (self.gamename, rel_statename, self.movie_id)))
            self.movie_id += 1
        if self.movie:
            self.movie.step()
        self.data.reset()
        self.data.update_ram()
        return self._update_obs()

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer:
                self.viewer.close()
            return

        img = self.get_screen() if self.img is None else self.img
        if mode == "rgb_array":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif mode == "human":
            if self.viewer is None:
                from gym.envs.classic_control.rendering import SimpleImageViewer
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if hasattr(self, 'em'):
            del self.em

    def get_action_meanings(self):
        # actions = []
        # actions.append([self.buttons[i] for i in np.extract(action, np.arange(len(action)))])
        # print(self.buttons)
        # # return self.
        return self.unwrapped.buttons

    def get_action_meaning(self, act):
        actions = []
        for p, action in enumerate(self.action_to_array(act)):
            actions.append([self.buttons[i] for i in np.extract(action, np.arange(len(action)))])
        if self.players == 1:
            return actions[0]
        return actions

    def get_ram(self):
        blocks = []
        for offset in sorted(self.data.memory.blocks):
            arr = np.frombuffer(self.data.memory.blocks[offset], dtype=np.uint8)
            blocks.append(arr)
        return np.concatenate(blocks)

    def get_screen(self, player=0):
        img = self.em.get_screen()
        # cv2.imwrite(r"C:\Users\alith\Desktop\Files\school\thesis\PC2-RL-MUGEN\testing\test.jpg", img)
        x, y, w, h = self.data.crop_info(player)
        if not w or x + w > img.shape[1]:
            w = img.shape[1]
        else:
            w += x
        if not h or y + h > img.shape[0]:
            h = img.shape[0]
        else:
            h += y
        if x == 0 and y == 0 and w == img.shape[1] and h == img.shape[0]:
            # img = self.pre_proc(img)
            return img
        return img[y:h, x:w]

    def load_state(self, statename, inttype=retro.data.Integrations.DEFAULT):
        if not statename.endswith('.state'):
                statename += '.state'

        with gzip.open(retro.data.get_file_path(self.gamename, statename, inttype), 'rb') as fh:
            self.initial_state = fh.read()

        self.statename = statename

    def reset_game_state(self):
        # self.old_distance = None

        # self.enemy_one_x = None
        # self.enemy_one_y = None
        # self.enemy_two_x = None
        # self.enemy_two_y = None
        # self.enemy_three_x = None
        # self.enemy_three_y = None
        # self.enemy_four_x = None
        # self.enemy_four_y = None
        # self.enemy_five_x = None
        # self.enemy_five_y = None
        # self.enemy_six_x = None
        # self.enemy_six_y = None

        # self.enemy_one_exists = False
        # self.enemy_two_exists = False
        # self.enemy_three_exists = False
        # self.enemy_four_exists = False
        # self.enemy_five_exists = False
        # self.enemy_six_exists = False

        # self.enemy_one_killed = False
        # self.enemy_two_killed = False
        # self.enemy_three_killed = False
        # self.enemy_four_killed = False
        # self.enemy_five_killed = False
        # self.enemy_six_killed = False

        # self.enemy_on_screen_count = 0
        # self.enemy_killed_count = None
        # self.room_location = None
        # self.enemy_locations = []
        # self.enemy_distances = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        # self.map_location = None
        # self.link_kill_count = None

        # self.enemy_one_timer = self.time_limit
        # self.enemy_two_timer = self.time_limit
        # self.enemy_three_timer = self.time_limit
        # self.enemy_four_timer = self.time_limit
        # self.enemy_five_timer = self.time_limit
        # self.enemy_six_timer = self.time_limit

        self.enemies = []
        self.check_starting_buffer = True
        self.starting_buffer = 500
        self.map_location = None
        self.link_health = None

    def convert_link_health(self, full, partial):

        # Link's health does unexpected things
        assert full == 34 or full == 33 or full == 32
        assert (partial > 115 and partial < 260) or partial == 0

        converted_health = 0
        
        # At least 2 full hearts
        if full == 34:
            converted_health += 2
        
        # At least 1 full heart
        elif full == 33:
            converted_health += 1

        # No full hearts
        elif full == 32:
            converted_health += 0

        # 130 seems to be the sweet spot for
        # if Link has half a heart or not
        if partial > 130:
            converted_health += 1
        elif partial > 0 and partial < 130:
            converted_health += 0.5

        # Link is dead
        if partial == 0:
            converted_health = 0

        # Check that we get expected values
        assert converted_health <= 3 and converted_health >= 0
        return converted_health

    def convert_aob_health(self, value):
        str_value = str(value)

        values = []
        for i in range(6):

            new_value = str_value[i*2:(i+1)*2]
            if new_value.isnumeric():
                new_value = int(new_value)
            else:
                new_value = 0

            # All values in this aob
            # appear to be multiples of 10
            new_value //= 10

            # Droped items seem to take up the same
            # address as enemy health, but they start
            # with a value of 8
            if new_value >= 8:
                new_value = 0

            values.append(new_value)

        return values

    def compute_step(self):
        # Hijacking the reward function

        reward = 0
        done = False
        # enemies_on_screen = 0

        #Used in both
        map_location = self.data.lookup_all()['Map location']
        hearts_partial = self.data.lookup_all()['Partial heart']
        hearts_full = self.data.lookup_all()['Heart containers']
        
        
        #Combat
        mobs_with_health_aob = self.data.lookup_all()['mobs_with_health_aob']
        one_shotable_health_aob = self.data.lookup_all()['one_shotable_health_aob']


        #Exploration
        has_bow = self.data.lookup_all()['Has_Bow']
        number_of_keys = self.data.lookup_all()['Number_of_keys']
        number_of_rupees = self.data.lookup_all()['Number_of_rupees']

        pulse_1 = self.data.lookup_all()['Pulse_1']
        pulse_2 = self.data.lookup_all()['Pulse_2']


        ###############################################################################
        ## Combat Section                                                            ##
        ###############################################################################

        # Mobs with health only loads when enemies with more than 1 hit are on screen
        if mobs_with_health_aob > 0:
            converted_mobs_with_health_aob = self.convert_aob_health(mobs_with_health_aob)
            converted_enemy_health_aob = converted_mobs_with_health_aob
            # print("> 1 Hit")
        else:
            one_shotable_health_aob = str(one_shotable_health_aob)
            if len(str(one_shotable_health_aob)) %2 != 0:
                one_shotable_health_aob = one_shotable_health_aob[1:]
            converted_one_shotable_health_aob = self.convert_aob_health(one_shotable_health_aob)
            converted_enemy_health_aob = converted_one_shotable_health_aob
            # print("== 1 Hit")

        if self.map_location is None:
            self.map_location = map_location

        if self.link_health is None:
            self.link_health = self.convert_link_health(hearts_full, hearts_partial)


        #Exploration initalizing variables
        if self.number_of_rupees is None:
            self.number_of_rupees = number_of_rupees

        if self.number_of_keys is None:
            self.number_of_keys = number_of_keys

        # Create and get starting position
        if len(self.enemies) == 0:
            for i in range(self.max_enemies):
                enemy = Enemy()
                enemy.health = converted_enemy_health_aob[i]

                self.enemies.append(enemy)

        # Get current location
        for i in range(self.max_enemies):

            # Link did damage to the enemy
            if self.enemies[i].health > converted_enemy_health_aob[i]:
                reward += 0.5

            # Outside the if to cover the case of an enemy
            # potentially healing itself
            self.enemies[i].health = converted_enemy_health_aob[i]

        #Commented for exploration
        '''
        if all(health == 0 for health in converted_enemy_health_aob):
            done = True
            reward += 5
            print("Cleared")
        '''
        # Link has left the room, punish him
        '''
        if self.map_location != map_location:
            done = True
            reward -= 10
            print("Left room")
        '''
        new_health = self.convert_link_health(hearts_full, hearts_partial)

        # Link took damage
        if new_health < self.link_health:
            reward -= 1
            # print("Took damage")
        # Link picked up a heart
        elif new_health > self.link_health:
            reward += 1
            # print("Healed")
        # Link died
        #Changed value to 1 for less punishment
        if new_health == 0:
            reward -= 1
            done = True
            print("Died")

        self.link_health = new_health

        
        ###############################################################################
        ## Exploration Section (using health in combat section)                      ##
        ###############################################################################
        #Do not leave bow room (change later)
    
        if map_location == 55:
            done = True
            reward -= 50
            print("Left the dungeon")
        
        if map_location == 35:
            done = True
            reward -= 10
            print("Agent has left the 10th room!")

        #Bow reward
        if has_bow:
            reward += 10
            done = True
            print("Got the bow!")


        #Key reward
        if pulse_1 == 8 and self.number_of_keys != number_of_keys:
            self.number_of_keys = number_of_keys
            reward += 5
            print("Got a key!")

        #Rupee reward
        if pulse_2 == 1:
            rupee_reward = (number_of_rupees - self.number_of_rupees) * 0.1
            reward += rupee_reward
            self.number_of_rupees = number_of_rupees
            print("Collected Rupees")

        #Secret Reward
        if pulse_2 == 4:
            reward += 7
            print("Secret Found")

    
        if self.number_of_keys == 3:
            reward += 50 
            print("Found all keys in state")
            done = True

        if done:
            self.reset_game_state()

        if self.use_render:
            self.render()
        return reward, done, self.data.lookup_all()

























        # done = False
        # reward = 0
        # round_value = 2
        # max_link_health = 2.5

        # link_health_dict = {255:3, 
        #                     254:3,
        #                     127:2.5,
        #                     126:2.5,
        #                     253:2,
        #                     125:1.5,
        #                     252:1,
        #                     124:0.5,
        #                     0:0,
        #                     251:0}

        # link_x = self.data.lookup_all()['link_x']
        # link_y = self.data.lookup_all()['link_y']

        # map_location = self.data.lookup_all()['Map location']
        # game_data_kill_count = self.data.lookup_all()['killed_enemy_count']

        # enemy_one_x = self.data.lookup_all()['enemy_one_x']
        # enemy_one_y = self.data.lookup_all()['enemy_one_y']
        # enemy_one_loc = (enemy_one_x, enemy_one_y)

        # if self.enemy_one_last_pos != enemy_one_loc:
        #     self.enemy_one_last_pos = enemy_one_loc
        #     self.enemy_one_timer = self.time_limit

        # if self.enemy_one_exists and self.enemy_one_last_pos == enemy_one_loc:
        #     self.enemy_one_timer -= 1

        #     if self.enemy_one_timer <= 0:
        #         self.enemy_one_exists = False
        #         self.enemy_one_killed = True
        #         print("Enemy 1 force killed")
        #         game_data_kill_count += 1

        # enemy_two_x = self.data.lookup_all()['enemy_two_x']
        # enemy_two_y = self.data.lookup_all()['enemy_two_y']
        # enemy_two_loc = (enemy_two_x, enemy_two_y)

        # if self.enemy_two_last_pos != enemy_two_loc:
        #     self.enemy_two_last_pos = enemy_two_loc
        #     self.enemy_two_timer = self.time_limit

        # if self.enemy_two_exists and self.enemy_two_last_pos == enemy_two_loc:
        #     self.enemy_two_timer -= 1

        #     if self.enemy_two_timer <= 0:
        #         self.enemy_two_exists = False
        #         self.enemy_two_killed = True
        #         print("Enemy 2 force killed")
        #         game_data_kill_count += 1

        # enemy_three_x = self.data.lookup_all()['enemy_three_x']
        # enemy_three_y = self.data.lookup_all()['enemy_three_y']
        # enemy_three_loc = (enemy_three_x, enemy_three_y)

        # if self.enemy_three_last_pos != enemy_three_loc:
        #     self.enemy_three_last_pos = enemy_three_loc
        #     self.enemy_three_timer = self.time_limit

        # if self.enemy_three_exists and self.enemy_three_last_pos == enemy_three_loc:
        #     self.enemy_three_timer -= 1

        #     if self.enemy_three_timer <= 0:
        #         self.enemy_three_exists = False
        #         self.enemy_three_killed = True
        #         print("Enemy 3 force killed")
        #         game_data_kill_count += 1

        # enemy_four_x = self.data.lookup_all()['enemy_four_x']
        # enemy_four_y = self.data.lookup_all()['enemy_four_y']
        # enemy_four_loc = (enemy_four_x, enemy_four_y)

        # if self.enemy_four_last_pos != enemy_four_loc:
        #     self.enemy_four_last_pos = enemy_four_loc
        #     self.enemy_four_timer = self.time_limit

        # if self.enemy_four_exists and self.enemy_four_last_pos == enemy_four_loc:
        #     self.enemy_four_timer -= 1

        #     if self.enemy_four_timer <= 0:
        #         self.enemy_four_exists = False
        #         self.enemy_four_killed = True
        #         print("Enemy 4 force killed")
        #         game_data_kill_count += 1

        # enemy_five_x = self.data.lookup_all()['enemy_five_x']
        # enemy_five_y = self.data.lookup_all()['enemy_five_y']
        # enemy_five_loc = (enemy_five_x, enemy_five_y)

        # if self.enemy_five_last_pos != enemy_five_loc:
        #     self.enemy_five_last_pos = enemy_five_loc
        #     self.enemy_five_timer = self.time_limit

        # if self.enemy_five_exists and self.enemy_five_last_pos == enemy_five_loc:
        #     self.enemy_five_timer -= 1

        #     if self.enemy_five_timer <= 0:
        #         self.enemy_five_exists = False
        #         self.enemy_five_killed = True
        #         print("Enemy 5 force killed")
        #         game_data_kill_count += 1

        # enemy_six_x = self.data.lookup_all()['enemy_six_x']
        # enemy_six_y = self.data.lookup_all()['enemy_six_y']
        # enemy_six_loc = (enemy_six_x, enemy_six_y)

        # if self.enemy_six_last_pos != enemy_six_loc:
        #     self.enemy_six_last_pos = enemy_six_loc
        #     self.enemy_six_timer = self.time_limit

        # if self.enemy_six_exists and self.enemy_six_last_pos == enemy_six_loc:
        #     self.enemy_six_timer -= 1

        #     if self.enemy_six_timer <= 0:
        #         self.enemy_six_exists = False
        #         self.enemy_six_killed = True
        #         print("Enemy 6 force killed")
        #         game_data_kill_count += 1

        # key_pulse = self.data.lookup_all()['key_pulse']
        # link_health = self.data.lookup_all()['link_health']

        # try:
        #     assert link_health in link_health_dict.keys()
        # except:
        #     print(link_health)
        #     assert link_health in link_health_dict.keys()

        # # Audio key noise
        # # if key_pulse == 2:
        # #     done = True
        # #     reward += 5
        # #     self.reset_game_state()
        # #     return reward, done, self.data.lookup_all()

        # if len(self.enemy_locations) == 0:
        #     self.enemy_locations.append(enemy_one_loc)
        #     self.enemy_locations.append(enemy_two_loc)
        #     self.enemy_locations.append(enemy_three_loc)
        #     self.enemy_locations.append(enemy_four_loc)
        #     self.enemy_locations.append(enemy_five_loc)
        #     self.enemy_locations.append(enemy_six_loc)

        # if self.enemy_locations[0] != enemy_one_loc and not self.enemy_one_exists and not self.enemy_one_killed:
        #     self.enemy_one_exists = True
        # if self.enemy_locations[1] != enemy_two_loc and not self.enemy_two_exists and not self.enemy_two_killed:
        #     self.enemy_two_exists = True
        # if self.enemy_locations[2] != enemy_three_loc and not self.enemy_three_exists and not self.enemy_three_killed:
        #     self.enemy_three_exists = True
        # if self.enemy_locations[3] != enemy_four_loc and not self.enemy_four_exists and not self.enemy_four_killed:
        #     self.enemy_four_exists = True
        # if self.enemy_locations[4] != enemy_five_loc and not self.enemy_five_exists and not self.enemy_five_killed:
        #     self.enemy_five_exists = True
        # if self.enemy_locations[5] != enemy_six_loc and not self.enemy_six_exists and not self.enemy_six_killed:
        #     self.enemy_six_exists = True

        # def compute_distance(enemy_loc):
        #     return math.sqrt((link_x - enemy_loc[0])**2 + (link_y - enemy_loc[1])**2)

        # self.enemy_on_screen_count = 0
        # if self.enemy_one_exists:
        #     self.enemy_on_screen_count += 1
        #     self.enemy_distances[0] = compute_distance(enemy_one_loc)

        # if self.enemy_two_exists:
        #     self.enemy_on_screen_count += 1
        #     self.enemy_distances[1] = compute_distance(enemy_two_loc)

        # if self.enemy_three_exists:
        #     self.enemy_on_screen_count += 1
        #     self.enemy_distances[2] = compute_distance(enemy_three_loc)

        # if self.enemy_four_exists:
        #     self.enemy_on_screen_count += 1
        #     self.enemy_distances[3] = compute_distance(enemy_four_loc)

        # if self.enemy_five_exists:
        #     self.enemy_on_screen_count += 1
        #     self.enemy_distances[4] = compute_distance(enemy_five_loc)

        # if self.enemy_six_exists:
        #     self.enemy_on_screen_count += 1
        #     self.enemy_distances[5] = compute_distance(enemy_six_loc)

        # # print(self.enemy_on_screen_count)

        # # closest_enemy = np.argmin(self.enemy_distances)
        # # reward += 0.00001 * (1 / (1 + self.enemy_distances[closest_enemy]))

        # # if self.enemy_distances[closest_enemy] > 150:
        # #     print("Out of range of enemy")
        # #     done = True
        # #     self.reset_game_state()
        # #     reward -= 5
        # #     return reward, done, self.data.lookup_all()

        # if self.link_kill_count is None:
        #     self.link_kill_count = game_data_kill_count

        # if self.previous_link_health is None:
        #     self.previous_link_health = link_health_dict.get(link_health)

        # if link_health_dict.get(link_health) < self.previous_link_health:
        #     reward -= 1
        #     self.previous_link_health = link_health_dict.get(link_health)

        # # Link picked up a heart
        # if link_health_dict.get(link_health) > self.previous_link_health:
        #     reward += 1
        #     self.previous_link_health = link_health_dict.get(link_health)

        # if link_health_dict.get(link_health) == 0:
        #     # reward -= 10
        #     done = True
        #     self.reset_game_state()
        #     return reward, done, self.data.lookup_all()
        
        # with open(PUT_PATH_HERE, "a") as f:
        #     f.write(chunk)
        # #     myfile.write("Enemy 2: " + str(self.enemy_two_exists) + " " + str(self.enemy_two_killed) + " " + str(self.enemy_two_timer) + "\n")
        # #     myfile.write("Enemy 3: " + str(self.enemy_three_exists) + " " + str(self.enemy_three_killed) + " " + str(self.enemy_three_timer) + "\n")
        # #     myfile.write("Enemy 4: " + str(self.enemy_four_exists) + " " + str(self.enemy_four_killed) + " " + str(self.enemy_four_timer) + "\n")
        # #     myfile.write("Enemy 5: " + str(self.enemy_five_exists) + " " + str(self.enemy_five_killed) + " " + str(self.enemy_five_timer) + "\n")
        # #     myfile.write("Enemy 6: " + str(self.enemy_six_exists) + " " + str(self.enemy_six_killed) + " " + str(self.enemy_six_timer) + "\n")
        # #     myfile.write("Game Data Kill Count:" + str( game_data_kill_count) + "\n")
        # #     myfile.write("Self.Link_Kill_Count:" + str( self.link_kill_count) + "\n")
        # #     myfile.write("Enemy Distances     :" + str( self.enemy_distances) + "\n")
        # #     myfile.write("\n")

        # if game_data_kill_count > self.link_kill_count:

        #     assert game_data_kill_count - self.link_kill_count != 0

        #     self.enemy_on_screen_count -= game_data_kill_count - self.link_kill_count
        #     reward += game_data_kill_count - self.link_kill_count
        #     # print("Link killed something")

        #     # print(self.enemy_on_screen_count)
        #     # print(self.data.lookup_all()['killed_enemy_count'])
        #     # print(self.link_kill_count)

        #     # Link killed everything on screen
        #     if self.enemy_on_screen_count <= 0:
        #         # print("Link Cleared the room")
        #         done = True
        #         reward += 5
        #         self.reset_game_state()
        #         print("Room Cleared")
        #         print()
        #         return reward, done, self.data.lookup_all()
        #     else:
        #         cloest_enemies = np.array(self.enemy_distances).argsort()
        #         now_dead_enemies = cloest_enemies[:game_data_kill_count - self.link_kill_count]
        #         self.link_kill_count = game_data_kill_count

        #         for enemy in now_dead_enemies:
        #             if enemy == 0:
        #                 self.enemy_one_killed = True
        #                 self.enemy_one_exists = False
        #                 print("Enemy 1 died")
        #             if enemy == 1:
        #                 self.enemy_two_killed = True
        #                 self.enemy_two_exists = False
        #                 print("Enemy 2 died")
        #             if enemy == 2:
        #                 self.enemy_three_killed = True
        #                 self.enemy_three_exists = False
        #                 print("Enemy 3 died")
        #             if enemy == 3:
        #                 self.enemy_four_killed = True
        #                 self.enemy_four_exists = False
        #                 print("Enemy 4 died")
        #             if enemy == 4:
        #                 self.enemy_five_killed = True
        #                 self.enemy_five_exists = False
        #                 print("Enemy 5 died")
        #             if enemy == 5:
        #                 self.enemy_six_killed = True
        #                 self.enemy_six_exists = False
        #                 print("Enemy 6 died")
        #             self.enemy_distances[enemy] = np.inf
        #         # print(now_dead_enemies)
        #         # print(self.enemy_distances)

        # if self.map_location is None:
        #     self.map_location = map_location

        # if game_data_kill_count < self.link_kill_count:
        #     assert self.link_kill_count != game_data_kill_count
        #     self.link_kill_count = game_data_kill_count

        # # Stop Link from leaving the room
        # if self.map_location != map_location:
        #     done = True
        #     self.reset_game_state()
        #     reward = -10
            # print("Link left the room")

        # boss_x = self.data.lookup_all()['boss_x?']
        # boss_y = self.data.lookup_all()['boss_y?']
        # link_bomb_count = self.data.lookup_all()['bomb_count']
        # distance = math.sqrt((link_x - boss_x)**2 + (link_y - boss_y)**2)

        # if self.old_distance == None:
        #     self.old_distance = distance
        # else:
        #     if distance < self.old_distance:
        #         reward += 0.001 * (1 / (1 + distance))
        #         # self.old_distance = distance

        #     if distance > self.old_distance:
        #         reward += 0.001 * ((1/distance if distance != 0 else 1) - 1)
        #     self.old_distance = distance

        # if self.players > 1:
        #     reward = [self.data.current_reward(p) for p in range(self.players)]
        # else:

        #     if self.data.lookup_all()['Map location'] in boss_room_locations:

        #         if self.data.lookup_all()['Map location'] == boss_room_locations[2] or self.data.lookup_all()['Map location'] == boss_room_locations[2]:
        #             limited_resource_end = True

        #         if self.previous_boss_health == None:
        #             self.previous_boss_health = self.data.lookup_all()['boss_health'] / boss_health_divisor
                
        #         if self.previous_link_health == None:
        #             self.previous_link_health = link_health_dict.get(self.data.lookup_all()['link_health'])

        #         if self.previous_boss_health > self.data.lookup_all()['boss_health'] / boss_health_divisor:
        #             reward += 1
        #             self.previous_boss_health = self.data.lookup_all()['boss_health'] / boss_health_divisor
                
        #         if self.previous_link_health > link_health_dict.get(self.data.lookup_all()['link_health']):
        #             reward -= self.previous_link_health - link_health_dict.get(self.data.lookup_all()['link_health'])
        #             self.previous_link_health = link_health_dict.get(self.data.lookup_all()['link_health'])
        #             # print(reward)

        #         if link_health_dict.get(self.data.lookup_all()['link_health']) == 0:
        #             reward -= round_value ** (1 - (1 - (self.data.lookup_all()['boss_health'] / boss_health_divisor / max_boss_health)))
        #             done = True
        #             self.old_distance = None
        #             self.previous_boss_health = None
        #             self.previous_link_health = None
        #             print("Boss -> Link")

        #         if self.data.lookup_all()['boss_health'] / boss_health_divisor == 0:
        #             #Exp_decay
        #             reward += round_value ** (1 - (1 - (link_health_dict.get(self.data.lookup_all()['link_health']) / max_link_health)))

        #             print("Link -> Boss")
        #             self.time = 0
        #             done = True
        #             self.old_distance = None
        #             self.previous_boss_health = None
        #             self.previous_link_health = None

        #         if distance > 120:
        #             done = True
        #             self.old_distance = None
        #             self.previous_boss_health = None
        #             self.previous_link_health = None
        #             reward -= 10
        #             print("Link out of range of boss")

        #         if limited_resource_end:
        #             if self.data.lookup_all()['Map location'] == boss_room_locations[2] or self.data.lookup_all()['Map location'] == boss_room_locations[2]:
        #                 if link_bomb_count == 0:
        #                     done = True
        #                     self.old_distance = None
        #                     self.previous_boss_health = None
        #                     self.previous_link_health = None

        #     else:
        #         done = True
        #         self.old_distance = None
        #         self.previous_boss_health = None
        #         self.previous_link_health = None
        #         print("Link left the boss room")
        #         reward -= 10

        # if self.use_render:
        #     self.render()
        # return reward, done, self.data.lookup_all()

    def record_movie(self, path):
        self.movie = retro.Movie(path, True, self.players)
        self.movie.configure(self.gamename, self.em)
        if self.initial_state:
            self.movie.set_state(self.initial_state)

    def stop_record(self):
        self.movie_path = None
        self.movie_id = 0
        if self.movie:
            self.movie.close()
            self.movie = None

    def auto_record(self, path=None):
        if not path:
            path = os.getcwd()
        self.movie_path = path
