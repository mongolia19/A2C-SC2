from functools import partial
import numpy as np
import tensorflow as tf
import gym
import pandas as pd
from pysc2.lib import features
from common.preprocess import ObsProcesser, ActionProcesser, FEATURE_KEYS
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType
from pysc2.env import sc2_env
import math
MAP_LEN = 32
OUTPUT_GRAPH = False
MAX_EPISODE = 5000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 500   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 5e-4   # learning rate for actor
LR_C = 10e-4     # learning rate for critic
map_name = "CollectMineralShards"
step_mul = 8
resolution = MAP_LEN
visualize = True
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
M_PI = 3.1415926535
DIM_DIRECTION = 8
up_left = 0
up = 1
up_right = 2
right = 3
right_down = 4
down = 5
left_down = 6
left = 7
DEBUG = False
obsProcesser = ObsProcesser()
action_processer = ActionProcesser(MAP_LEN)

def get_friend_group_x_y(player_selected):
    selected_y, selected_x = (player_selected == _PLAYER_FRIENDLY).nonzero()
    if len(selected_x) == 0 or len(selected_y) == 0:
        return -1, -1
    mean_x = np.mean(selected_x)
    mean_y = np.mean(selected_y)
    return int(mean_x), int(mean_y)

def distanceNormalized(d):
    SIGHT_RANGE = 256
    if d < 0:
        res = -9999
    elif d > SIGHT_RANGE:
        res = 0.05
    else:
        res = 1.0 - 0.95 / SIGHT_RANGE * d
    return res

def calulate_distance_in_phase_simple(self_x, self_y, enemy_x, enemy_y):
    vector = np.zeros(8)
    vector[:] = 10000
    phase_dim = -1
    delta_x = float(enemy_x - self_x)
    delta_y = float(enemy_y - self_y)
    distance = math.sqrt(delta_y ** 2 + delta_x ** 2)
    if DEBUG: print("enemy at ")
    # phase_dim = Direction2Index(delta_x, delta_y)
    if delta_x > 0 and delta_y > 0:
        phase_dim = right_down
        if DEBUG: print("down right")
    elif delta_x == 0 and delta_y > 0:
        phase_dim = down
        if DEBUG: print("down down")
    elif delta_x < 0 and delta_y > 0:
        phase_dim = left_down
        if DEBUG: print("down left")
    elif delta_x < 0 and delta_y == 0:
        phase_dim = left
        if DEBUG: print("left")
    elif delta_x < 0 and delta_y < 0:
        phase_dim = up_left
        if DEBUG: print("up left")
    elif delta_x == 0 and delta_y < 0:
        phase_dim = up
        if DEBUG: print("up up")
    elif delta_x > 0 and delta_y < 0:
        phase_dim = up_right
        if DEBUG: print("up right")
    elif delta_x >= 0 and delta_y == 0:
        phase_dim = right
        if DEBUG: print("right")
    if phase_dim == -1:
        print('maybe wrong in  calulate_distance_in_phase_simple')
        return vector
    else:
        vector[phase_dim] = distance
        return vector

def calculate_enemy_distance_distribution(self_x, self_y, enemy_x_list, enemy_y_list):
    total_vector = list()
    if DEBUG: print("calculate_enemy_distance_distribution")
    for index, one in enumerate(enemy_x_list):
        vec_tmp = calulate_distance_in_phase_simple(self_x, self_y, one, enemy_y_list[index])
        vec_tmp = [distanceNormalized(dist) for dist in vec_tmp]
        total_vector.append((vec_tmp, np.linalg.norm(vec_tmp)))
    total_vector = sorted(total_vector, key=lambda x: x[1])
    return total_vector[0][0]

def distance_reward(distance_2_target_obs, last_distance_2_target_obs):
    if max(distance_2_target_obs[0])> max(last_distance_2_target_obs[0]):
        return 0.01
    else:
        return -0.01

def get_hand_crafted_feature(obs):
    # obs = obs[0]
    _PLAYER_HIT = features.SCREEN_FEATURES.unit_hit_points.index
    # player_hit_points = obs["screen"][_PLAYER_HIT]
    _PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
    player_relative = obs["player_relative_screen"][0]
    _PLAYER_SELECTED = features.SCREEN_FEATURES.selected.index
    # player_selected = obs["player_relative_screen"][_PLAYER_SELECTED]
    selected_x, selected_y = get_friend_group_x_y(player_relative)
    enemy_y, enemy_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
    friend_y, friend_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
    if len(friend_x) != len(friend_y):
        print("friend_x num != friend_y num")
    self_pos = np.zeros(2)
    self_pos[0] = selected_x
    self_pos[1] = selected_y
    vector_enemy_dist_8_dim = calculate_enemy_distance_distribution(selected_x, selected_y,
                                                                    enemy_x, enemy_y)
    vector_enemy_dist_8_dim = np.expand_dims(vector_enemy_dist_8_dim, axis=0)
    return self_pos, vector_enemy_dist_8_dim

def prepro_sc(obs):
    return get_hand_crafted_feature(obs)

def make_sc2env(**kwargs):
    env = sc2_env.SC2Env(**kwargs)
    # env = available_actions_printer.AvailableActionsPrinter(env)
    return env
def merge_target_with_position(target, x, y, ssize):
    mov_step = 2
    # 0 up left ；1 up； 2 up right； 3 left； 4 right；5 left down 6 down 7 right down
    target_x = x
    target_y = y
    # direction_vector = Index2Direction(target)
    # target_x = x + direction_vector[0] * mov_step
    # target_y = y + direction_vector[1] * mov_step
    # target_x = round(target_x)
    # target_y = round(target_y)
    if target == up_left:
      target_x = x - mov_step
      target_y = y - mov_step
      # if DEBUG: print("up left ↖")
    elif target == up:
      target_x = x
      target_y = y - mov_step
      if DEBUG: print("up ↑")
    elif target == up_right:
      target_x = x + mov_step
      target_y = y - mov_step
      # if DEBUG: print("up right ↗")
    elif target == left:
      target_x = x - mov_step
      target_y = y
      if DEBUG: print("left ←")
    elif target == right:
      target_x = x + mov_step
      target_y = y
      if DEBUG: print("right →")
    elif target == left_down:
      target_x = x - mov_step
      target_y = y + mov_step
      # if DEBUG: print("left down ↙")
    elif target == down:
      target_x = x
      target_y = y + mov_step
      if DEBUG: print("down ↓")
    elif target == right_down:
      target_x = x + mov_step
      target_y = y + mov_step
      # if DEBUG: print("down right ↘")
    if target_y>=ssize:
      target_y = ssize - 1
    if target_x>=ssize:
      target_x = ssize - 1
    if target_x<0:
      target_x = 0
    if target_y<0:
      target_y = 0
    return target_x, target_y
def action_convertor(indic, x, y, size):
    return merge_target_with_position(indic, x, y, size)
class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "action")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
    def learn(self, s, a, td):
        # s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v
    def choose_action(self, s):
        # s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
    def learn(self, s, r, s_):
        # s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error
# action有两个，即向左或向右移动小车
# state是四维
env_args = dict(
    map_name=map_name,
    step_mul=step_mul,
    game_steps_per_episode=0,
    screen_size_px=(resolution,) * 2,
    minimap_size_px=(resolution,) * 2,
    visualize=visualize
)
# env = gym.make('CartPole-v0')
# env.seed(1)  # reproducible
# env = env.unwrapped
env = partial(make_sc2env, **env_args)()
# N_F = env.observation_space.shape[0]
# N_A = env.action_space.n
N_F = 8
N_A = 8
sess = tf.Session()
actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)
sess.run(tf.global_variables_initializer())
res = []
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    s = obsProcesser.process(s)
    self_pos, s = prepro_sc(s)
    last_self_pos = self_pos
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()
        a = actor.choose_action(s)
        t_y, t_x = action_convertor(a, self_pos[0], self_pos[1], MAP_LEN)
        # 在实际操作之前，先执行选中动作
        selection_action_id = np.array(7)[np.newaxis, ...]
        action_param = np.expand_dims(np.array(0)[np.newaxis, ...], axis=0)
        actions_pp = action_processer.process(selection_action_id, action_param)
        obs_raw = env.step(actions_pp)
        # latest_obs = self.obs_processer.process(obs_raw)
        # 选中完成后，执行实际操作
        spatial_action_2ds = np.array([np.array([t_x, t_y], dtype=np.int64)])
        actual_action_ids = np.array(331)[np.newaxis, ...]
        actions_pp = action_processer.process(actual_action_ids, spatial_action_2ds)
        # s_, r, done, info = env.step(a)
        s_ = env.step(actions_pp)
        r = s_[0].reward
        done = s_[0].last()
        s_ = obsProcesser.process(s_)
        self_pos, s_ = prepro_sc(s_)
        dis_r = distance_reward(s_, s)
        # r = r + dis_r
        if done:
            r = -0
        track_r.append(r)
        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
        s = s_
        t += 1
        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            res.append([i_episode, running_reward])
            break
pd.DataFrame(res,columns=['episode','a2c_reward']).to_csv('../a2c_reward.csv')
