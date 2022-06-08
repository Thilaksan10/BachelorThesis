import os
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU, Flatten
from mcts import TreeNode, MCTS

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *(1,) + input_shape))
        self.state_value_memory = np.zeros(self.mem_size)
        self.new_state_memory = np.zeros((self.mem_size, *(1,) + input_shape))
        self.new_state_value_memory = np.zeros(self.mem_size)
        self.action_memory = np.zeros(self.mem_size)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, state_value, action, reward, state_, state_value_, done):
        index = self.mem_cntr % self.mem_size
        # print(state.shape)
        # print(tf.convert_to_tensor(state).shape)
        self.state_memory[index] = state
        self.state_value_memory[index] = state_value
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.new_state_value_memory[index] = state_value_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size-1)

        states = self.state_memory[batch]
        states_values = self.state_value_memory[batch]
        new_states = self.new_state_memory[batch]
        new_states_values = self.new_state_value_memory[batch]
        reward = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, states_values, actions, reward, new_states, new_states_values, terminal

class PolicyNetwork(keras.Model):
    def __init__(self, layer_dims, n_actions, input_shape, n_tasks, m_sets, name='policy', chkpt_dir='tmp/policy'):
        super(PolicyNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_dir = f'{chkpt_dir}-{n_tasks}x{m_sets}-multiple_resources'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.model_input_shape = (1,) + input_shape
        print(self.model_input_shape)

        if len(layer_dims) == 0:
            self.model = tf.keras.models.load_model('Supervised-v6')
        else:
            self.model = tf.keras.Sequential()
            for index in range(3):
                if index == 0:
                    self.model.add(Conv2D(64, (3,3), (1,1), input_shape=self.model_input_shape)) 
                else:
                    self.model.add(Conv2D(64, (1,1), (1,1))) 
                self.model.add(BatchNormalization())
                self.model.add(ReLU())
            self.model.add(Flatten())
            self.model.add(Dense(2048, activation='relu'))
            self.model.add(Dense(1024, activation='relu'))
            self.model.add(Dense(n_actions, activation='softmax'))

    def call(self, state):
        value = state
        for layer in self.model.layers:
            value = layer(value)
        return value
        
class Agent:
    def __init__(self, input_shape, n_tasks, m_sets, alpha=0.0001, gamma=0.99, policy_layer_dims=[], mem_size=10000, n_actions=5, load_memory=0) -> None:
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.batch_size = 64
        self.m_sets = m_sets
        self.n_tasks = n_tasks

        self.policy_network = PolicyNetwork(layer_dims=policy_layer_dims, n_actions=n_actions, input_shape=input_shape, n_tasks=n_tasks, m_sets=m_sets)

        self.policy_network.compile(optimizer=Adam(learning_rate=alpha))
        self.mcts = MCTS()
        if load_memory:
            with open(f'replay_buffer-{n_tasks}x{m_sets}.pkl', 'rb') as f:
                self.memory = pickle.load(f)
                print(f'Current Memory Counter is {self.memory.mem_cntr}')
        else:
            self.memory = ReplayBuffer(max_size=mem_size, input_shape=input_shape, n_actions=n_actions)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([[observation]])
        probs = self.policy_network(state)
        # print(probs)
        action_probabilites = tfp.distributions.Categorical(probs=probs)
       
        action = action_probabilites.sample()
        self.action = action
        # print(action)
        
        return self.action.numpy()[0]

    def choose_action_eval(self, observation):
        state = tf.convert_to_tensor([[observation]])
        probs = self.policy_network(state)

        self.action = tf.math.argmax(probs, axis=1)
        # print(self.action)
        return self.action.numpy()[0]

    def save_models(self):
        print('... save models ...')
        self.policy_network.save_weights(self.policy_network.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.policy_network.load_weights(self.policy_network.checkpoint_file)

    def remember(self, state, state_value, action, reward, new_state, new_state_value, done):
        self.memory.store_transition(state, state_value, action, reward, new_state, new_state_value, done)
        if done:
            with open(f'replay_buffer-{self.n_tasks}x{self.m_sets}.pkl', 'wb') as f:
                pickle.dump(self.memory, f, pickle.HIGHEST_PROTOCOL)

    def learn(self, state, reward, state_, done):
        if self.memory.mem_cntr < self.batch_size-1:
            if self.mcts.root:
                if state.to_string() == self.mcts.root.scheduler.to_string():
                    _ = self.mcts.search(state, self.mcts.root)
                    state_value = tf.math.tanh([float(self.mcts.root.score)])
                    # print('Yeah')
                else:
                    _ = self.mcts.search(state)
                    state_value = tf.math.tanh([float(self.mcts.root.score)])
            else:
                _ = self.mcts.search(state)
                state_value = tf.math.tanh([float(self.mcts.root.score)])

            if state_.to_string() not in self.mcts.root.children:
                if state.to_string() != state_.to_string():
                    _ = self.mcts.search(state_)
                    state_value_ = tf.math.tanh([float(self.mcts.root.score)])
                else:
                    state_value_ = tf.math.tanh([float(-1)])
            else:
                _ = self.mcts.search(state_, self.mcts.root.children[state_.to_string()])
                state_value_ = tf.math.tanh([float(self.mcts.root.score)])

            if done:
                self.mcts.root = None

            return state_value, state_value_

        states, states_values, actions, rewards, new_states, new_states_values, dones = self.memory.sample_buffer(self.batch_size)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        rewards = tf.concat(axis=-1, values=[rewards, reward])

        states = tf.convert_to_tensor(states, dtype=tf.float32)

        states_values = tf.convert_to_tensor(states_values, dtype=tf.float32)

        new_states_values = tf.convert_to_tensor(new_states_values, dtype=tf.float32)

        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        action = tf.convert_to_tensor(self.action, dtype=tf.int32)
        actions = tf.concat(axis=-1, values=[actions, action])
        # print(f'Reward: {reward}')

        with tf.GradientTape() as tape:
            
            if self.mcts.root:
                if state.to_string() == self.mcts.root.scheduler.to_string():
                    _ = self.mcts.search(state, self.mcts.root)
                    state_value = tf.math.tanh([float(self.mcts.root.score)])
                    # print('Yeah')
                else:
                    _ = self.mcts.search(state)
                    state_value = tf.math.tanh([float(self.mcts.root.score)])
            else:
                _ = self.mcts.search(state)
                state_value = tf.math.tanh([float(self.mcts.root.score)])
                # print('no')
            # print(states_values.shape)
            # print(state_value.shape)
            states_values = tf.concat(axis=-1, values=[states_values, state_value])
            # print(state.to_string())
            # print(state_.to_string())
            # print(len(self.mcts.root.children))
            # state_value = tf.math.tanh([float(self.mcts.root.score)])
            # print(f'Current State')
            # print(f'Current State Value: {state_value}')
            # print(float(self.mcts.root.score))
           
            if state_.to_string() not in self.mcts.root.children:
                if state.to_string() != state_.to_string():
                    _ = self.mcts.search(state_)
                    state_value_ = tf.math.tanh([float(self.mcts.root.score)])
                else:
                    state_value_ = tf.math.tanh([float(-1)])
            else:
                _ = self.mcts.search(state_, self.mcts.root.children[state_.to_string()])
                state_value_ = tf.math.tanh([float(self.mcts.root.score)])
            # print(new_states_values.shape)
            # print(state_value_.shape)
            new_states_values = tf.concat(axis=-1, values=[new_states_values, state_value_])
            # print(f'next State')
            # print(f'Next State Value: {state_value_}')
            # print(float(self.mcts.root.score))
            if done:
                self.mcts.root = None
            # input()
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            state = tf.convert_to_tensor([[state.to_array()]], dtype=tf.float32)
            states = tf.concat(axis=0, values=[states, state])

            # print(dones)
            done = tf.convert_to_tensor([int(done)], dtype=tf.float32)
            # print(done)
            dones = tf.concat(axis=-1, values=[dones, done])
            probs = self.policy_network(states)

            target = []
            for j in range(self.batch_size):
                actions_probs = tfp.distributions.Categorical(probs=probs[j])
            
                log_prob = actions_probs.log_prob(actions[j])
                # print(f'Log Prob: {log_prob}')
                delta = rewards[j] + self.gamma * new_states_values[j] * (1-dones[j]) - states_values[j]
                # print(f'Delta: {delta}')
                loss = -log_prob * delta
                target.append(loss)
            
            policy_loss = tf.reduce_mean(tf.convert_to_tensor(target))
            # print(f'Loss {policy_loss}')
            # print('----------------------')
            # input()
        gradient = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        
        self.policy_network.optimizer.apply_gradients(zip(gradient, self.policy_network.trainable_variables))
        return state_value, state_value_
