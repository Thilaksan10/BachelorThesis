from math import gamma
import os
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.optimizers import Adam
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

    # store given values in replay buffer
    def store_transition(self, state, state_value, action, reward, state_, state_value_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.state_value_memory[index] = state_value
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.new_state_value_memory[index] = state_value_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    # get a random sample in batch size
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
    def __init__(self, cnn_layers, n_actions, input_shape, n_tasks, m_sets, name='policy', chkpt_dir='tmp/policy'):
        super(PolicyNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_dir = f'{chkpt_dir}-{n_tasks}x{m_sets}'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.model_input_shape = (1,) + input_shape
        # init neural net
        self.model = tf.keras.Sequential()
        # add cnn_layers
        for index in range(cnn_layers):
            if index == 0:
                self.model.add(Conv2D(64, (3,3), (1,1), input_shape=self.model_input_shape)) 
            else:
                self.model.add(Conv2D(64, (1,1), (1,1))) 
            self.model.add(BatchNormalization())
            self.model.add(ReLU())
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        # add output layer
        self.model.add(Dense(n_actions, activation='softmax'))

    def call(self, state):
        value = state
        for layer in self.model.layers:
            value = layer(value)
        return value
        
class Agent:
    def __init__(self, input_shape, n_tasks, m_sets, alpha=0.0001, gamma=0.99, cnn_layers=5, mem_size=10000, n_actions=5, load_memory=0) -> None:
        # init decay rate
        self.gamma = gamma

        self.n_actions = n_actions
        self.action = None

        # init batch size
        self.batch_size = 64

        self.m_sets = m_sets
        self.n_tasks = n_tasks

        # init learning rate
        self.alpha = alpha

        # init policy network
        self.policy_network = PolicyNetwork(cnn_layers=cnn_layers, n_actions=n_actions, input_shape=input_shape, n_tasks=n_tasks, m_sets=m_sets)
        # compile with Adam optimizer
        self.policy_network.compile(optimizer=Adam(learning_rate=alpha, clipnorm=1.))
        # init MCTS
        self.mcts = MCTS()

        # load stored replay buffer
        if load_memory:
            with open(f'replay_buffer-{n_tasks}x{m_sets}.pkl', 'rb') as f:
                self.memory = pickle.load(f)
                print(f'Current Memory Counter is {self.memory.mem_cntr}')
        # init new replay buffer
        else:
            self.memory = ReplayBuffer(max_size=mem_size, input_shape=input_shape, n_actions=n_actions)

    # choose action heuristically
    def choose_action(self, observation):
        state = tf.convert_to_tensor([[observation]])
        probs = self.policy_network(state)
        
        action_probabilites = tfp.distributions.Categorical(probs=probs)
       
        action = action_probabilites.sample()
        self.action = action
        
        return self.action.numpy()[0]

    # choose action with higest probability
    def choose_action_eval(self, observation):
        state = tf.convert_to_tensor([[observation]])
        probs = self.policy_network(state)
        
        self.action = tf.math.argmax(probs, axis=1)
        
        return self.action.numpy()[0]

    # save model
    def save_models(self):
        print('... save models ...')
        self.policy_network.save_weights(self.policy_network.checkpoint_file)

    # load model
    def load_models(self):
        print('... loading models ...')
        self.policy_network.load_weights(self.policy_network.checkpoint_file)

    # store given values in replay buffer
    def remember(self, state, state_value, action, reward, new_state, new_state_value, done):
        self.memory.store_transition(state, state_value, action, reward, new_state, new_state_value, done)
        # save replay buffer in seperate file after end of every episode
        if done:
            with open(f'replay_buffer-{self.n_tasks}x{self.m_sets}.pkl', 'wb') as f:
                pickle.dump(self.memory, f, pickle.HIGHEST_PROTOCOL)

    # train agent 
    def learn(self, state, reward, state_, done):
        # not enough steps played to generate batch
        if self.memory.mem_cntr < self.batch_size-1:
            # calculate value of state
            # is not first step in episode
            if self.mcts.root:
                # state is root of tree 
                if state.to_string() == self.mcts.root.scheduler.to_string():
                    # mcts on root
                    _ = self.mcts.search(state, self.mcts.root)
                    state_value = tf.math.sigmoid([float(self.mcts.root.score)])
                # set state as new root and do mcts
                else:
                    _ = self.mcts.search(state)
                    state_value = tf.math.sigmoid([float(self.mcts.root.score)])
            # is first step in episode
            else:
                _ = self.mcts.search(state)
                state_value = tf.math.sigmoid([float(self.mcts.root.score)])

            # calculate value of new state
            # new state is not a node in tree
            if state_.to_string() not in self.mcts.root.children:
                # step was a valid move
                if state.to_string() != state_.to_string():
                    # set new state as root and mcts
                    _ = self.mcts.search(state_)
                    state_value_ = tf.math.sigmoid([float(self.mcts.root.score)])
                # step was invalid
                else:
                    state_value_ = tf.math.sigmoid([float(-1)])
            # new state is in tree
            else:
                # set new state as root and do mcts
                _ = self.mcts.search(state_, self.mcts.root.children[state_.to_string()])
                state_value_ = tf.math.sigmoid([float(self.mcts.root.score)])

            # episode is finished
            if done:
                self.mcts.root = None

            return state_value, state_value_

        # decay learning rate 
        if self.memory.mem_cntr % self.batch_size == 0:
            self.alpha = self.alpha * self.gamma
            self.policy_network.compile(optimizer=Adam(learning_rate=self.alpha, clipnorm=1.))

        # get sample from replay buffer and convert to tensor
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

        with tf.GradientTape() as tape:
            
            # calculate value of state
            # is not first step in episode
            if self.mcts.root:
                # state is root of tree 
                if state.to_string() == self.mcts.root.scheduler.to_string():
                    # mcts on root
                    _ = self.mcts.search(state, self.mcts.root)
                    state_value = tf.math.sigmoid([float(self.mcts.root.score)])
                # set state as new root and do mcts
                else:
                    _ = self.mcts.search(state)
                    state_value = tf.math.sigmoid([float(self.mcts.root.score)])
            # is first step in episode
            else:
                _ = self.mcts.search(state)
                state_value = tf.math.sigmoid([float(self.mcts.root.score)])

            # calculate value of new state
            # new state is not a node in tree
            if state_.to_string() not in self.mcts.root.children:
                # step was a valid move
                if state.to_string() != state_.to_string():
                    # set new state as root and mcts
                    _ = self.mcts.search(state_)
                    state_value_ = tf.math.sigmoid([float(self.mcts.root.score)])
                # step was invalid
                else:
                    state_value_ = tf.math.sigmoid([float(-1)])
            # new state is in tree
            else:
                # set new state as root and do mcts
                _ = self.mcts.search(state_, self.mcts.root.children[state_.to_string()])
                state_value_ = tf.math.sigmoid([float(self.mcts.root.score)])
            
            new_states_values = tf.concat(axis=-1, values=[new_states_values, state_value_])
            
            # last step in episode
            if done:
                self.mcts.root = None
            
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            state = tf.convert_to_tensor([[state.to_array()]], dtype=tf.float32)
            states = tf.concat(axis=0, values=[states, state])

            done = tf.convert_to_tensor([int(done)], dtype=tf.float32)
            dones = tf.concat(axis=-1, values=[dones, done])
            
            # get action probabilities for state
            probs = self.policy_network(states)

            target = []
            # calculate loss for every sample
            for j in range(self.batch_size):
                # calculate delta
                delta = rewards[j] + self.gamma * new_states_values[j] * (1-dones[j]) - states_values[j]
                # negative reward
                if rewards[j] < 0:
                    # input for log (0,1]
                    if 1 - probs[j][actions[j]] == 0:
                        probability = 1e-32
                    else:
                        probability = 1 - probs[j][action[j]]
                    
                else:
                    # bound input for log (0,1]
                    if probs[j][actions[j]] == 0:
                        probability = 1e-32
                    else:
                        probability = probs[j][action[j]]
                    
                # calculate log probability and loss
                log_prob = tf.math.log(probability)
                loss = -log_prob * delta
                target.append(loss)
            
            # calculate mean of all losses
            policy_loss = tf.reduce_mean(tf.convert_to_tensor(target))
        # calculate gradients
        gradient = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        # update trainable varaibles
        self.policy_network.optimizer.apply_gradients(zip(gradient, self.policy_network.trainable_variables))
        return state_value, state_value_
