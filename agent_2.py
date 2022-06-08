import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense
from mcts import TreeNode, MCTS



class PolicyNetwork(keras.Model):
    def __init__(self, layer_dims, n_actions, input_shape, n_tasks, m_sets, name='policy', chkpt_dir='tmp/policy'):
        super(PolicyNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_dir = f'{chkpt_dir}-{n_tasks}x{m_sets}-multiple_resources'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.model_input_shape = (None,) + input_shape
        
        if len(layer_dims) == 0:
            self.model = tf.keras.models.load_model('Supervised-v6')
        else:
            self.model = tf.keras.Sequential()
            for index, dim in enumerate(layer_dims):
                if index == 0:
                    self.model.add(Dense(dim, input_shape=self.model_input_shape, activation='leaky_relu')) 
                self.model.add(Dense(dim, activation='leaky_relu')) 
            self.model.add(Dense(n_actions, activation='softmax'))

    def call(self, state):
        value = state
        for layer in self.model.layers:
            value = layer(value)
        return value
        
class Agent:
    def __init__(self, input_shape, n_tasks, m_sets, alpha=0.0001, gamma=0.99, policy_layer_dims=[], n_actions=5) -> None:
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None

        self.policy_network = PolicyNetwork(layer_dims=policy_layer_dims, n_actions=n_actions, input_shape=input_shape, n_tasks=n_tasks, m_sets=m_sets)

        self.policy_network.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        probs = self.policy_network(state)

        self.action = tf.math.argmax(probs, axis=1)
        return self.action.numpy()[0]

    def choose_action2(self, observation):
        state = tf.convert_to_tensor([observation])
        probs = self.policy_network(state)
        # print(probs)
        action_probabilites = tfp.distributions.Categorical(probs=probs)
       
        action = action_probabilites.sample()
        self.action = action
        # print(action)
        
        return self.action.numpy()[0]

    def save_models(self):
        print('... save models ...')
        self.policy_network.save_weights(self.policy_network.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.policy_network.load_weights(self.policy_network.checkpoint_file)

    def learn(self, state, state_value, reward, state_, state_value_, done):
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        # print(f'Reward: {reward}')

        with tf.GradientTape() as tape:
            probs = self.policy_network(tf.convert_to_tensor([state.to_array()], dtype=tf.float32))
            
            state_value = tf.math.tanh([float(state_value)])
            
            state_value_ = tf.math.tanh([float(state_value_)])
            
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)
        
            actions_probs = tfp.distributions.Categorical(probs=probs)
        
            log_prob = actions_probs.log_prob(self.action)
            # print(f'Log Prob: {log_prob}')
            delta = reward + self.gamma * state_value_ * (1-int(done)) - state_value
            # print(f'Delta: {delta}')
            policy_loss = -log_prob * delta
            # print(f'Loss {policy_loss}')
            # print('----------------------')
            # input()
        gradient = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        
        self.policy_network.optimizer.apply_gradients(zip(gradient, self.policy_network.trainable_variables))
