from ml_scheduler import Scheduler, SubJob, Job, Task, generate_tasksets, load_tasksets
from scheduler_env2 import SchedulerEnv
from agent import Agent
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU, Flatten
from rl.agents import DQNAgent, SARSAAgent, CEMAgent
from rl.policy import BoltzmannQPolicy, SoftmaxPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy,  MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
import random
import matplotlib.pyplot as plt
import numpy as np
import datetime
from mcts import MCTS, TreeNode
from copy import deepcopy
import csv

def build_model(states, actions):
    model = Sequential()
    print((1,) + states)
    model.add(Conv2D(64, (3,3), (1,1), input_shape=(1,) + states)) 
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(64, (1,1), (1,1))) 
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(64, (1,1), (1,1))) 
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(actions, activation='softmax'))
    return model

def build_dqnagent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=1000, target_model_update=1e-2)
    return dqn

def build_sarsaagent(model, actions):
    test_policy = BoltzmannQPolicy()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=50000)
    sarsa = SARSAAgent(model=model, policy=policy, test_policy=test_policy, nb_actions=actions, nb_steps_warmup=1000, gamma=0.99, train_interval=1)
    return sarsa

def build_cemagent(model, actions):
    memory = EpisodeParameterMemory(limit=50000, window_length=1)
    cem = CEMAgent(model, actions, memory, batch_size=128, nb_steps_warmup=1000, train_interval=64, elite_frac=0.05, memory_interval=1, theta_init=None, noise_decay_const=0.0, noise_ampl=0.0)
    return cem

if __name__ == '__main__':
    # tasks per taskset
    ntasks = 5
    # number of tasksets
    msets = 1
    # number of processors
    processors = 1
    # num of resources
    res_num = 1

    c_min = 0.05
    c_max = 0.1
    subset = 1

    # sporadic setting 0 = Periodic, 1 = Sporadic
    SPORADIC = 0
    mod = 1

    generate_tasksets(ntasks=ntasks, msets=msets, processors=processors, res_num=res_num, c_min=c_min, c_max=c_max, subset=subset, mod=mod)
    tasksets = load_tasksets(ntasks=ntasks, msets=msets, processors=processors, res_num=res_num, c_min=c_min, c_max=c_max, subset=subset, SPORADIC=SPORADIC)
    settings = {
        'ntasks': ntasks, 
        'msets': msets, 
        'processors': processors,
        'res_num': res_num,
        'c_min': c_min,
        'c_max': c_max,
        'subset': subset,
        'SPORADIC': SPORADIC,
        'mod': mod,
        }

    scheduler = Scheduler(tasksets, settings)
    # scheduler2 = Scheduler(scheduler=scheduler)
    # print(f'Scheduler1: {scheduler}')
    # print(f'Scheduler2: {scheduler2}')

    # print(f'Tasksets1: {scheduler.tasksets}\n')
    # print(f'Tasksets2: {scheduler2.tasksets}')

    # print()
    # print()

    # print(f'Tasks1: {scheduler.tasksets[0][0].tasks}')
    # print(f'Tasks2: {scheduler2.tasksets[0][0].tasks}')
    # scheduler.schedule_loop()
    env = SchedulerEnv(scheduler)
    print(env.action_shape)
    print(env.observation_shape)

    model = build_model(env.observation_shape, env.action_shape)
    model.summary()
    agent = build_dqnagent(model, env.action_shape)
    # agent.load_weights(f'tmp/dqn-{ntasks}x{msets}-multiple_resources-periodic')
    agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    agent.fit(env, nb_steps=250000, visualize=False, verbose=1)
    agent.save_weights(f'tmp/dqn-{ntasks}x{msets}-multiple_resources-periodic', overwrite=True)
    # agent = Agent(n_actions=env.action_shape, input_shape=env.observation_shape, alpha=1e-5, n_tasks=ntasks, m_sets=msets, policy_layer_dims=[512, 1024, 2048, 1024, 512], load_memory=0)
    # agent.load_models()
    # np.random.seed(0)
    scores = agent.test(env, nb_episodes=150, visualize=False)
    print(np.mean(scores.history['episode_reward']))
    # observations = []
    # actions = []
    # score_history = []
    # step = 0
    # # step_score = []
    # best_score = env.reward_range[0]
    # avg_score_history = []
    '''
    Highscores:
    policy-3x1-multiple_resources: 56.77 mcts: 20, 25, lr: 1e-10, u: 10%
    policy-5x1-multiple_resources: 24.18 mcts: 15, 20, lr: 1e-15, u: 10%
    policy-8x1-multiple_resources: 0 mcts: 16, 20, lr: 1e-25, u: 10%
    policy-10x1-multiple_resources: 0 mcts: 10, 5, lr: 1e-5, u: 10%
    
    '''
    # best_score = 2400.18
    # EPISODES = 150
    # wons = 0
    # lost = 0
    # invalids = 0
    # won = 1
    # for i in range(EPISODES):
    #     time_start = datetime.datetime.now()
    #     # mcts = MCTS()
    #     # best_move = None
    #     done = False
    #     score = 0
    #     observation = env.reset()
    #     invalid = 0
    #     # env.render()        
    #     while not done:
    #         # observations.append(observation.to_array())
    #         # choose action
    #         # print(action)
    #         # states = deepcopy(observation).generate_states()
    #         # print(f'# States : {len(states)}')
    #         # action = [i[1] for i in states]
    #         # actions.append(action)    
    #         # state_, reward, done, info = env.step(random.choice(states))

    #         action = agent.choose_action(observation.to_array())
    #         state_, reward, done, info = env.step(action)
    #         # agent.remember(observation.get_observation(), action, reward, state_.get_observation(), int(done))
    #         # agent.learn()
    #         # state_value, state_value_ = agent.learn(observation, reward, state_, done)
    #         # agent.remember(np.asarray([[observation.to_array()]]), state_value, action, reward, np.asarray([[state_.to_array()]]), state_value_, int(done))
    #         # step_score.append(reward)
    #         # if step % 1000 == 0:
    #         #     print(f'Episode: {i}, Step: {step}, 1000 steps average {np.mean(step_score[-1000:])}')
    #         score += reward
    #         observation = state_
    #         # if reward != -10000:
    #         #     env.render()
    #         #     print(f'Score: {score} , Reward: {reward}')
    #         #     print(f'Action: {action}, Sum: {action.sum()-1}')
    #         #     print(f'Step: {step}')
    #         # print(action)
    #         # print(observation.get_observation())
    #         step += 1
    #         # print(f'reward: {reward} 1000 steps average {np.mean(step_score[-1000:])}')
    #         # input()
    #         # env.render()
    #     if reward == 1:
    #         score = 10000
    #     score_history.append(score)
    #     avg_score = np.mean(score_history[-100:])
    #     avg_score_history.append(avg_score)
            
    #     # if avg_score > best_score or i % 10 == 0:
    #     #     best_score = avg_score
    #     #     agent.save_models()
        
    #     invalid = info['invalid']
    #     # if reward == 1:
    #     #     wons += 1
    #     # elif reward == 0:
    #     #     lost += 1
    #     # else:
    #     #     invalids += 1
        
    #     time_end = datetime.datetime.now()
    #     time_delta = time_end - time_start
    #     print(f'episode {i} cummulative score {score}, steps {step}, invalid moves {invalid}, final score {reward}, 100 game average {avg_score}, time {time_delta}')

        # with open('states.csv', 'a') as f:
        #     # create the csv writer
        #     writer = csv.writer(f)

        #     # write a row to the csv file
        #     for observation in observations:
        #         writer.writerow(observation)
        #     f.close()

        # with open('actions.csv', 'a') as f:
        #     # create the csv writer
        #     writer = csv.writer(f)

        #     # write a row to the csv file
        #     for action in actions:
        #         writer.writerow(action)
        #     f.close()
        # observations = []
        # actions = []
    # print(f'Games Won: {wons}, Games Lost: {lost}, Invalids: {invalids}')
    # print(f'Accuracy: {(wons/EPISODES)*100}%')
# plt.title('trailing 100 episode points average')
# plt.plot(avg_score_history[100:])
# plt.xlabel('Episode #')
# plt.ylabel('points average')
# plt.show()