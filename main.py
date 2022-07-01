from ml_scheduler import Scheduler, generate_tasksets, load_tasksets
from scheduler_env import SchedulerEnv
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import datetime

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
    # framebased = 0
    mod = 1

    # generate and load tasksets
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

    # generate Scheduler and Environment
    scheduler = Scheduler(tasksets, settings)
    env = SchedulerEnv(scheduler)

    # init Agent
    agent = Agent(n_actions=env.action_shape, input_shape=env.observation_shape, alpha=1e-5, n_tasks=ntasks, m_sets=msets, cnn_layers=5, load_memory=0)
    # agent.load_models()
    
   
    score_history = []
    step = 0

    best_score = env.reward_range[0]
    avg_score_history = []
    
    EPISODES = 1500
    won = 0
    for i in range(EPISODES):
        time_start = datetime.datetime.now()
        done = False
        score = 0
        observation = env.reset()
        invalid = 0
        # env.render()        
        while not done:
            # choose actions heurustically
            action = agent.choose_action(observation.to_array())
            # do step in environment
            state_, reward, done, info = env.step(action)
            # let agent learn
            state_value, state_value_ = agent.learn(observation, reward, state_, done)
            # let agent remember important values
            agent.remember(np.asarray([[observation.to_array()]]), state_value, action, reward, np.asarray([[state_.to_array()]]), state_value_, int(done))
            
            # calculate current score and set observation to new state
            score += reward
            observation = state_
            
            step += 1
            # env.render()

        # episode was won
        if reward == 1:
            won += 1
        
        # append scores to score history and save agents
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
            
        if avg_score > best_score or i % 10 == 0:
            best_score = avg_score
            agent.save_models()
        
        invalid = info['invalid']
        
        time_end = datetime.datetime.now()
        time_delta = time_end - time_start

        # debug
        print(f'episode {i} cummulative score {score}, steps {step}, invalid moves {invalid}, final score {reward}, 100 game average {avg_score}, time {time_delta}, won {won}')

plt.title('trailing 100 episode points average')
plt.plot(avg_score_history[100:])
plt.xlabel('Episode #')
plt.ylabel('points average')
plt.show()