from ml_scheduler import load_tasksets, generate_tasksets, Scheduler
import random

def check_schedulability(scheduler):
    time = 0
    for taskset in scheduler.tasksets:
            for task in taskset:
                computation = 0
                for segment in task.segments:
                    computation += segment.execution
                time += (scheduler.hyper_period/task.period) * computation 
    return time <= scheduler.hyper_period

def scheduling_time(scheduler):
    time = 0
    for taskset in scheduler.tasksets:
            for task in taskset:
                computation = 0
                for segment in task.segments:
                    computation += segment.execution
                time += (scheduler.hyper_period/task.period) * computation 
    return time 

if __name__ == '__main__':
    # tasks per taskset
    ntasks = 10
    # number of tasksets
    msets = 1
    # number of processors
    processors = 1

    c_min = 0.05
    c_max = 0.1
    subset = 1

    # sporadic setting 0 = Periodic, 1 = Sporadic
    SPORADIC = 0

    # Least common multiple of all Periods in tasksets
    hyper_period = 10
    possible = 0
    mod = 0
    iterations = 1000
    for i in range(iterations):
        # num of resources
        res_num = random.choice([1, 2, 4, 8])

        generate_tasksets(ntasks, msets, processors, res_num, c_min, c_max, subset, mod)
        tasksets = load_tasksets(ntasks=ntasks, msets=msets, processors=processors, res_num=res_num, c_min=c_min, c_max=c_max, subset=subset, SPORADIC=SPORADIC)
        settings = {
                'hyper_period': hyper_period,
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
        time = 0
        for taskset in scheduler.tasksets:
            for task in taskset:
                computation = 0
                for segment in task.segments:
                    computation += segment.execution
                time += (scheduler.hyper_period/task.period) * computation 

        print(f'Episode {i} Time: {time}')
        if time < scheduler.hyper_period:
            possible += 1

    print(f'{possible} out of {iterations} --> {possible/iterations * 100}%')
                
