import drs
import numpy as np

# Generates sets of tasks of the form 
# [[normal, -1], [critical, res_id], [normal, -1], [critical, res_id],..., period]

# mod used to control whether to generate the frame based or periodic tasks
# mod == 0: frame-based
# mod == 1: periodic
def generate(nsets, msets, processors, num_resources, utilization, critical_min, critical_max, mod):
    # print(f'Processors: {processors}')
    # utilizations = uniform_utilizations(nsets, msets, processors, utilization) 
    utilizations = []
    upper_bounds = [utilization] * nsets
    lower_bounds = [0] * nsets
    
    # print(f'Upper: {upper_bounds}')
    # print(f'Lower: {lower_bounds}')

    for _ in range(msets):
        utilizations.append(drs.drs(nsets, processors, upper_bounds, lower_bounds)) 
    # print(utilizations)   
    tasksets = []
    
    # bounds of the critical sections
    c_min = critical_min
    c_max = critical_max
    num_res = num_resources

    periods = [1, 2, 5, 10]
    # periods = [5, 10]

    for i in range(msets):
        taskset = []
        for j in range(nsets):
            task = []
            execution = utilizations[i][j]
            # print(f'Execution: {execution}')
            critical = np.random.uniform(c_min, c_max) * execution
            normal = execution - critical
            # print(f'Normal: {normal}')
            # print(f'Critical: {critical}')

            num_critical = np.random.randint(2, 6)
            # print(num_critical)

            # normal_sets = randfixed.UUniFastDiscard((num_critical+1), normal, 10)
            normal_sets = []
            critical_sets = []
            for s in range(10):
                normal_sets.append(drs.drs(n=num_critical, sumu=normal))
                critical_sets.append(drs.drs(n=num_critical, sumu=critical))
            # normal_sets = randfixed.UUniFastDiscard(num_critical, normal, 10)
            
            # critical_sets = randfixed.UUniFastDiscard(num_critical, critical, 10)
            normal_set = normal_sets[np.random.randint(0,10)]
            critical_set = critical_sets[np.random.randint(0, 10)]
            # print('_________________________________NORMALSETS_______________________________')
            # print(normal_set)
            # print('_________________________________CRITICALSETS_______________________________')
            # print(critical_set)

            sum = 0
            for z in normal_set:
                # print(z)
                sum += z
            
            # print('______________________________SUM_______________________________')
            # print(sum)

            for k in range(0, num_critical):
                resource_id = np.random.randint(0, num_res)
                task.append([normal_set[k], -1])
                task.append([critical_set[k], resource_id])

            # task.append([normal_set[num_critical], -1])
            task.append([0, -1])
            if (mod == 0):
                period = 1
            # else periodic tasks
            else:
                period = periods[np.random.randint(0, 4)]
                for k in range(0, 2*num_critical+1):
                    task[k][0] = task[k][0] * period

            task.append(period)

            taskset.append(task)			
        tasksets.append(taskset)
    return tasksets


