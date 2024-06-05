import copy
import heapq
import random
from random import randint, shuffle
from collections import defaultdict, deque
import matplotlib.pyplot as plt 



def read_data_from_file(filename):
    try:
        with open(filename, 'r') as file:
            data = [line.strip() for line in file if not line.strip().startswith('#')]

        # Extracting number of machines and tasks for one job
        num_tasks, num_machines = map(int, data[0].split())

        # Initializing list to store job data
        job_data = []

        # Parsing machine number and processing time for each job
        for line in data[1:]:
            pairs = list(map(int, line.split()))
            job_data.append([pairs[i:i + 2] for i in range(0, len(pairs), 2)])

        return num_machines, num_tasks, job_data

    except FileNotFoundError:
        print("File not found.")
        return None


def calculate_cmax(num_machines, job_data):
    # Initialize a list to store the completion time for each machine
    machine_completion_times = [0] * num_machines

    # Initialize a list to store the start time for each job's tasks
    job_current_times = [0] * len(job_data)

    # Create a list to store the fixed job data with adjusted start times
    fixed_job_data = []

    # Iterate through each job
    for job_index, job in enumerate(job_data):
        job_time = job_current_times[job_index]  # Initialize the current time for the current job

        fixed_job = []  # To store adjusted tasks for the current job
        for task in job:
            machine = task[0]  # Extract machine number
            processing_time = task[1]  # Extract processing time

            # Calculate start time for the current task on the current machine
            start_time = max(job_time, machine_completion_times[machine])

            # Update completion time for the current machine
            machine_completion_times[machine] = start_time + processing_time

            # Update current time for the job to the completion time of the current task
            job_time = machine_completion_times[machine]

            # Append the adjusted task with its start time to the fixed job
            fixed_job.append((machine, processing_time, start_time))

        # Store the adjusted tasks for the current job
        fixed_job_data.append(fixed_job)
        job_current_times[job_index] = job_time

    # Calculate the completion times of each job using the fixed job data
    job_completion_times = [max(task[2] + task[1] for task in job) for job in fixed_job_data]

    # Find the maximum completion time among all jobs
    cmax = max(job_completion_times)

    return cmax, fixed_job_data


class Schedule():
    def __init__(self, num_machines, machines_perm):
        self.num_machines = num_machines
        self.machines_perm = machines_perm

    def initRandom(self):
        for i in range(self.num_machines):
            shuffle(self.machines_perm[i])

    def mutation(self):
        machine = randint(0, self.num_machines-1)
        num_tasks = len(self.machines_perm[machine])
        first = randint(0, num_tasks-1)
        second = randint(0, num_tasks-1)
        self.machines_perm[machine][first], self.machines_perm[machine][second] = self.machines_perm[machine][second], self.machines_perm[machine][first]

    def crossover(self, schedule2):
         # Step 1: Select a random segment from self.jobs
        start = random.randint(0, self.num_machines - 1)
        end = random.randint(start, self.num_machines)
        child_perm = copy.deepcopy(self.machines_perm)
        
        # Step 2: Fill missing jobs from schedule2
        for machine_no in range(self.num_machines):
            if machine_no < start or machine_no > end:
                child_perm[machine_no] = schedule2.machines_perm[machine_no]
        
        return Schedule(self.num_machines, child_perm)
    
    def evaluate(self):
        return create_schedule(self.machines_perm)[1]

    def scheduling(self):
        return create_schedule(self.machines_perm)[0]
    
    def eval_schedule(self):
        return create_schedule(self.machines_perm)


class GeneticAlgorithm():
    def __init__(self, populationSize, generationLimit, num_machines, jobs):
        self.population = []
        self.populationSize = populationSize
        self.generationLimit = generationLimit
        self.num_machines = num_machines
        self.jobs = jobs

    def firstPopulation(self):
        for i in range(self.populationSize):
            unit = Schedule(self.num_machines, self.jobs)
            unit.initRandom()
            self.population.append(unit)

    def getBestUnit(self, units=None):
        if units is None:
            units = self.population
        best = units[0]
        best_cmax = units[0].evaluate()
        for unit in units:
            curr_Cmax = unit.evaluate()
            if curr_Cmax < best_cmax:
                best = unit
                best_cmax = curr_Cmax

        return best

    def combineBest(self):
        no1 = self.getBestUnit()
        tmp_pop = self.population
        tmp_pop.remove(no1)
        no2 = self.getBestUnit(tmp_pop)

        return no1.crossover(no2)

    def tournament(self, tournament_size=10):
        tournament_pop = random.sample(self.population, tournament_size)

        return self.getBestUnit(tournament_pop)

    def run(self, crossover_chance=0.50, mutation_chance=0.50, choice_percent=10):
        self.firstPopulation()
        tournament_size = self.populationSize * choice_percent // 100

        for generation in range(self.generationLimit):
            #print("Generation: ", generation)
            best_unit = self.getBestUnit()
            #print("Best chosen")
            new_population = []
            for p in range(self.populationSize - 1):
                parent1 = self.tournament(tournament_size)
                #print("Tournament 1 fin")
                parent2 = self.tournament(tournament_size)
                #print("Tournament 2 fin")
                child = copy.deepcopy(parent1)
                if random.uniform(0, 1) < crossover_chance:
                    child = parent1.crossover(parent2)
                if random.uniform(0, 1) < mutation_chance:
                    child.mutation()
                new_population.append(child)
                #print("Created pop: ", p)
            new_population.append(best_unit)
            self.population = copy.deepcopy(new_population)
            new_population.clear()
            """ if generation in limits:
                best_unit = self.getBestUnit()
                s, c = best_unit.eval_schedule()
                print('gen: '+str(generation)+' cmax: '+str(c))
                cmaxs.append(c) """

        best_unit = self.getBestUnit()

        return best_unit


def group_tasks_by_machine(job_data, num_machines):

    # Create a dictionary to hold lists of tasks for each machine
    machine_dict = defaultdict(list)

    # Iterate over each job (row in job_data)
    for job_index, job in enumerate(job_data):
        # Iterate over each task in the job (columns in the row)
        for task_index, (machine, processing_time) in enumerate(job):
            # Append the (job index, task index) pair and processing time to the corresponding machine's list
            machine_dict[machine].append(((job_index, task_index), processing_time))

    # Convert the dictionary to a list of lists
    grouped_data = [[] for _ in range(num_machines)]

    for machine, tasks in machine_dict.items():
        grouped_data[machine] = tasks

    return grouped_data


def create_schedule(machines):
    # Initialize the schedule dictionary
    schedule = defaultdict(list)
    job_end_times = defaultdict(int)
    machine_end_times = [0] * len(machines)
    job_next_task = defaultdict(int)  # Keep track of the next task index for each job
    task_ready = defaultdict(bool)  # Keep track of task readiness
    task_added = set()  # Keep track of added tasks to prevent duplicates
    remaining = copy.deepcopy(machines)

    # Initialize the priority queue to process tasks in order
    pq = []
    for machine_index, tasks in enumerate(machines):
        if tasks:
            job_index, task_index = tasks[0][0]
            processing_time = tasks[0][1]
            if task_index == 0:  # Only add the first task of each job initially
                heapq.heappush(pq, (0, job_index, task_index, machine_index, processing_time))
                task_added.add((job_index, task_index, machine_index))
                remaining[machine_index].remove(((job_index, task_index), processing_time))

    cmax = 0  # Initialize the maximum end time

    # Function to handle deadlock by adding a task with task_index == 0
    def handle_initial_deadlock():
        for machine_index, tasks in enumerate(remaining):
            if tasks:
                for i, task in enumerate(tasks):
                    job_index, task_index = task[0]
                    if task_index == 0:
                        processing_time = task[1]
                        heapq.heappush(pq, (0, job_index, task_index, machine_index, processing_time))
                        remaining[machine_index].remove(((job_index, task_index), processing_time))
                        
                        # Move the task to the first position in machines[machine_index]
                        machines[machine_index].insert(0, machines[machine_index].pop(machines[machine_index].index(task)))
                        return

    # Function to handle deadlock by swapping tasks
    def handle_swap_deadlock():
        order = list(range(0, (len(machines) - 1)))
        shuffle(order)
        for random_index in order:
            tasks = remaining[random_index]
            if tasks:
                for i, task in enumerate(tasks):
                    job_index, task_id = task[0]
                    if task_ready[(job_index, task_id)] or task_id == 0:
                        next_task = tasks[0]
                        tasks[i] = next_task
                        tasks[0] = task

                        #poprawić wyszukiwanie tasków w maszynie orginalnej

                        # Also update the tasks in the original machines list
                        machines_tasks = machines[random_index]
                        mach_ind = 0
                        for j, m_task in enumerate(machines_tasks):
                            if m_task[0] == tasks[0]:
                                mach_ind = j

                            if m_task[0] == next_task[0]:
                                machines_tasks[j] = machines_tasks[mach_ind]
                                machines_tasks[mach_ind] = m_task
                                break

                        heapq.heappush(pq, (end_time, job_index, task_id, random_index, task[1]))
                        remaining[random_index].remove(task)
                        return

    # Check for initial deadlock
    if not pq and any(remaining):
        handle_initial_deadlock()

    while pq or any(remaining):
        if pq:
            current_time, job_index, task_index, machine_index, processing_time = heapq.heappop(pq)

            # Determine the start time for the current task
            start_time = max(current_time, job_end_times[job_index], machine_end_times[machine_index])
            end_time = start_time + processing_time

            # Schedule the task
            schedule[machine_index].append(((job_index, task_index), start_time, end_time))

            # Update end times
            job_end_times[job_index] = end_time
            machine_end_times[machine_index] = end_time

            # Update cmax
            cmax = max(cmax, end_time)

            # Mark the next task of this job as ready
            job_next_task[job_index] = task_index + 1
            if job_next_task[job_index] < len(machines[machine_index]):
                task_ready[(job_index, job_next_task[job_index])] = True

        task_added_during_loop = False

        for i, tasks in enumerate(remaining):
            if tasks:
                next_job_index, next_task_id = tasks[0][0]
                next_processing_time = tasks[0][1]
                if task_ready[(next_job_index, next_task_id)] or next_task_id == 0:
                    heapq.heappush(pq, (end_time, next_job_index, next_task_id, i, next_processing_time))
                    remaining[i].remove(((next_job_index, next_task_id), next_processing_time))
                    task_added_during_loop = True

        # Check for deadlock during the loop
        if not pq and any(remaining) and not task_added_during_loop:
            handle_swap_deadlock()

    return dict(schedule), cmax

filename = 'job_data.txt'  # Replace 'job_data.txt' with the actual filename
num_machines, num_tasks, job_data = read_data_from_file(filename)

grouped_data = group_tasks_by_machine(job_data, num_machines)


ga = GeneticAlgorithm(20,20, num_machines, grouped_data)
best = ga.run()

schedule, c_max = best.scheduling(), best.evaluate()
for machine, tasks in schedule.items():
    print(f"Machine {machine}:")
    for task in tasks:
        print(f"  Task {task[0]}: Start at {task[1]}, End at {task[2]}")
print("Cmax: ", c_max)


#for testing different pop sizes and gen limits:
""" sizes = [100,150,200,250]
limits = [25,50,75,100,125,150,175,200,225,250]
cmaxs = []
opt = 666       #optimal solution
optimal = [opt for i in limits]
#limits = [lim for lim in range(limit)]
plt.plot(limits,optimal,label='optimal solution',linestyle='dashed',color='blue')
print(filename)

for size in sizes:
    cmaxs = []
    print('size: '+str(size))
    #print(limits[-1])
    ga = GeneticAlgorithm(size, limits[-1], num_machines, grouped_data)
    best = ga.run(mutation_chance=0.75, crossover_chance=0.6, choice_percent=15)
    schedule, c_max = best.eval_schedule()
    print('gen: '+str(limits[-1])+' cmax: '+str(c_max))
    cmaxs.append(c_max)
    plt.plot(limits,cmaxs, label='pop. size='+str(size))
    

plt.xlabel('Generation')
plt.ylabel('Scheduling time')
plt.title('Scheduling times for la01')
plt.legend()
plt.show() """