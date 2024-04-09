def read_data_from_file(filename):
    try:
        with open(filename, 'r') as file:
            data = [line.strip() for line in file if not line.strip().startswith('#')]

        # Extracting number of machines and tasks for one job
        num_machines, num_tasks = map(int, data[0].split())

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

    # Iterate through each job
    job_completion_times = []
    for job in job_data:
        job_time = 0  # Initialize completion time for the current job
        for task in job:
            machine = task[0]  # Extract machine number
            processing_time = task[1]  # Extract processing time

            # Calculate start time for the current task on the current machine
            start_time = max(job_time, machine_completion_times[machine])

            # Update completion time for the current machine
            machine_completion_times[machine] = start_time + processing_time

            # Update completion time for the current job
            job_time = machine_completion_times[machine]

        job_completion_times.append(job_time)  # Store completion time for the current job

    # Find the maximum completion time among all jobs
    cmax = max(job_completion_times)

    return cmax

# Example usage
filename = 'job_data.txt'  # Replace 'job_data.txt' with the actual filename
num_machines, num_tasks, job_data = read_data_from_file(filename)
if num_machines is not None:
    print("Number of machines:", num_machines)
    print("Number of tasks per job:", num_tasks)
    print("Job data:")
    for job in job_data:
        print(job)
    print("Cmax:", calculate_cmax(num_machines, job_data))