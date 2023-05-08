import csv

def save_data_to_csv(steps, rewards, filename, time=0):
    if time == 0:
        fields = ['steps', 'rewards']
    else:
        fields = ['steps', 'rewards', 'times']

    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        # writing the fields
        writer.writeheader()
        if len(steps) == len(rewards):
            for i in range(len(steps)):
                # writing the data rows
                if time == 0:
                    writer.writerow({'steps': steps[i], 'rewards': rewards[i]})
                else:
                    writer.writerow({'steps': steps[i], 'rewards': rewards[i], 'times': time[i]})


def load_data_from_csv(filename, with_time=False):
    with open(filename, mode='r') as file:
        steps, rewards, times = [], [], []
        csvFile = csv.DictReader(file)
        # displaying the contents of the CSV file
        for lines in csvFile:
            steps.append(int(lines['steps']))
            rewards.append(float(lines['rewards']))
            if with_time:
                times.append(str(lines['times']))
    return steps, rewards, times

