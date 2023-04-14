import matplotlib.pyplot as plt
import configs.parse_csv as parse

task_name = 'reach'

x_1, y_1 = parse.load_data_from_csv(f'TD_MPC/saved_models/{task_name}/train_info_mu.csv')
x_2, y_2 = parse.load_data_from_csv(f'HER/saved_models/{task_name}/train_info_mu.csv')
x_3, y_3 = parse.load_data_from_csv(f'LOGO/saved_models/{task_name}/train_info_mu.csv')
l1, = plt.plot(x_1, y_1, color='g')
l2, = plt.plot(x_2, y_2, color='r')
l3, = plt.plot(x_3, y_3, color='b')
x_max = max([x_1[len(x_1)-1], x_2[len(x_2)-1]])
plt.axis([0, x_max, -50, 0])
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title(task_name)
plt.legend([l1, l2, l3], ['TD-MPC', 'HER', 'LOGO'])
plt.show()
