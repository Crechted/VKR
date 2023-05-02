import matplotlib.pyplot as plt
import configs.parse_csv as parse

def get_task_name_on_rus(name):
    if name == 'reach':
        return 'Достижение точки'
    if name == 'push':
        return 'Толкание объекта'
    if name == 'slide':
        return 'Скольжение объекта'
    if name == 'pick_and_place':
        return 'Перетаскивание объекта'

task_name = 'push'

# x_1, y_1 = parse.load_data_from_csv(f'TD_MPC/saved_models/{task_name}/train_info_mu.csv')
x_1, y_1 = parse.load_data_from_csv(f'Archive/TD_MPC/{task_name}/1/train_info_mu.csv')
x_2, y_2 = parse.load_data_from_csv(f'HER/saved_models/{task_name}/train_info_mu.csv')
# x_3, y_3 = parse.load_data_from_csv(f'LOGO/saved_models/{task_name}/train_info_mu.csv')
l1, = plt.plot(x_1, y_1, color='g')
l2, = plt.plot(x_2, y_2, color='r')
# l3, = plt.plot(x_3, y_3, color='b')
x_max = max([x_1[len(x_1)-1], x_2[len(x_2)-1]])
plt.axis([0, x_max, -50, 0])
plt.xlabel('Шаги')
plt.ylabel('Награда')
plt.title(get_task_name_on_rus(task_name))
plt.legend([l1, l2], ['TD-MPC', 'HER', 'LOGO'])
plt.show()
