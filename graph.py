import matplotlib.pyplot as plt
import configs.parse_csv as parse
from datetime import datetime


def get_task_name_on_rus(name):
    if name == 'reach':
        return 'Достижение точки'
    if name == 'push':
        return 'Толкание объекта'
    if name == 'slide':
        return 'Скольжение объекта'
    if name == 'pick_and_place':
        return 'Перетаскивание объекта'


def reform_time(dates):
    seconds = []
    for i in range(len(dates)):
        seconds.append((datetime.fromisoformat(dates[i]) - datetime.fromisoformat(dates[0])).total_seconds())
    return seconds


task_name = 'push'

# x_1, y_1 = parse.load_data_from_csv(f'TD_MPC/saved_models/{task_name}/train_info_mu.csv')
x, y, t, l = [], [], [], []
fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(2):
    xt, yt, tt = parse.load_data_from_csv(f'Archive/TD_MPC/{task_name}/{i + 1}/train_info_mu.csv', with_time=True)
    x.append(xt)
    y.append(yt)
    times = reform_time(tt)
    t.append(times)
    # print(times)

for i in range(len(x)):
    ax1.plot(x[i], y[i], label=f'{i}')
    ax2.plot(t[i], y[i], label=f'{i}')



# l3, = plt.plot(x_3, y_3, color='b')
print(max(max(t)))
x_max = max(max(x))
t_max = max(max(t))
ax1.axis([0, x_max, -50, 0])
ax1.set_xlabel('Шаги')
ax1.legend(loc='upper right')
ax1.set_ylabel('Награда')
ax1.grid(True)
ax2.axis([0, t_max, -50, 0])
ax2.set_xlabel('Время, c')
ax2.legend(loc='upper right')
ax2.grid(True)
ax2.set_ylabel('Награда')
fig.suptitle(get_task_name_on_rus(task_name))
# plt.legend([l1, l2], ['TD-MPC', 'HER', 'LOGO'])
plt.show()
