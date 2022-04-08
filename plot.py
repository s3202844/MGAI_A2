import numpy as np
import matplotlib.pyplot as plt

dataset = []
for i in range(10):
    path = "data/" + str(i) + ".npy"
    data = np.load(path)
    temp_data = [[], []]
    for j in range(data.shape[1]):
        if data[0][j] == -1:
            if len(temp_data[0]) > 0:
                dataset += [np.array(temp_data)]
            temp_data = [[], []]
        else:
            temp_data[0] += [data[0][j]]
            temp_data[1] += [data[1][j]]
    if len(temp_data[0]) > 0:
        dataset += [np.array(temp_data)]
max_size, min_size = 0, 100
for d in dataset:
    min_size = min(min_size, d.shape[1])
    max_size = max(max_size, d.shape[1])
distance0 = []
distance1 = []
control0 = []
control1 = []
for i in range(len(dataset)):
    c0, c1 = 0, 0
    for j in range(dataset[i].shape[1]):
        if dataset[i][0][j] == 0:
            c0 += 1
        if dataset[i][1][j] == 0:
            c1 += 1
    control0 += [c0/dataset[i].shape[1]]
    control1 += [c1/dataset[i].shape[1]]
    step = dataset[i].shape[1]/(max_size-1)
    dx = np.arange(0, dataset[i].shape[1]+step, step)
    line0 = np.interp(dx, np.arange(0, dataset[i].shape[1]), dataset[i][0])
    line1 = np.interp(dx, np.arange(0, dataset[i].shape[1]), dataset[i][1])
    distance0 += [line0]
    distance1 += [line1]
distance0 = np.mean(distance0, axis=0)
distance1 = np.mean(distance1, axis=0)

plt.plot(distance0, label="MCTS")
plt.plot(distance1, label="Random")
plt.xlabel("steps")
plt.ylabel("distance to ball")
plt.title("Performance of the closest players' distance to the ball. \n Result is averaged over 20 repetitions.")
plt.legend()
plt.savefig("ball.png")
plt.show()
plt.clf()

plt.boxplot((control0, control1), labels=("MCTS", "Random"))
plt.ylabel("ratio of carrying ball")
plt.title("Performance of different methods' ratio of carrying ball.")
plt.savefig("ratio.png")
plt.show()
plt.clf()