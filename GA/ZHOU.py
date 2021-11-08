import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
DNA_SIZE = 24
POP_SIZE = 300
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.005
N_GENERATIONS = 100
# X_BOUND = [-3, 3]
# Y_BOUND = [-3, 3]
X_BOUND = [-32.768, 32.768]
Y_BOUND = [-32.768, 32.768]
the_result = []
epoch = []

# Ackley测试函数
def F(x, y, a=20, b=0.2, c=2 * np.pi):
    d = 2
    sum = -a * np.exp(-b * np.sqrt((x ** 2 + y ** 2) / d)) - np.exp((np.cos(c * x) + np.cos(c * y)) / d) + a + np.exp(1)
    return sum



# def F(x, y):
# 	return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)- 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2)- 1/3**np.exp(-(x+1)**2 - y**2)

def plot_3d(ax):
    # X = np.linspace(*X_BOUND, 70)
    # Y = np.linspace(*Y_BOUND, 70)
    X = np.linspace(*X_BOUND, 350)
    Y = np.linspace(*Y_BOUND, 350)
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.5)
    ax.set_zlim(0, 20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()

def get_fitness(pop):
    x, y = translateDNA(pop)
    pred = F(x, y)
    return np.max(pred) - pred + 1e-3  # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度

import math
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 1::2]  # 奇数列表示X
    y_pop = pop[:, ::2]  # 偶数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y


def crossover_and_mutation(pop, CROSSOVER_RATE=0.6):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop

# 变异
def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转

# 选择
def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))

# 函数迭代图
def result(epoch,result):
    plt.subplot(1, 2, 2)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('迭代次数')
    plt.ylabel('优化最小值')
    plt.plot(epoch,result,'r-')
    plt.show()


if __name__ == "__main__":
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    ax = Axes3D(fig)

    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_3d(ax)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    for i in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, y, F(x,y), c='black', marker='o')  # 画点
        the_result.append(np.min(F(x,y)))
        epoch.append(i)
        plt.show()
        plt.pause(0.1)
        F_values = F(x, y)#x, y --> Z matrix
        fitness = get_fitness(pop)
        pop = select(pop, fitness)  # 选择生成新的种群
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))

    print_info(pop)
    plt.ioff()
    plot_3d(ax)

    result(epoch,the_result)