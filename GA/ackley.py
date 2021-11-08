import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_pic(ax , X, Y, Z, z_max, title, z_min=0):
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    # ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    # plt.savefig("./myProject/Algorithm/pic/%s.png" % title) # 保存图片
    plt.show()


def get_X_AND_Y(X_min, X_max, Y_min, Y_max):
    X = np.arange(X_min, X_max, 0.1)
    Y = np.arange(Y_min, Y_max, 0.1)
    X, Y = np.meshgrid(X, Y)
    return X, Y

# Ackley测试函数
def Ackley(X_min = -32, X_max = 32, Y_min = -32, Y_max = 32):
    X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max)
    Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X**2 + Y**2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))) + np.e + 20
    return X, Y, Z, 25, "Ackley function"