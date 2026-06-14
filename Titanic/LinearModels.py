import numpy as np
import pandas as pd
import scipy.linalg as sla
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from MyLogisticRegression import MyLogisticRegression
from MyElasticLogisticRegression import MyElasticLogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def f(r):
    """
    :param r: np.array(np.float) вектор длины 2
    :return: np.float

    На вход функции мы подаем двумерный массив: r = [x,y]
    Используем функцию np.sin() (https://numpy.org/doc/stable/reference/generated/numpy.sin.html) и оператор возведения в степень ** (https://numpy.org/doc/stable/reference/generated/numpy.power.html), которые применяются к массивам поэлементно
    """
    return np.sum(np.sin(r)**2)

# Сначала реализуем функцию, вычисляющую градиент

def grad_f(r):
    """
    Градиент функциии f, определенной выше.
    :param r: np.array[2]: float вектор длины 2
    :return: np.array[2]: float вектор длины 2
    """
    return np.sin(2 * r)

# Проверим, что градиент принимает вектор из двух чисел и выдает на этой точке верное значение
assert np.allclose(grad_f(np.array([1, 2])),np.array([0.90929743, -0.7568025]))

def grad_descent_2d(f, grad_f, lr, num_iter=100, r0=None):
    """
    функция, которая реализует градиентный спуск для функции f от двух переменных.
        :param f: скалярная функция двух переменных
        :param grad_f: функция, возвращающая градиент функции f (устроена так, как реализованная вами выше grad_f)
        :param lr: learning rate алгоритма
        :param num_iter: количество итераций градиентного спуска
        :param r0: начальное значение аргумента f(r), которое мы инициируем случайно или можем передать как аргумент в функцию grad_descent_2d()

        :return: np.array[num_iter, 2] пары вида (r, f(r))
    """
    if r0 is None:
        r0 = np.random.random(2)

    # в процессе градиентного спуска будем сохранять в переменную history значения
    history = []

    # итерация цикла -- шаг градиентного спуска
    curr_r = r0.copy()
    for iter_num in range(num_iter):
        entry = np.hstack((curr_r, f(curr_r)))
        history.append(entry)

        # curr_r - текущие значения аргумента в процессе градиентного спуска
        curr_r -= grad_f(curr_r) * lr # YOUR CODE. Не забудьте про lr!

    return np.vstack(history)

steps = grad_descent_2d(f, grad_f, lr=0.1, num_iter=20)

# Создание сетки
border = 1.5 # Меняйте значения border, чтобы увеличивать или уменьшать область
X, Y = np.meshgrid(np.linspace(-border, border, 100), np.linspace(-border, border, 100))
Z = np.array([f(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

# Создание 3D-графика
fig = go.Figure(data=[
    go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')
])

# Добавление линии и маркеров
fig.add_trace(go.Scatter3d(
    x=steps[:, 0],
    y=steps[:, 1],
    z=steps[:, 2],
    mode='lines+markers',
    marker=dict(size=8, color='red', symbol='cross'),
    line=dict(color='black', width=5)
))

# Настройка осей
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        zaxis=dict(range=[-1, 5]),  # Установка пределов для Z
        aspectratio=dict(x=1, y=1, z=0.7)  # Соотношение сторон
    ),
    margin=dict(l=0, r=0, b=0, t=0)  # Удаление отступов
)

# Отображение графика
pio.renderers.default = "browser"
# fig.show()

path = []

X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

zs = np.array([f(np.array([x,y]))
              for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)


ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, zorder=2)

ax.plot(xs=steps[:, 0], ys=steps[:, 1], zs=steps[:, 2],
        marker='*', markersize=20, zorder=3,
        markerfacecolor='y', lw=3, c='black')

ax.set_zlim(0, 5)
ax.view_init(elev=60)
# plt.show()

# Посмотрим на график значений функции от шага - loss
plt.figure(figsize=(14,7))
plt.xlabel('grad descent step number')
plt.ylabel('$f(r)$')
plt.title('Значение функции на каждом шаге гардиентного спуска.')

f_values = list(map(lambda x: x[2], steps))
plt.plot(f_values, label='gradient descent result')
plt.legend()
# plt.show()

# генератор батчей, который будет принимать на вход признаки, ответы и размер батча.
# Генератор должен возвращать tuple из ( Xbatch ,  ybatch ).
# Если размер датасета не делится на batch_size, то последний маленький батч возвращать не нужно.

def generate_batches(X, y, batch_size):
    """
    param X: np.array[n_objects, n_features] --- матрица объекты-признаки
    param y: np.array[n_objects] --- вектор целевых переменных
    """
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)

    perm = np.random.permutation(len(X))
    n_batches = len(X) // batch_size

    for batch_start in range(n_batches):
        yield X[batch_start*batch_size: (batch_start+1)*batch_size], y[batch_start*batch_size: (batch_start+1)*batch_size]

def logit(x, w):
    return np.dot(x, w)

def sigmoid(h):
    return 1. / (1 + np.exp(-h))


# Тестируем написанную функцию
X_fake = np.arange(100)[:, np.newaxis]
y_fake = np.arange(100) + 1000

X_reconstructed, y_reconstructed = [], []
for X_batch, y_batch in generate_batches(X_fake, y_fake, 10):
    X_reconstructed.append(X_batch)
    y_reconstructed.append(y_batch)

X_reconstructed = np.concatenate(X_reconstructed)
y_reconstructed = np.concatenate(y_reconstructed)
print(" ---- reconstructed ---- ")
print(X_reconstructed)
print(y_reconstructed)

assert (X_fake == X_reconstructed).all(), "Что-то не так!"
assert (y_fake == y_reconstructed).all(), "Что-то не так!"

assert (np.sort(X_reconstructed, axis=0) == X_fake).all(), "Что-то не так!"

print(" ----- MyLogisticRegression -----------")
m = MyLogisticRegression()
X = np.array([[1, 3, 4], [1, -5, 6], [-3, 5, 3]])
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
y = np.array([1, 0, 1])
preds = np.array([.55, .22, .85])
print("preds : ", preds)
grads = m.get_grad(X, y, preds)
print("grads : ", grads)
assert np.allclose(grads, np.array([-0.38,  0.22, -3.2 , -0.93])), "Что-то не так!"

print(" ----- MyLogisticRegression test 2-----------")
np.random.seed(42)
m = MyLogisticRegression()
X = np.random.rand(100,3)
y = np.random.randint(0, 1, size=(100,))
preds = np.random.rand(100)
grads = m.get_grad(X, y, preds)
assert np.allclose(grads, np.array([23.8698149, 25.27049356, 24.4139452])), "Что-то не так!"

print(" ----- MyElasticLogisticRegression test 1 ----------")
me = MyElasticLogisticRegression(.2,.2)
X = np.array([[1, 3, 4], [1, -5, 6], [-3, 5, 3]])
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
y = np.array([1, 0, 1])
preds = np.array([.55, .22, .85])
me.w = np.array([1,1,1,1])
grads = me.get_grad(X, y, preds)
assert np.allclose(grads, np.array([-0.38,  0.82, -2.6 , -0.33])), "Что-то не так!"

np.random.seed(42)
m2 = MyElasticLogisticRegression(.2, .2)
X = np.random.rand(100,3)
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
y = np.random.randint(0, 1, size=(100,))
preds = np.random.rand(100)
m2.w = np.array([1,1,1,1])
grads = m2.get_grad(X, y, preds)
assert np.allclose(grads, np.array([49.11489408, 24.4698149, 25.87049356, 25.0139452])), "Что-то не так!"


X, y = make_blobs(n_samples=1000, centers=[[-2,0.5],[3,-0.5]], cluster_std=1, random_state=42)

colors = ("red", "green")
colored_y = np.zeros(y.size, dtype=str)

for i, cl in enumerate([0,1]):
    colored_y[y.ravel() == cl] = str(colors[i])

plt.figure(figsize=(15,10))
plt.scatter(X[:, 0], X[:, 1], c=colored_y)
#plt.grid()
#plt.show()

clf = MyElasticLogisticRegression(0.1, 0.1)
clf.fit(X, y, epochs=1000)
w = clf.get_weights()

plt.figure(figsize=(15,8))

eps = 0.1
xx, yy = np.meshgrid(np.linspace(np.min(X[:,0]) - eps, np.max(X[:,0]) + eps, 200),
                     np.linspace(np.min(X[:,1]) - eps, np.max(X[:,1]) + eps, 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=colored_y)
# plt.show()

#Теперь протестируем на датасете MNIST. Это очень простой класcический датасет, на котором часто тестируются модели.
# С помощью нейронных сетей люди научились получать на нем качество 99.84%.
print("------------------- MNIST -------------------------------------")
data = pd.read_csv('../data/train.csv')
print(data.head())
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Выберем только картинки, где изображен 0 и 1
X = X[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]


model = MyElasticLogisticRegression(0.2, 0.2)
model.fit(X, y)
preds = model.predict(X)
print("TEST ------------------ ", preds[:10])


pipeline = make_pipeline(
    StandardScaler(),
    MyElasticLogisticRegression(
        l1_coef=0.02,
        l2_coef=0.02
    )
)
# взрываемся на весах Fit иногда не проходит. исправление приводит к поломке всего задания выше
# sklearn.exceptions.NotFittedError: This Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

mean_accuracy = scores.mean()
print(f"Mean accuracy of Logistic Regression for two classes is {mean_accuracy:.4f}")