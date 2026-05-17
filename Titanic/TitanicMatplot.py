# библиотека для визуализации данных
from cProfile import label

import matplotlib.pyplot as plt
import numpy as np


x = np.arange(10)
y = np.random.randint(1,10, size=10)
z = np.random.randint(1,10, size=10)
print(f'x = {x}')
print(f'y = {y}')
print(f'z = {z}')
# размер графика
plt.figure(figsize=(10,7))
# линейный график с установкой названий данных и цветом
# plt.plot(x, y, color = 'blue', label = 'y')
# plt.plot(y, z, color = 'red', label = 'z')

plt.style.use('fivethirtyeight')
plt.plot(x, y, label = 'y')
plt.plot(y, z, label = 'z')

# точечный график
plt.scatter(x, y)
plt.scatter(y, z)
# установка пределов по осям
plt.xlim([-5, 20])
plt.ylim([0, 10])
# отрисовываем сетку
plt.grid()
# название графика
plt.title("График зависимости x от y")
plt.xlabel(' время')
plt.ylabel(' данные')
# выводим легенду на график
plt.legend()
plt.show()