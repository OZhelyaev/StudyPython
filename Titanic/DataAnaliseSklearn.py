import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import kagglehub

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
# кросс валидация
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
# перебор параметров по сетке-набору (например для модели DecisionTreeRegressor)
from sklearn.model_selection import GridSearchCV
# сохраняем модель
import pickle as pkl

from math import sqrt

# считаем расстояние между векторами
def euclidean_distance(vect1, vect2):
    dist = 0.0
    for i in range(len(vect1)):
        dist += pow(vect1[i] - vect2[i], 2)
    return sqrt(dist)

# считаем расстояние между векторами через numpy
def euclidean_distance_numpy(vec1, vec2):
	return np.linalg.norm(vec1 - vec2)

# находим num_neighbors ближайших соседей
def get_neighbors_dummy(train, test_row, num_neighbors):
    distances = []
    for tr_id, tr_row in enumerate(train):
        dist = euclidean_distance(tr_row, test_row)
        distances.append((tr_id, dist))

    distances.sort(key=lambda x: x[1])
    return [i[0] for i in distances[:num_neighbors]]

def get_neighbors_numpy(train, test_row, num_neighbors):
  distances = np.linalg.norm(train - test_row, axis=1)
  nearest_neighbor_ids = distances.argsort()[:num_neighbors]
  return nearest_neighbor_ids

# находим средний возраст для тестовой выборки x_test
def predict_dummy(x_train, x_test, y_train,  num_neighbors = 3):
    y_predict = []
    for test_row in x_test:
        nearest_neighbors = get_neighbors_dummy(x_train, test_row, num_neighbors)
        y_prd = y_train[nearest_neighbors]
        y_prd = y_prd.mean()
        y_predict.append(float(y_prd))

    return y_predict

# Download latest version
# загрузка файла из kagglehub. не удачная... разобраться
# path = kagglehub.dataset_download("abrambeyer/openintro-possum", path='openintro-possum.csv', output_dir='./data')
path ='./data/possum.csv'

print("Path to dataset files:", path)
df = pd.read_csv(path)
#print(df.head())

# print(df.info())
# print(df.describe())
# удалим не значимые признаки
df.drop(columns=['case','site','Pop','sex'], inplace=True)
print(df.head())
# проверим где есть пустые значения
df.isna().sum()
# удалим пустые значения
df.dropna(inplace=True)
print(df.isna().sum())

# разделим признаки
# все признаки без возраста
x = df.drop(columns=['age']).values
# возраст - итоговое значение
y = df['age'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)
# KNN - метод поиска К-ближайших соседей.
# Шаг 1: Посчитать расстояние.
# Шаг 2: Найти ближайших соседей.
# Шаг 3: Сделать предсказание.
# print(x_train[:5])
# print(x_test[0])
print(get_neighbors_dummy(x_train[:5], x_test[0], 3))
print(get_neighbors_numpy(x_train[:5], x_test[0], 3))

assert euclidean_distance(x_train[0], x_test[0]) == euclidean_distance_numpy(x_train[0], x_test[0])
# assert get_neighbors_dummy(x_train[:5], x_test[0], 3) == get_neighbors_numpy(x_train[:5], x_test[0], 3)
# предсказываем возраст для тестовой выборки
print(' предсказываем сами: ', predict_dummy(x_train, x_test, y_train, 3))


# то же самое только с помощью библиотеки sklearn
# создание модели
# ---------------------------------------------
#model = KNeighborsRegressor(n_neighbors=3)
# обучение модели
#model.fit(x_train, y_train)
# -------------------------------------------
# деревья решений
model = DecisionTreeRegressor()
model.fit(x_train, y_train)
# предсказание
y_predict = model.predict(x_test)
print(' предсказываем с sklearn: ', y_predict)

# 3. Метрики
# Итак, мы научились объявлять модель, обучать ее и получать предсказания на тестовые элементы.
# Давайте теперь подумаем, как оценивать то, насколько хороша наша модель.
# Роль оценки моделей на себя берут метрики. Это функции, которые принимают на вход правильные ответы
# на тестовые данные и ответы модели, и выдают число — меру "хорошести" предсказаний модели.
# Для каждой задачи существуют свои метрики. В данном семинаре мы рассмотрим лишь несколько
# метрик для задачи регрессии. Более подробный обзор метрик для задач регрессии и классификации можно найти в Yandex ML Book.


# Для нашей задачи мы будем использовать метрику MSE
# MSE (Mean Squared Error, или, по русски, среднеквадратичная ошибка) — одна из самых популярных метрик в задаче регрессии.
# Иногда для того, чтобы значение метрики MSE имело масштаб значений целевой переменной,
# из него извлекают квадратный корень. Это называют метрикой RMSE (Root Mean Squared Error)
# MSE не ограничен сверху. По значению MSE может быть нелегко понять, насколько оно «хороше» или «плохое».
# Для ориентира часто берут наилучшее константное предсказание с точки зрения MSE — среднее арифметическое значений
# целевой переменной обучающей части данных. Далее можно посчитать метрику

#Верхняя граница  R2  — 1. Чем значение  R2 , тем лучше обучилась модель.

# Так как в MSE ошибка на каждом элементе возводится в квадрат, MSE сильно штрафует за большие ошибки на элементах.
# И если в наших тестовых данных присутствуют выбросы, то ошибки на этих выбросах будут вносить существенный вклад в MSE,
# и, чтобы уменьшить метрику, модель будет стараться уменьшать ошибки именно на этих на объектах-выбросах,
# а не на остальных элементах. Поэтому MSE может быть не лучшей метрикой для сравнения моделей на выборках с большим
# количеством выбросов. В таких случаях прибегают к метрике MAE (Mean Absolute Error)

# сделаем предсказания
pred_train = model.predict(x_train)
pred_test = model.predict(x_test)
# посчитаем метрики
mse_train = mean_squared_error(y_train, pred_train)
rmse_train = sqrt(mse_train)
r2_train = r2_score(y_train, pred_train)
mae_train = mean_absolute_error(y_train, pred_train)

mse_test = mean_squared_error(y_test, pred_test)
rmse_test = sqrt(mse_test)
r2_test = r2_score(y_test, pred_test)
mae_test = mean_absolute_error(y_test, pred_test)

print(f'MSE на обучении {mse_train:.2f}')
print(f'MSE на тесте {mse_test:.2f}', end='\n\n')

print(f'RMSE на обучении {rmse_train:.2f}')
print(f'RMSE на тесте {rmse_test:.2f}', end='\n\n')

print(f'R2 на обучении {r2_train:.2f}')
print(f'R2 на тесте {r2_test:.2f}', end='\n\n')

print(f'MAE на обучении {mae_train:.2f}')
print(f'MAE на тесте {mae_test:.2f}', end='\n\n')

# Переобучение и методы борьбы с ним
# 1. кросс-валидация
# используем кросс-валидацию для борьбы с переобучением, но в данном кейсе это не помогло
scores = cross_validate(DecisionTreeRegressor(random_state = 44), x_train, y_train, cv=5,
                       scoring={'r2': make_scorer(r2_score),
                                'mean_squared_error': make_scorer(mean_squared_error)}, return_train_score=True )

print('R2 train mean = ', scores['train_r2'].mean())
print('R2 test mean = ', scores['test_r2'].mean())

print('MSE train mean = ', scores['train_mean_squared_error'].mean())
print('MSE test mean = ', scores['test_mean_squared_error'].mean())

# пробуем поиграть с параметрами дерева
# Основные параметры деревьев решений, которые помогают бороться с переобучением: max_depth
# - глубина дерева min_samples_leaf
# - минимальное количество объектов в одном листе.
# - max_leaf_nodes - максимальное количество листьев.
# Дефолтные параметры
print('Дефолтные параметры ', model.get_params())
model = DecisionTreeRegressor(random_state=1, max_depth=4, min_samples_leaf=1, max_leaf_nodes=3)
print('Обновленные параметры ', model.get_params())
model.fit(x_train, y_train)
print(f'MSE train = {mean_squared_error(y_train, model.predict(x_train))}')
print(f'MSE test = {mean_squared_error(y_test, model.predict(x_test))}')

model = DecisionTreeRegressor()
param_grid = {
    'max_depth': np.arange(1, 5), # эквивалентно [1,2,3,4]
    'min_samples_leaf': [1,2,3],
}
# refit=True - поиск по сетке всех, оцениваем по r2_score
gridsearch = GridSearchCV(model, param_grid, refit=True, scoring=make_scorer(r2_score))
# запустим поиск
gridsearch.fit(x_train, y_train)
print(gridsearch.best_params_)

# 5. Сохранение модели через pickle
filename = './model/model1.pkl'
pkl.dump(model, open(filename, 'wb'))

lmodel = pkl.load(open(filename, 'rb'))
print(model.get_params())