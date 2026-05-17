import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Seaborn базируется на библиотеке Matplotlib,
# поэтому для отображения графиков требуется импорт Matplotlib и вызов функции plt.show()
import seaborn as sns
from pandas.core.interchange import column

# создаем dataframe из файла  titanic.csv, за index возьмем первую колонку PassengerId
df =pd.read_csv('titanic.csv', index_col='PassengerId')
#print(df.head())

# ------------------------------------------- Предобработка данных ------------------------------------

# вывод основных статистических данных по числовым столбцам
# print(df.describe())
# выводим в ячейках True или False если значение не пустое или пустое соответственно
# print(df.isna())
# суммируем все False по оси axis. 0 - по всем столбцам, 1 по строке 1
# print(df.isna().sum(axis=0))
data = df.isna().sum(axis=0)
# размер графика
plt.figure(figsize=(10,7))
# отображаем график баров с количеством Nan -данных
# plt.barh(data.index,data.values)
# plt.show()

# array = np.random.randint(low=0, high=100, size=(4,4))
# построение тепловой карты, для каждого элемента отображается его величина цветом,
# чем больше тем темнее

#sns.heatmap(df.isna().transpose())


na_persentage = round((df['Cabin'].isna().sum(axis=0) / len(df['Cabin'])) * 100, 3)
print(na_persentage)

# сколько раз каждое уникальное значение встречается в столбце
print(df['Embarked'].value_counts())
# выводим самое часто встречающееся значение
print(df['Embarked'].mode())
# удаляем все NaN значения,  inplace=True - значит удалить в DataSet-е
# !!! изменяет DataSet
df.dropna(inplace=True)

# Пропуски можно заполнять медианными или средними значениями, особенно когда мало данных
# вычисляем среднее
mean_Age = np.mean(df['Age'].values)
# вычисляем медианное значение колонки, после удаления NaN значений, без изменения DataSet
median_Age = np.quantile(df['Age'].dropna().values, q=0.5)
print(f'Среднее Age = {mean_Age}')
print(f'Медиана Age = {median_Age}')

#plt.hist(df['Age'])
#plt.axvline(x=mean_Age, color='red', linestyle='dashed', label='mean_Age')
#plt.axvline(x=median_Age, color='black', linestyle='-.', label='median_Age')

# Вычленим все значения Mr. Miss. в отдельный DataSet
# extract(' ([A-Za-z]+)\.' - регулярное выражение
# expand=False - возращает Dataframe
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
# выводим пересечение полученых статусов с колонкой Sex-пол

# Заменяем все редкие статусы на Rare и сужаем различные Mlle, Mme до Miss
df['Title'] = df['Title'].replace(['Sir', 'Capt', 'Col', 'Countess', 'Dr', 'Major', 'Lady', 'Master'], 'Rare')
df['Title'] = df['Title'].replace(['Miss', 'Mlle'], 'Miss')
df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
print(pd.crosstab(df['Title'], df['Sex']))
# вычисляем корреляцию выживаемости по полу
print(df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
#plt.legend()

# 1. создание новых признаков нужно для объединения или удаления излишней информации
# 2. часть признаков не влияет на результат их так же удаляем
# удаляем bp Dataset не значимые данные inplace=True - изменяет DF 
df.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace=True)
print(df.head(10))

# data = df['Embarked'].value_counts()
# plt.bar(data.index, data.values)
# plt.show()