import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import axis
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# метод разделения данных на train и test
from sklearn.model_selection import train_test_split
# методы для масштабирования
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


df =pd.read_csv('titanic.csv', index_col='PassengerId')

# print(df['Survived'].value_counts())

#df['Survived'].hist()
#plt.xlabel('Survived')
#plt.ylabel('count')
#plt.suptitle('Distribution of Survived')
#plt.show()
# Анализ признаков:
# Категориальные - результат относится к какой-либо категории
# Основные методы кодирования категориальных признаков:
#   - one-hot encoding
#   - label encoding
# Количественные - число
# -----------------------------------------------------------
# Вычленим все значения Mr. Miss. в отдельный DataSet
# extract(' ([A-Za-z]+)\.' - регулярное выражение
# expand=False - возращает Dataframe
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
# выводим пересечение полученых статусов с колонкой Sex-пол

# Заменяем все редкие статусы на Rare и сужаем различные Mlle, Mme до Miss
df['Title'] = df['Title'].replace(['Rev', 'Sir', 'Capt', 'Col', 'Countess', 'Dr', 'Major', 'Master', 'Jonkheer'], 'Rare')
df['Title'] = df['Title'].replace(['Lady', 'Miss', 'Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
df['Title'] = df['Title'].replace(['Don'], 'Mr')

# удаляем не анализируемые колонки
df.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace=True)
# вычисляем  медиану для замены Nan
median_Age = np.quantile(df['Age'].dropna().values, q=0.5)
# заполнили все Nan значения медианой
df.loc[df['Age'].isna(), 'Age'] = median_Age
# ----------------------------------------------------------
print(df.columns)
# создаем список категориальных признаков
categorial_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title']
# создаем новый DF
categorial_features = df[categorial_columns]
#print(categorial_features.head(10))
#print(categorial_features.info())
unprocessed_categorial_features = categorial_features.select_dtypes(include=['str']).columns.tolist()
# one-hot encoding создаем из одномерного массива матрицу, строки ['Sex', 'Embarked', 'Title'] будут теперь колонками
print(df[unprocessed_categorial_features])
# one-hot encoding with pandas
#    one_hot = pd.get_dummies(df['Title'])
#    df_one_hot = pd.concat([df, one_hot], axis=1)
#    df_one_hot.drop(columns=['Title'], inplace=True)
#    print(df_one_hot.head())
# --------------------------------------------------------------------
# one-hot encoding with sklearn
encoder = OneHotEncoder()
# проводим трансформацию всего dataset в 0 и 1
one_hot_enc = encoder.fit_transform(df[unprocessed_categorial_features]).toarray()
# создаем новый dataset в котором имена колонок   Title_Mrs Title_Rare и значения по строкам 1 и 0
one_hot_df = pd.DataFrame(one_hot_enc, columns=encoder.get_feature_names_out(unprocessed_categorial_features), index=df.index)
df_one_hot = pd.concat([df, one_hot_df], axis=1)
# удаляем обработанные колонки
df_one_hot.drop(columns=unprocessed_categorial_features, inplace=True)
# print(df_one_hot.head())
# --------------------------------------------------------------------
# Label encoding - каждому уникальному значению присваиваем уникальный код 1,2,3...
# некоторые модели могут считать что в значениях есть порядок
# label encoding with sklearn
encoder = LabelEncoder()
for col in unprocessed_categorial_features:
    df[col] = encoder.fit_transform(df[col])
# проверяем - выводим уникальные значения которые мы получили после кодирования в столбце Title
# print(df.Title.unique())

# ---------------------------------------------------------------------
# выделяем количественные признаки
numeric_columns = ['Age', 'Fare']
num_features = df[numeric_columns]
print(num_features)
# оцениваем распределение значений признаков. Видим, что Age имеет нормальное расп., а Fare смещен к 0
#num_features.hist(figsize=(10, 5), bins=20, xlabelsize=10, ylabelsize=10)
# plt.show()
# строим взаимные графики на куче признаков для оценки распределения
# оцениваем зависимости визуально, если у признаков сильная корреляция, то с такими признаками модели плохо работают
# и их нужно либу удалять, либо объединять
#sns.pairplot(df[numeric_columns], height=3, kind='scatter', diag_kind='kde')

# построим матрицу корреляций признаков (мера линейной зависимости признаков)
k = 10
corrmat = df.corr()
cols = corrmat.nlargest(k, ['Survived'])['Survived'].index
cm = np.corrcoef(df[cols].values.T)
sns.set_theme(font_scale=1.25)
#sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
# print(df.columns)
# plt.show()
# ----------------------------------------------------------------------
# далее нужно определить как признак влияет на конечную переменную
data = df['Survived'].value_counts()
# посчитаем для класса 1 сколько человек выжило, а сколько нет
data1 = df.loc[(df.Pclass == 1), ['Survived']].value_counts()
data2 = df.loc[(df.Pclass == 2), ['Survived']].value_counts()
data3 = df.loc[(df.Pclass == 3), ['Survived']].value_counts()
# print(df.loc[(df.Pclass == 1), ['Survived']].value_counts())
fig, (ax1, ax2, ax3 )= plt.subplots(1,3, figsize=(10, 8))
ax1.pie(data1.values, labels=data1.index, autopct='%1.1f%%')
ax2.pie(data2.values, labels=data2.index, autopct='%1.1f%%')
ax3.pie(data3.values, labels=data3.index, autopct='%1.1f%%')
ax1.set_title('Pclass = 1')
ax2.set_title('Pclass = 2')
ax3.set_title('Pclass = 3')
#  plt.show()
# --------------------------------------------------------------------
# посмотрим зависимость выживаемости от возраста
plt.figure(figsize=(7,5))
plt.hist(df.loc[(df.Survived == 1), ['Age']], alpha=0.5, label='Survived')
plt.hist(df.loc[(df.Survived == 0), ['Age']], alpha=0.5, label='Not Survived')
plt.legend()
# plt.show()
# --------------------------------------------------------------------
# посмотрим зависимость выживаемости от пола
data1 = df.loc[(df.Sex == 0), ['Survived']].value_counts()
data2 = df.loc[(df.Sex == 1), ['Survived']].value_counts()

# print(df.loc[(df.Pclass == 1), ['Survived']].value_counts())
fig, (ax1, ax2)= plt.subplots(1,2, figsize=(10, 8))
ax1.pie(data1.values, labels=data1.index, autopct='%1.1f%%')
ax2.pie(data2.values, labels=data2.index, autopct='%1.1f%%')
ax1.set_title('Female')
ax2.set_title('Male')

# -------------------------------------------------------------------
# влияет ли стоимость билета на выживаемость
print(df['Fare'])
data1 = df.loc[df.Survived == 1]['Fare']
data2 = df.loc[df.Survived == 0]['Fare']
# строим boxplot отображает вариативнось признака и размах в квартиле
# q=0.25 - первый квартиль q=0.75 - третий квартиль
plt.figure(figsize=(7,5))
plt.boxplot([data1, data2], showfliers=False )
plt.title('Влияние стоимости билета на выживание')
plt.xlabel('Выжили')
plt.ylabel('Стоимость билета')
plt.xticks([1,2], ['Нет', 'Да'])

fq = np.quantile(df['Fare'].values, q=0.25)
print(f'first quartile = {fq}')
tq = np.quantile(df['Fare'].values, q=0.75)
print(f'third quartile = {tq}')
# plt.show()
# нужно обращать внимание на влияние признака на конечную переменну (зависимость)
# нужно оценить независимость признаков

# ---------------------------------------------------------------------
# подготовим данные для модели
# 1. разделим данные на train (обучающую) и test (тестовую) часть с помощью sklern
print(df.head(3))
x = df.drop(columns=['Survived']).values # только значения без данных о выживании
y = df['Survived'].values # выделим целевые значения
# делим данные, 20% на тест, shuffle=True - каждый раз перемешиваем данные
x_tr, x_test,  y_tr, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

# масштабирование данных - необходимо привести все данные к одному масштабу
# данные от 0-1 и от 0-1000 нужно приводить к одному масштабу
# 1. масштабирования - расчет стандартного распределения StandartScaler
# 2. MinMaxScaler - масштабирование от минимального до максимального

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_tr) # передаем только данные для обучения!
x_tr_scaled = scaler.transform(x_tr)
x_test_scaled = scaler.transform(x_test)

print(f'До масштабирования {x_tr.mean(axis=0)}')
print(f'После масштабирования {x_tr_scaled.mean(axis=0)}')
