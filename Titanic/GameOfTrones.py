# В данном задании вы будете работать с датасетом о персонажах из вселенной Игры Престолов
# A Wiki of Ice and Fire.
# Вам предстоит предсказать, кто из персонажей умрет, а кто останется в живых.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import RandomForestClassifier
# from sklearn.linear_model import AdaBoostClassifier
#from sklearn.linear_model import GaussianProcessClassifier
#from sklearn.linear_model import GaussianNB
#from sklearn.linear_model import KNeighborsClassifier
#from sklearn.linear_model import SVC
#from sklearn.linear_model import DecisionTreeClassifier

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score
from math import sqrt

data = pd.read_csv('../data/game_of_thrones_train.csv', index_col='S.No')
# print(data.head())
print(data.describe(include = 'str').T)
print('-' * 30)
print(data.describe(include = ['int64','float64']).T[['count', 'min', 'max']])
print('-' * 30)
print(data.isna().sum(axis=0))
print('-' * 30, ' num columns scale')
num_columns = data.select_dtypes(include=['int64','float64']).columns.tolist()
print(num_columns)

print(data[num_columns].head())
# ------------------------  popularity  ---------------------------------------------------------------
# У нас есть признак popularity. Распределение сильно несимметрично. Можно преобразовать данный признак,
# но лучше используем Квантильное преобразование для получения равномерного распределения
print('-' * 30)
qt_normal = QuantileTransformer(output_distribution='uniform')
data['popularity'] = qt_normal.fit_transform(data[['popularity']])
print(data[num_columns].head())
print('-' * 30, ' popularity hist')
##plt.figure(figsize = (10, 5))

data['popularity'].hist(density=False, bins=50)
plt.ylabel('count')
plt.xlabel('popularity')
plt.suptitle('Distribution of popularity')
## plt.show()
# ------------------------  boolDeadRelations  ---------------------------------------------------------------
# Создайте признак boolDeadRelations. Давайте упростим признак numDeadRelations,
# и просто поделим людей на тех, у кого были хоть какие-то отношения с мертвыми персонажами,
# т.е. numDeadRelations > 0, и те, у которых не было, т.е. numDeadRelations = 0.
print('-' * 30, ' boolDeadRelations hist')
data['boolDeadRelations'] = np.where(data['numDeadRelations'] > 0, 1, 0)

data['boolDeadRelations'].hist(density=False, bins=50)
plt.ylabel('count')
plt.xlabel('boolDeadRelations')

plt.suptitle('Distribution of boolDeadRelations')
## plt.show()
data = data.drop(columns=['numDeadRelations'])
# ------------------------  Age, dateOfBirth  ---------------------------------------------------------------
#Признак Age нем очень много пропущенных значений. Для того чтобы использовать в модели
# информацию о возрасте персонажа, мы создадим два новых признака: age_value и age_no_data
# Там где возраст указан, age_value принимает значение age, а age_no_data - значение 0.
# Там где возраст не указан, age_value принимает значение 0, а age_no_data - значение 1.
print('-' * 30)
data['age_value'] = [x if not np.isnan(x) else 0 for x in data['age']]
data['age_no_data'] = [1 if np.isnan(x) else 0 for x in data['age']]
# таким же образом преобразовываем признак dateOfBirth. У одних и тех же персонажей не
# указан и возраст, и год рождения
data['dateOfBirth_no_data'] = [1 if np.isnan(x) else 0 for x in data['dateOfBirth']]

# трансформируем age
data['age_value'] = qt_normal.fit_transform(data[['age_value']])
print('-' * 30, ' age_value hist')
data['age_value'].hist(density=False, bins=50)
plt.ylabel('count')
plt.xlabel('age_value')
plt.suptitle('Distribution of age_value')
##plt.figure(figsize = (10, 5))
## plt.show()

print(data[['age', 'age_value', 'age_no_data', 'dateOfBirth', 'dateOfBirth_no_data']].head(10))
print('-' * 30)
data = data.drop(columns=['age', 'dateOfBirth'])
# ------------------------  culture  ---------------------------------------------------------------
# Задание 1.5. Категориальные признаки с большим количеством категорий
# Признак culture содержит информацию о принадлежности к одному из народов во вселенной Игры Престолов.

## print(data['culture'].value_counts(dropna=False))

# для большого числа персонажей значения данного признака не указаны.
# Также есть много редких значений признака. Данную проблему мы попытаемся решить,
# сгруппировав народы в более крупные категории
cultures_grouped = {
    'Old Nations': ['valyrian', 'first men', 'andal', 'andals', 'rhoynar'],
    'the North': ['northmen', 'northern mountain clans', 'crannogmen'],
    'the Iron Islands': ['ironborn', 'ironborn', 'ironmen'],
    'the Mountain and the Vale': ['valemen', 'vale', 'vale mountain clans',
                              'sistermen'],
    'the Isles and Rivers': ['riverlands', 'rivermen'],
    'the Rock': ['westerman', 'westermen', 'westerlands'],
#    'the Stormlands': ['stormlander', 'stormlands'], # включили в Other Nations из-замалого количества
    'the Reach': ['reach', 'reachmen', 'the reach'],
    'Dorne': ['dornish', 'dornishmen', 'dorne'],
    'Essos Nations': ['astapor', 'astapori', 'braavosi', 'braavos', 'tyroshi', 'lysene', 'lyseni',
                      'myrish', 'pentoshi', 'qartheen', 'qarth', 'dothraki',
                      'lhazarene', 'lhazareen','meereen', 'meereenese',
                      'norvoshi', 'qohor', 'summer isles', 'summer islands',
                      'summer islander', 'asshai', "asshai'i", 'norvos', 'ghiscari',
                      'ghiscaricari'],
    'Other Nations': ['ibbenese', 'westeros', 'free folk', 'wildling', 'wildlings', 'naathi', 'stormlander', 'stormlands']}

# Инвертируем словарь cultures_grouped
cultures_grouped_inverted = {}
for k in cultures_grouped.keys():
  for v in cultures_grouped[k]:
      cultures_grouped_inverted.update({v:k})
## print(cultures_grouped_inverted)

# Теперь создадим новый столбец с укрупненными значениями culture.
data['culture_grouped'] = data['culture'].str.lower().map(cultures_grouped_inverted)
# Заменим все NaN в созданном столбце на категорию culture_no_data:
data['culture_grouped'] = data['culture_grouped'].fillna('culture_no_data')
print(data['culture_grouped'].value_counts(dropna=False))
data = data.drop(columns=['culture'])
# ------------------------  title  ---------------------------------------------------------------
# группируем tittle в 10 с самой большой частотой повторения
top_titles = data['title'].value_counts().nlargest(10).index
print("top_titles : ", top_titles)
data['top_titles'] = data['title'].where(
    data['title'].isin(top_titles),
    'Other'
)
data = data.drop(columns=['title'])
# ------------------------  house  ---------------------------------------------------------------
# проводим частотное преобразование, предварительно заменив Nan на Missing
data['house'] = data['house'].fillna('Missing')
house_freq = data['house'].value_counts(normalize=True)
print("house_freq : ", house_freq)
data['house_freq'] = data['house'].map(house_freq)
print(data['house_freq'].value_counts(dropna=False))
data = data.drop(columns=['house'])
# ------------------------  mother, father, heir, spouse -------------------------------------------
# заменяем имя родственника на бинарное отношение есть или нет
data['has_mother'] = data['mother'].notna().astype(int)
print('has_mother : ', data['has_mother'].value_counts(dropna=False))
data['has_father'] = data['father'].notna().astype(int)
data['has_heir'] = data['heir'].notna().astype(int)
data['has_spouse'] = data['spouse'].notna().astype(int)
print('has_spouse : ', data['has_spouse'].value_counts(dropna=False))
data = data.drop(columns=['mother', 'father', 'heir', 'spouse'])

# 1.6. Категориальные признаки в линейных моделях.
# Для включения категориальных признаков в линейную модель их нужно преобразовать в числовые признаки
# Для того, чтобы найти все порядковые признаки, посмотрим на количество уникальных значений, которые встречаются в столбцах.
print(data.nunique())
# Для числовых столбцов можно вывести в одну таблицу более детальную статистику, объединив выводы describe() и nunique()
# Код ниже требуется дополнить по аналогии с заданием 1.2.
print(data.describe(include = ['int64','float64']).T[['count', 'min', 'max']].assign(N_unique_values = data.nunique()))
#unprocessed_categorial_features = categorial_features.select_dtypes(include=['str']).columns.tolist()
print(data.describe(include = ['str']).T[['count']].assign(N_unique_values = data.nunique()))
# ------------------------  isAliveSpouse  ---------------------------------------------------------------
# сравнение средних зависимой пременной isAlive для признака isAliveSpouse
data.groupby('isAliveSpouse', dropna = False)['isAlive'].mean()
print("isAliveSpouse <->  isAlive")
print(pd.pivot_table(data = data, values = 'isAlive', index = 'isAliveSpouse', aggfunc=['mean', 'count'], dropna=False))
# высокая корреляция - оставляем
# ------------------------  isAliveMother  ---------------------------------------------------------------
# сравнение средних зависимой пременной isAlive для признака isAliveMother
data.groupby('isAliveMother', dropna = False)['isAlive'].mean()
print("isAliveMother <->  isAlive")
print(pd.pivot_table(data = data, values = 'isAlive', index = 'isAliveMother', aggfunc=['mean', 'count'], dropna=False))
# низкая корреляция - удаляем
# ------------------------  isAliveFather  ---------------------------------------------------------------
# сравнение средних зависимой пременной isAlive для признака isAliveFather
data.groupby('isAliveFather', dropna = False)['isAlive'].mean()
print("isAliveFather <->  isAlive")
print(pd.pivot_table(data = data, values = 'isAlive', index = 'isAliveFather', aggfunc=['mean', 'count'], dropna=False))
# низкая корреляция - удаляем
# ------------------------  isAliveHeir  ---------------------------------------------------------------
# сравнение средних зависимой пременной isAlive для признака isAliveHeir
data.groupby('isAliveHeir', dropna = False)['isAlive'].mean()
print("isAliveHeir <->  isAlive")
print(pd.pivot_table(data = data, values = 'isAlive', index = 'isAliveHeir', aggfunc=['mean', 'count'], dropna=False))
# низкая корреляция - удаляем
# ------------------------  isMarried  ---------------------------------------------------------------
# сравнение средних зависимой пременной isAlive для признака isMarried
data.groupby('isMarried', dropna = False)['isAlive'].mean()
print("isMarried <->  isAlive")
print(pd.pivot_table(data = data, values = 'isAlive', index = 'isMarried', aggfunc=['mean', 'count'], dropna=False))
# высокая корреляция - оставляем
# ------------------------  isNoble  ---------------------------------------------------------------
# сравнение средних зависимой пременной isAlive для признака isNoble
data.groupby('isNoble', dropna = False)['isAlive'].mean()
print("isNoble <->  isAlive")
print(pd.pivot_table(data = data, values = 'isAlive', index = 'isNoble', aggfunc=['mean', 'count'], dropna=False))
# высокая корреляция - оставляем
# удаляем признаки с низкой корреляцией, в т.ч. name == ID
data = data.drop(columns=['name', 'isAliveMother', 'isAliveFather', 'isAliveHeir'])

# проводим one-hot преобразование
numeric_columns = ['culture_grouped','top_titles', 'isAliveSpouse', 'isMarried', 'isNoble']
# one-hot encoding with sklearn
encoder = OneHotEncoder()
# проводим трансформацию всего dataset в 0 и 1
one_hot_enc = encoder.fit_transform(data[numeric_columns]).toarray()
# создаем новый dataset в котором имена колонок заменяются на значения по строкам 1 и 0
one_hot_df = pd.DataFrame(one_hot_enc, columns=encoder.get_feature_names_out(numeric_columns), index=data.index)
df_one_hot = pd.concat([data, one_hot_df], axis=1)
# удаляем обработанные колонки
df_one_hot.drop(columns=numeric_columns, inplace=True)
print(one_hot_df.head())
print('data.nunique() --------------------------------')
print(data.nunique())
print('df_one_hot.nunique() --------------------------')
print(df_one_hot.nunique())
# построим матрицу корреляций признаков (мера линейной зависимости признаков)
k = 10
## plt.figure(figsize = (15, 15))
corrmat = df_one_hot.corr()
cols = corrmat.nlargest(k, ['isAlive'])['isAlive'].index
cm = np.corrcoef(df_one_hot[cols].values.T)
sb.set_theme(font_scale=1.25)
sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()
# перед обучением модели проверим, что не осталось Nan значений
print('------------- check for Nan ---------------')
print(df_one_hot.isna().sum())

# 1.10. Разделим датасет на обучающую и валидационные части (train и val) при помощи функции train_test_split
x = df_one_hot.drop(columns=['isAlive']).values # только значения без данных о выживании
y = df_one_hot['isAlive'].values # выделим целевые значения
x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

# Шаг 1. создание модели
model = LogisticRegression()
#from sklearn.linear_model import RandomForestClassifier
#from sklearn.linear_model import AdaBoostClassifier
#from sklearn.linear_model import GaussianProcessClassifier
#from sklearn.linear_model import GaussianNB
#from sklearn.linear_model import KNeighborsClassifier
#from sklearn.linear_model import SVC
#from sklearn.linear_model import DecisionTreeClassifier

# Шаг 2. обучение модели
model.fit(x_tr, y_tr)
# Шаг 3. Предсказание на обучающих и тестовых данных
pred_train = model.predict(x_tr)
print(' предсказываем с sklearn pred_train: ', pred_train)
pred_test = model.predict(x_test)
print(' предсказываем с sklearn pred_test: ', pred_test)

# посчитаем метрики
mse_train = mean_squared_error(y_tr, pred_train)
rmse_train = sqrt(mse_train)
r2_train = r2_score(y_tr, pred_train)
mae_train = mean_absolute_error(y_tr, pred_train)

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

accuracy = accuracy_score(y_tr, pred_train)
print("train Accuracy : %.4f" % accuracy)
accuracy = accuracy_score(y_test, pred_test)
print("test Accuracy : %.4f" % accuracy)

# 5. Сохранение модели через pickle
filename = './model/model_got .pkl'
pkl.dump(model, open(filename, 'wb'))

lmodel = pkl.load(open(filename, 'rb'))
print(model.get_params())
# test fo commit