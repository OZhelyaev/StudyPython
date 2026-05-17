import pandas as pd


a = [1,4,5,6,7,8,9,10,11,12]
# создание да Series - последовательности
series = pd.Series(a, index = ['a','b','c','d','e','f','g','h','i','j'])
# получение данных по индексу
print(series.loc['f'])
print(series.iloc[4])

# DataFrame - таблица данных
data = {
    'calories' : [402, 380, 600],
    'duration' : [50, 30, 40]
}
print('-'*30)
print(data)
print('-'*30)
df = pd.DataFrame(data)

print(df['calories']) # обращение к столбцу
print(df.iloc[2]) # обращение к 2-му столбцу
# результат
# calories    600
# duration     40
print('-'*30)
# создаем dataframe из файла  titanic.csv, за index возьмем первую колонку PassengerId
df =pd.read_csv('titanic.csv', index_col='PassengerId')

print(df.head(10)) # выводим первые 10 строк
# print(df.sample(10)) # выводим случайные 10 строк
# print(df.tail(10)) # выводим последние 10 строк
# print(df.loc[10:12]) # выводим последние 10 строк
# print(df.iloc[9:12]) # выводим последние 10 строк
# print(df.iloc[[9,11]]) # выводим последние 9 и 11 строки, передавая параметр массив
print(df.columns) # выводим наименования всех колонок
# print(df.loc[9:12, ['Pclass','Name']]) # выводим значение по строкам  9 до 12 из колонок Pclass и Name
# print(df.loc[9:12, 'Survived':'Name']) # выводим значение по строкам  9 до 12 из колонок от Survived до Name
print(df.loc[(df.Survived == 1) | (df.Pclass == 3)]) # выводим значение строк в которых Survived == 1 или Pclass == 3
print(df.loc[(df.Survived == 1) & (df.Pclass == 3)]) # выводим значение строк в которых Survived == 1 и Pclass == 3

# изменение знечение
# df.loc[(df.Survived == 1), 'Survived'] = 1
# print(df.head(10))

# получение общей иннформации по dataFrame
# RangeIndex: 891 entries - общее количество строк
#  4   Age       714 non-null    float64 - 714 строк из 891 заполнены - т.е. пропуски в данных
# print(df.info())

# ------------------------------------------- Предобработка данных ------------------------------------

# вывод основных статистических данных по числовым столбцам
# print(df.describe())
# выводим в ячейках True или False если значение не пустое или пустое соответственно
# print(df.isna())
# суммируем все False по оси axis. 0 - по всем столбцам, 1 по строке 1
print(df.isna().sum(axis=1))
