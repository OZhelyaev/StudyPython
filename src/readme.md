# Python Data Science Libraries

---

# Pandas

pandas

## Что это

`Pandas` — библиотека для работы с таблицами и анализа данных.

Используется для:

* CSV / Excel
* фильтрации данных
* аналитики
* обработки таблиц
* статистики

---

## Установка

```bash
pip install pandas
```

---

## Импорт

```python
import pandas as pd
```

---

# Основные структуры

## Series

Одномерный массив:

```python
s = pd.Series([1, 2, 3])
```

---

## DataFrame

Таблица:

```python
data = {
    "name": ["Alex", "John"],
    "age": [25, 30]
}

df = pd.DataFrame(data)
```

---

# Чтение файлов

## CSV

```python
df = pd.read_csv("data.csv")
```

## Excel

```python
df = pd.read_excel("data.xlsx")
```

---

# Просмотр данных

## Первые строки

```python
df.head()
```

## Последние строки

```python
df.tail()
```

## Информация о таблице

```python
df.info()
```

## Размер таблицы

```python
df.shape
```

---

# Работа со столбцами

## Получить столбец

```python
df["name"]
```

## Несколько столбцов

```python
df[["name", "age"]]
```

## Новый столбец

```python
df["salary"] = [1000, 2000]
```

---

# Фильтрация

```python
df[df["age"] > 25]
```

---

# Сортировка

```python
df.sort_values("age")
```

---

# Группировка

```python
df.groupby("city").mean()
```

---

# Удаление пустых значений

```python
df.dropna()
```

---

# Сохранение

## CSV

```python
df.to_csv("result.csv")
```

## Excel

```python
df.to_excel("result.xlsx")
```

---

# Matplotlib

Matplotlib

## Что это

`Matplotlib` — библиотека для построения графиков и визуализации данных.

Используется для:

* графиков
* диаграмм
* аналитики
* визуализации ML

---

# Установка

```bash
pip install matplotlib
```

---

# Импорт

```python
import matplotlib.pyplot as plt
```

---

# Линейный график

```python
x = [1, 2, 3]
y = [10, 20, 30]

plt.plot(x, y)
plt.show()
```

---

# Заголовок и подписи

```python
plt.title("Sales")
plt.xlabel("Month")
plt.ylabel("Money")
```

---

# Точечный график

```python
plt.scatter(x, y)
```

---

# Столбчатая диаграмма

```python
plt.bar(x, y)
```

---

# Гистограмма

```python
plt.hist(y)
```

---

# Несколько графиков

```python
plt.plot(x, y)
plt.plot(x, [5, 15, 25])
```

---

# Сохранение графика

```python
plt.savefig("chart.png")
```

---

# Sklearn

scikit-learn

## Что это

`Scikit-learn (sklearn)` — библиотека машинного обучения.

Используется для:

* классификации
* регрессии
* кластеризации
* обучения моделей
* оценки ML

---

# Установка

```bash
pip install scikit-learn
```

---

# Импорт

```python
from sklearn.model_selection import train_test_split
```

---

# Разделение данных

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
```

---

# Линейная регрессия

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Оценка модели

```python
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, predictions)
```

---

# Классификация

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)
```

---

# Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)
```

---

# Нормализация данных

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# KMeans кластеризация

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)

model.fit(X)
```

---

# Полезные модули sklearn

| Модуль        | Назначение             |
| ------------- | ---------------------- |
| linear_model  | Регрессия              |
| tree          | Decision Tree          |
| svm           | Support Vector Machine |
| cluster       | Кластеризация          |
| metrics       | Метрики                |
| preprocessing | Подготовка данных      |

---

# Часто используемый стек

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

---

# Типичный pipeline

```text
Pandas - анализ данных 
↓
Очистка данных - незначимые признаки, сильная корреляция между признаками
↓
Matplotlib - визуальный анализ 
↓
Разделить данные: train_test_split(..., stratify=y)
↓
Обработать пропуски: fit на train → transform train и test
↓
Закодировать категориальные признаки: fit на train → transform train и test
↓
Нормировать числовые признаки: fit на train → transform train и test
↓
Обучать модель только на train, оценивать — только на test
↓
Оценка 
```
