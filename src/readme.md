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

---

# Кросвалидация cross-validation
 Способ более надёжно оценить качество модели, чем просто одно обучение и один прогон на тесте.
 
 Кросс-валидация даёт более честную и стабильную оценку, потому что модель проверяется
 на 5 разных "тестовых" кусках данных, а не на одном.
```python
scores = cross_val_score(
    pipeline,
    X_tr,
    y_tr,
    cv=5,
    scoring='accuracy'
)
```
X_tr, y_tr разбиваются на 5 частей (folds) — так как cv=5
Далее в цикле 5 раз:

Берутся 4 части как обучающая выборка, 1 часть — как валидационная
Pipeline заново обучается с нуля (fit) на этих 4 частях
Считается accuracy на оставшейся 1 части

В итоге получаем массив из 5 чисел — точность на каждом из 5 разбиений:

---

# Разбиении данных на фолды

```python
from sklearn.model_selection import KFold

# Обычный KFold просто режет данные на 5 частей подряд, без учёта меток классов
cv = KFold(n_splits=5)

# лучше использовать
# При разбиении на 5 фолдов он явно следит за распределением y и гарантирует, что в каждом из 5 фолдов сохранится 
# то же соотношение классов, что и в целом по датасету
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
n_splits=5 — количество фолдов (частей), на которые делится выборка. Каждая часть по очереди становится валидационной, остальные 4 — обучающими

shuffle=True — перемешивает данные перед разбиением на фолды. Важно, если в исходных данных есть скрытая упорядоченность (например, данные отсортированы по дате или по какому-то признаку) — без перемешивания фолды могут получиться систематически смещёнными

random_state=42 — фиксирует "случайность" перемешивания, чтобы результат был воспроизводимым

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
