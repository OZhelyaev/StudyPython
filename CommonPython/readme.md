# Основные команды python

## Запуск Python

```bash
python
python3
````

Запуск файла:

```bash
python main.py
```

---

# Переменные

```python
name = "Alex"
age = 25
height = 1.82
is_student = True
```

Типы данных:

```python
str     # строка
int     # целое число
float   # дробное число
bool    # True / False
```

Проверка типа:

```python
print(type(name))
```

---

# Строки (Strings)

Создание строки:

```python
text = "Hello"
```

Конкатенация:

```python
first = "Hello"
second = "World"

print(first + " " + second)
```

Длина строки:

```python
len(text)
```

Доступ к символам:

```python
text[0]
text[-1]
```

Срезы:

```python
text[0:3]
text[2:]
```

Методы строк:

```python
text.upper()
text.lower()
text.replace("H", "J")
text.split(" ")
```

Форматирование:

```python
name = "Alex"
age = 25

print(f"My name is {name}, age {age}")
```

---

# Списки (Lists)

Создание списка:

```python
numbers = [1, 2, 3, 4]
```

Доступ:

```python
numbers[0]
numbers[-1]
```

Добавление:

```python
numbers.append(5)
```

Удаление:

```python
numbers.remove(2)
numbers.pop()
```

Длина списка:

```python
len(numbers)
```

Перебор:

```python
for n in numbers:
    print(n)
```

Срезы:

```python
numbers[1:3]
```

---

# Кортежи (Tuple)

```python
point = (10, 20)
```

---

# Словари (Dictionary)

Создание:

```python
user = {
    "name": "Alex",
    "age": 25
}
```

Получение значения:

```python
user["name"]
```

Добавление:

```python
user["city"] = "Astana"
```

Перебор:

```python
for key, value in user.items():
    print(key, value)
```

---

# Условия

```python
age = 18

if age >= 18:
    print("Adult")
else:
    print("Child")
```

---

# Циклы

## for

```python
for i in range(5):
    print(i)
```

## while

```python
count = 0

while count < 5:
    print(count)
    count += 1
```

---

# Функции

```python
def greet(name):
    return f"Hello {name}"

print(greet("Alex"))
```

---

# Работа с файлами

Запись:

```python
with open("test.txt", "w") as file:
    file.write("Hello")
```

Чтение:

```python
with open("test.txt", "r") as file:
    data = file.read()
    print(data)
```

---

# Исключения (try/except)

```python
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Division by zero")
```

---

# Импорт модулей

```python
import math

print(math.sqrt(16))
```

Импорт функции:

```python
from math import sqrt
```

---

# Классы

```python
class User:
    def __init__(self, name):
        self.name = name

    def hello(self):
        print(f"Hello {self.name}")

user = User("Alex")
user.hello()
```

---

# Полезные команды

Получить ввод:

```python
name = input("Введите имя: ")
```

Преобразование типов:

```python
int("10")
float("3.14")
str(100)
```

---

# List Comprehension

```python
numbers = [1, 2, 3, 4]

squared = [x * x for x in numbers]
```

---

# Lambda функции

```python
square = lambda x: x * x

print(square(5))
```

---

# Работа с JSON

```python
import json

data = {
    "name": "Alex"
}

json_text = json.dumps(data)

print(json_text)
```

---

# Установка библиотек

```bash
pip install requests
```

---

# Virtual Environment

Создание:

```bash
python -m venv venv
```

Активация:

## Windows

```bash
venv\Scripts\activate
```

## Linux / macOS

```bash
source venv/bin/activate
```

---

# Полезные библиотеки

| Библиотека | Назначение       |
| ---------- | ---------------- |
| requests   | HTTP запросы     |
| pandas     | Таблицы и анализ |
| numpy      | Математика       |
| flask      | Web API          |
| django     | Web framework    |
| matplotlib | Графики          |

---

# Комментарии

```python
# Однострочный комментарий
```

```python
"""
Многострочный комментарий
"""
```

---

# Проверка main

```python
if __name__ == "__main__":
    print("Start")
```

---

# Полезные ссылки

* [https://python.org](https://python.org)
* [https://docs.python.org/3/](https://docs.python.org/3/)
* [https://pypi.org](https://pypi.org)

```
```
