# временные заметки 
---

# Вариант 1 — простые преобразования через FunctionTransformer

```python
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

def clean_data(X):
    X = X.copy()
    
    # приведение TotalSpent к числу
    X['TotalSpent'] = pd.to_numeric(X['TotalSpent'], errors='coerce').fillna(0)
    
    # схлопывание "No internet/phone service" -> "No"
    service_cols = ['HasOnlineSecurityService', 'HasTechSupportAccess', 
                     'HasOnlineBackup', 'HasDeviceProtection', 
                     'HasOnlineTV', 'HasMovieSubscription']
    for col in service_cols:
        X[col] = X[col].replace('No internet service', 'No')
    
    X['HasMultiplePhoneNumbers'] = X['HasMultiplePhoneNumbers'].replace('No phone service', 'No')
    
    return X

def drop_insignificant(X):
    insignificant_cols = ['Sex', 'HasPhoneService', 'HasMultiplePhoneNumbers', 
                           'HasOnlineTV', 'HasMovieSubscription']
    return X.drop(columns=insignificant_cols, errors='ignore')  # errors='ignore' - не упадёт, если колонки уже нет

cleaning_pipeline = Pipeline([
    ('clean', FunctionTransformer(clean_data)),
    ('drop_cols', FunctionTransformer(drop_insignificant)),
    ('preprocessor', preprocessor),  # ваш ColumnTransformer (scaler + encoder)
    ('model', LogisticRegressionCV())
])

cleaning_pipeline.fit(X_tr, y_tr)
```

---

---

# Вариант 2 — кастомный класс через TransformerMixin (более гибкий, рекомендую)

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    """Приведение типов и схлопывание избыточных категорий"""
    
    def fit(self, X, y=None):
        return self  # здесь нечего "учить", просто правила
    
    def transform(self, X):
        X = X.copy()
        X['TotalSpent'] = pd.to_numeric(X['TotalSpent'], errors='coerce').fillna(0)
        
        service_cols = ['HasOnlineSecurityService', 'HasTechSupportAccess', 
                         'HasOnlineBackup', 'HasDeviceProtection', 
                         'HasOnlineTV', 'HasMovieSubscription']
        for col in service_cols:
            X[col] = X[col].replace('No internet service', 'No')
        
        X['HasMultiplePhoneNumbers'] = X['HasMultiplePhoneNumbers'].replace(
            'No phone service', 'No'
        )
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Удаление незначимых признаков"""
    
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Добавление новых признаков (TotalServices, IsNewClient и т.д.)"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        service_cols = ['HasOnlineSecurityService', 'HasTechSupportAccess', 
                         'HasOnlineBackup', 'HasDeviceProtection', 
                         'HasOnlineTV', 'HasMovieSubscription']
        X['TotalServices'] = (X[service_cols] == 'Yes').sum(axis=1)
        X['IsNewClient'] = (X['ClientPeriod'] <= 6).astype(int)
        return X
```
# Сборка полного pipeline

```python
insignificant_cols = ['Sex', 'HasPhoneService', 'HasMultiplePhoneNumbers', 
                       'HasOnlineTV', 'HasMovieSubscription']

full_pipeline = Pipeline([
    ('cleaner', DataCleaner()),                          # 1. приведение типов, схлопывание категорий
    ('feature_eng', FeatureEngineer()),                   # 2. новые признаки (опционально)
    ('dropper', ColumnDropper(insignificant_cols)),       # 3. удаление незначимых признаков
    ('preprocessor', preprocessor),                       # 4. ColumnTransformer (scaler + encoder)
    ('model', LogisticRegressionCV())                     # 5. модель
])

full_pipeline.fit(X_tr, y_tr)
```

---

# Использование для предсказания — теперь всё в одну строку

```python
# сохраняем весь pipeline целиком
import joblib
joblib.dump(full_pipeline, 'full_pipeline.pkl')

# при получении новых тестовых данных - загружаем и просто вызываем predict
loaded_pipeline = joblib.load('full_pipeline.pkl')

raw_test_data = pd.read_csv('test_data.csv')  # даже НЕОБРАБОТАННЫЕ сырые данные!
predictions = loaded_pipeline.predict(raw_test_data)
probabilities = loaded_pipeline.predict_proba(raw_test_data)[:, 1]
```

---

# Для CatBoost — аналогичный подход, но проще (без ColumnTransformer с encoder)

```python
catboost_pipeline = Pipeline([
    ('cleaner', DataCleaner()),
    ('feature_eng', FeatureEngineer()),
    ('dropper', ColumnDropper(insignificant_cols)),
    ('model', CatBoostClassifier(cat_features=cat_cols, auto_class_weights='Balanced', 
                                   verbose=False, random_state=42))
])

catboost_pipeline.fit(X_tr, y_tr)  # обратите внимание на проблему clone с cross_val_score,
                                     # но для обычного .fit()/.predict() всё работает нормально
```

---
