import itertools

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# метод разделения данных на train и test
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# методы для масштабирования
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier, Pool, cv as catboost_cv
from sklearn.model_selection import cross_val_score
# Числовые признаки
num_cols = [
    'ClientPeriod', # ClientPeriod (срок обслуживания клиента) — сильный предиктор оттока. Клиенты, которые уходят, в среднем являются клиентами компании гораздо меньше времени (медиана ~10 месяцев против ~37 у оставшихся — разница более чем в 3 раза).
    'MonthlySpending', # Клиенты, которые ушли (Да), в среднем тратят больше в месяц, чем те, кто остался. Слабая корреляция. Попробовать исключить из рассчета
    'TotalSpent' # признак мультиколлинеарности: TotalSpent сильно коррелирует с ClientPeriod и MonthlySpending. Исключить!
]

# Категориальные признаки
cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

feature_cols = num_cols + cat_cols
target_col = 'Churn'


# загрузка данных
def load_data():
    data = pd.read_csv('../data/kaggle_train.csv')
    return data

def train_info(data: pd.DataFrame):
    print('str: ', '-' * 30)
    print(data.describe(include='str').T)
    print('num: ','-' * 30)
    print(data.describe(include=['int64', 'float64']).T[['count', 'min', 'max']])
    print('isna:', '-' * 30)
    print(data.isna().sum(axis=0))
   # print(' num columns scale:', '-' * 30)
   # num_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(' num columns:', '-' * 30)
    print(num_cols)
    print(' num columns head:', '-' * 30)
    print(data[num_cols].head())
    print(' cat columns :', '-' * 30)
    print(cat_cols)
    print(' cat columns head:', '-' * 30)
    print(data[cat_cols].head())

# строим boxplot отображает вариативность признака и размах в квартиле
# q=0.25 - первый квартиль q=0.75 - третий квартиль
def boxplot_for_num_col(data: pd.DataFrame, col_name:str):
    # влияют числовые колонки на отток
    print(data[col_name].dtype)
    print(data[col_name])
    data0 = data.loc[data.Churn == 0][col_name]
    data1 = data.loc[data.Churn == 1][col_name]

    plt.figure(figsize=(7,5))
    plt.boxplot([data0, data1], showfliers=False )
    plt.title('Влияние '+ col_name + ' на отток')
    plt.xlabel('Отток')
    plt.ylabel(col_name)
    plt.xticks([1,2], ['Нет', 'Да'])

    fq = np.quantile(data[col_name].values, q=0.25)
    print(f'first quartile = {fq}')
    tq = np.quantile(data[col_name].values, q=0.75)
    print(f'third quartile = {tq}')
    plt.show()
    # нужно обращать внимание на влияние признака на конечную переменную (зависимость)
    # нужно оценить независимость признаков

def subplot_cat_columns(data: pd.DataFrame):
    # т.т. cat_cols 16 шт отобразим их на сетке 4 на 4
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))
    # собираем словарь: имя колонки -> Axes . axes.flatten() - расплющим 2D массив
    ax_map = dict(zip(cat_cols, axes.flatten()))
    for col in cat_cols:
        subplot_for_cat_col(data, col, ax_map[col])

    plt.tight_layout()
    plt.show()

def subplot_for_cat_col(data: pd.DataFrame, col:str, ax: plt.Axes):
    # доля оттока внутри каждой категории, в %
    ct = pd.crosstab(data[col], data['Churn'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=ax, color=['#2ca02c', '#d62728'])

    ax.set_title(f'Отток по {col}')
    ax.set_ylabel('% клиентов')
    ax.set_xlabel('')
    ax.legend(title='Отток', labels=['Нет', 'Да'])
    ax.tick_params(axis='x', rotation=30)
    # подписываем проценты на сегментах
    for cnt in ax.containers:
        ax.bar_label(cnt, fmt='%.1f%%', label_type='center', fontsize=8)

# Необходимо преобразовать TotalSpent т.к. он имеет тип object (строки), а не числовой тип
def total_spent_to_float(data: pd.DataFrame):
    data['TotalSpent'] = pd.to_numeric(data['TotalSpent'], errors='coerce')
    # Проверяем, сколько значений не удалось преобразовать
    print(data['TotalSpent'].isna().sum())
    # Смотрим на эти строки — часто это клиенты с ClientPeriod = 0
    print(data[data['TotalSpent'].isna()][['ClientPeriod', 'TotalSpent']])
    data['TotalSpent'] = data['TotalSpent'].fillna(0)

# распределение целевой переменной. анализ являются ли классы несбалансированными
def target_distribution(data: pd.DataFrame):
    print(data['Churn'].value_counts())
    print(data['Churn'].value_counts(normalize=True) * 100)

def target_distribution_on_plot(data: pd.DataFrame):
    (data['Churn'].value_counts(normalize=True) * 100).plot(kind='bar', color=['green', 'red'])
    plt.title('Распределение целевой переменной (Churn) %')
    plt.ylabel('% клиентов')
    plt.xlabel('Отток')
    plt.xticks(rotation=0)

    # подписать значения над столбцами
    for i, v in enumerate(data['Churn'].value_counts(normalize=True) * 100):
        plt.text(i, v, str(round(v,2))+'%', ha='center')
    plt.show()

def merge_no_internet_service(data: pd.DataFrame):
    service_cols = ['HasOnlineSecurityService', 'HasTechSupportAccess',
                    'HasOnlineBackup', 'HasDeviceProtection',
                    'HasOnlineTV', 'HasMovieSubscription']

    for col in service_cols:
        data[col] = data[col].replace('No internet service', 'No')


def drop_insignificant_signs(data: pd.DataFrame, ins_cols:list):
    print(data.info())
    data.drop(columns=ins_cols, inplace=True)
    print(data.info())

def split_data(data: pd.DataFrame):
    x = data.drop(columns=['Churn'])  # только значения без данных об оттоке
    y = data['Churn']  # выделим целевые значения
    # делим данные, 20% на тест, shuffle=True - каждый раз перемешиваем данные
    X_tr, X_test, y_tr, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
    return X_tr, X_test, y_tr, y_test

# нормируем числовые признаки, а категориальные закодируйте с помощью one-hot-encoding'а.
# возвращаем препроцессор: к каждой группе колонок - своё преобразование
def column_transformation(num_cols, cat_cols:list):

    # удалим незначимые столбцы
    insignificant_cols = ['Sex', 'HasPhoneService', 'HasMultiplePhoneNumbers', 'HasOnlineTV', 'HasMovieSubscription']
    num_cols.remove('TotalSpent')
    cat_cols = [col for col in cat_cols if col not in insignificant_cols]
    # обработаем признаки, сначала обработаем NaN через SimpleImputer, а затем уже значения
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return num_cols, cat_cols, preprocessor


def grid_search_catboost(X_tr, y_tr:pd.DataFrame):
    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    # Лучшие параметры: {'depth': 3, 'learning_rate': 0.05, 'iterations': 300}, ROC-AUC = 0.8534
    param_combinations = list(itertools.product(
        # 'depth':
        #[3, 5, 6],
        [3],
        # 'learning_rate':
        #[0.01, 0.03, 0.05],
        [0.05],
        # 'iterations':
        # [130, 200, 300]
        [300]
    ))

    best_score = -1
    best_params = None

    for depth, learning_rate, iterations in param_combinations:
        fold_scores = []
        for train_idx, val_idx in cv.split(X_tr, y_tr):
            X_fold_train, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_fold_train, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

            model = CatBoostClassifier(
                cat_features=cat_cols,
                loss_function='Logloss',
                eval_metric='AUC',
                auto_class_weights='Balanced',
                depth=depth,
                learning_rate=learning_rate,
                iterations=iterations,
                random_state=47,
                verbose=False,
                early_stopping_rounds=50
            )
            model.fit(X_fold_train, y_fold_train)
            proba = model.predict_proba(X_fold_val)[:, 1]
            fold_scores.append(roc_auc_score(y_fold_val, proba))

        mean_score = np.mean(fold_scores)
        print(f"depth={depth}, lr={learning_rate}, iterations={iterations} -> ROC-AUC = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = {'depth': depth, 'learning_rate': learning_rate, 'iterations': iterations}

    print(f"\nЛучшие параметры: {best_params}, ROC-AUC = {best_score:.4f}")
    # обучаем модель на лучших параметрах
    final_catboost = CatBoostClassifier(
        cat_features=cat_cols,
        auto_class_weights='Balanced',
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=47,
        verbose=False,
        **best_params  # depth, learning_rate, iterations из лучшей комбинации
    )
    final_catboost.fit(X_tr, y_tr)
    final_catboost.save_model('../model/kaggle_catboost_churn_model.cbm')
    return final_catboost


#  прогон модели на тестовых данных
def model_kaggle_test():
    # Шаг 3. Предсказание на тестовых данных
    X_test = pd.read_csv('../data/kaggle_test.csv')

    # !!!!! необходимо удалить незначимые столбцы ....

    # загрузи модель
    model = CatBoostClassifier()
    model.load_model('../model/kaggle_catboost_churn_model.cbm')

    pred_test = model.predict(X_test)
    print(' X_test CatBoostClassifier result: ', pred_test)
    return pd.Series(pred_test)

# сохраним результат в submissions.csv
def save_result(df):
    submission = pd.read_csv('../data/kaggle_submission.csv')
    submission['Churn'] = df
    submission.to_csv('../data/new_kaggle_submission.csv', index=False)



if __name__ == '__main__':
    data = load_data()
    train_info(data)
    total_spent_to_float(data)
    #for num_col in num_cols:
    #    boxplot_for_num_col(data, num_col)
    #subplot_cat_columns(data)
    target_distribution(data)
    target_distribution_on_plot(data)
    # схлопним признак No internet service
    merge_no_internet_service(data)
    # удалим из DataSet незначимые признаки Sex, HasPhoneService, HasMultiplePhoneNumbers, HasOnlineTV, HasMovieSubscription
    #  TotalSpent сильно коррелирует с ClientPeriod и MonthlySpending. Исключить!
    insignificant_cols = [
        'Sex', 'HasPhoneService', 'HasMultiplePhoneNumbers', 'HasOnlineTV', 'HasMovieSubscription', 'TotalSpent'
    ]
    drop_insignificant_signs(data, insignificant_cols)
    # разделим данные на train и test
    X_tr, X_test, y_tr, y_test = split_data(data)
    # препроцессор: к каждой группе колонок - своё преобразование
    # нормируем числовые признаки, а категориальные закодируйте с помощью one-hot-encoding'а.
    num_cols, cat_cols, preprocessor = column_transformation(num_cols, cat_cols)

    models = {
        'LogisticRegression': LogisticRegressionCV(),
        'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
        'SVC': SVC(class_weight='balanced', random_state=42)
    }
    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipeline = make_pipeline(preprocessor, clf)
        scores = cross_val_score(pipeline, X_tr, y_tr, cv=cv, scoring='roc_auc')
        print(f"{name}: roc_auc = {scores.mean():.4f} (+/- {scores.std():.4f})")

    # проверяем catboost
    final_catboost = grid_search_catboost(X_tr, y_tr)
    y_pred_proba = final_catboost.predict_proba(X_test)[:, 1]
    print(f"Финальный Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    # фиксируем результаты в файле
    save_result(model_kaggle_test())

