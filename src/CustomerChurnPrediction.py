import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# метод разделения данных на train и test
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipeline = make_pipeline(preprocessor, clf)
        scores = cross_val_score(pipeline, X_tr, y_tr, cv=cv, scoring='roc_auc')
        pipeline.fit(X_tr, y_tr)  # обучаем отдельно, ТОЛЬКО чтобы посмотреть classes_
        print(f"{name}: roc_auc = {scores.mean():.4f} (+/- {scores.std():.4f})")

    pool = Pool(X_tr, y_tr, cat_features=cat_cols)

    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'auto_class_weights': 'Balanced',
        'iterations': 50,
        'learning_rate': 0.01,
        'depth': 6,
        'random_state': 47,
        'verbose': False
    }

    cv_results = catboost_cv(pool, params, fold_count=5, stratified=True, shuffle=True, seed=42)
    print(cv_results.tail())  # последняя строка - метрики после всех итераций
    print(f"Mean test AUC: {cv_results['test-AUC-mean'].iloc[-1]:.4f}")
#    pipeline = make_pipeline(
#        preprocessor,
#        LogisticRegression() #CV(use_legacy_attributes=True)
#    )

#    scores = cross_val_score(
#        pipeline,
#        X_tr,
#        y_tr,
#        cv=cv,
#        scoring='roc_auc'
#    )
#    mean_accuracy = scores.mean()
#    print(f"Mean accuracy of Logistic Regression CV for two classes is {mean_accuracy:.4f}")
