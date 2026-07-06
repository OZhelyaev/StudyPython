import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# метод разделения данных на train и test
from sklearn.model_selection import train_test_split
# методы для масштабирования
from sklearn.preprocessing import StandardScaler
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

def drop_insignificant_signs(data: pd.DataFrame, ins_cols:list):
    print(data.info())
    data.drop(columns=ins_cols, inplace=True)
    print(data.info())

def split_data(data: pd.DataFrame):
    x = data.drop(columns=['Churn']).values  # только значения без данных об оттоке
    y = data['Churn'].values  # выделим целевые значения
    # делим данные, 20% на тест, shuffle=True - каждый раз перемешиваем данные
    x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
    return x_tr, x_test, y_tr, y_test

# сохраним результат в submissions.csv
def save_result(df):
    submission = pd.read_csv('../data/kaggle_submission.csv')
    submission['Churn'] = df
    submission.to_csv('../data/new_kaggle_submission.csv', index=False)


if __name__ == '__main__':
    data = load_data()
    train_info(data)
    total_spent_to_float(data)
   # for num_col in num_cols:
   #     boxplot_for_num_col(data, num_col)
    # subplot_cat_columns(data)
    target_distribution(data)
    target_distribution_on_plot(data)
    # удалим из DataSet незначимые признаки Sex, HasPhoneService, HasMultiplePhoneNumbers, HasOnlineTV, HasMovieSubscription
    #  TotalSpent сильно коррелирует с ClientPeriod и MonthlySpending. Исключить!
    insignificant_signs =  [
        'Sex', 'HasPhoneService', 'HasMultiplePhoneNumbers', 'HasOnlineTV', 'HasMovieSubscription','TotalSpent'
    ]
    drop_insignificant_signs(data, insignificant_signs)
    # разделим данные на train и test
    x_tr, x_test, y_tr, y_test = split_data(data)

    # препроцессор: к каждой группе колонок - своё преобразование
    # нормируем числовые признаки, а категориальные закодируйте с помощью one-hot-encoding'а.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            #('cat1', LabelEncoder, cat_cols),
            ('cat2', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    # собираем общий pipeline
    pipeline = make_pipeline(
        preprocessor,
        LogisticRegressionCV()
    )

    scores = cross_val_score(
        pipeline,
        x_tr,
        y_tr,
        cv=5,
        scoring='accuracy'
    )
    mean_accuracy = scores.mean()
    print(f"Mean accuracy of Logistic Regression for two classes is {mean_accuracy:.4f}")
