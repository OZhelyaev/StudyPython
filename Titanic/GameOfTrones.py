import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv('/data/game_of_thrones_test.csv', index_col='S.No')
print(data.head())