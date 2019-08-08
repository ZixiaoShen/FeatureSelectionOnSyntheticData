import numpy as np
import pandas as pd

# load the datasets
df = pd.read_csv('../Datasets/CorrAL.csv')

array = df.values
X = array[:, 0:6]
y = array[:, 6]
n_samples, n_features = X.shape

## All data has been used to rank the features

## Chi Square
from skfeature.function.statistical_based import chi_square
score_Chi = chi_square.chi_square(X, y)
print('Chi Square score:', score_Chi)

from skfeature.function.similarity_based import reliefF
score_Rel = reliefF.reliefF(X, y)
print('ReliefF score:', score_Rel)

from skfeature.function.statistical_based import f_score
score_F = f_score.f_score(X, y)
print('F-score:', score_F)

from skfeature.function.statistical_based import t_score
score_T = t_score.t_score(X, y)
print('t-score:', score_T)
