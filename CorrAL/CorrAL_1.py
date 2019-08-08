import numpy as np
import pandas as pd

# load the datasets
df = pd.read_csv('../Datasets/CorrAL.csv')

array = df.values
X = array[:, 0:6]
y = array[:, 6]
n_samples, n_features = X.shape

## All data has been used to rank the features

##################################################
## Statistical Based Feature Selection Methods ##

## Chi Square
from skfeature.function.statistical_based import chi_square
score_Chi = chi_square.chi_square(X, y)
Id_Chi = chi_square.feature_ranking(X,)
print('Chi Square score:', score_Chi)

## F score
from skfeature.function.statistical_based import f_score
score_F = f_score.f_score(X, y)
print('F-score:', score_F)

## Gini Index
from skfeature.function.statistical_based import gini_index
score_Gini = gini_index.gini_index(X, y)
print('Gini-score:', score_Gini)

from skfeature.function.statistical_based import CFS
idx = CFS.cfs(X, y)
print('CFS:', idx)

## T score
from skfeature.function.statistical_based import t_score
score_T = t_score.t_score(X, y)
print('t-score:', score_T)

## low variance
from skfeature.function.statistical_based import low_variance
p = 0.1
selected_features = low_variance.low_variance_feature_selection(X, p*(1-p))
#print('selected features:', selected_features)

#############################################
## Similarity Based Feature Selection Methods

from skfeature.function.similarity_based import reliefF
score_Rel = reliefF.reliefF(X, y)
print('ReliefF score:', score_Rel)

from skfeature.function.similarity_based import fisher_score
score_Fisher = fisher_score.fisher_score(X, y)
print('Fisher score:', score_Fisher)

