from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

# iris: 붓꽃 데이터 세트
# iris_data: iris에서 피처(특징)만으로 된 numpy
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

print('iris target값:', iris_label)
print('iris target명:', iris.target_names)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df.head(3)