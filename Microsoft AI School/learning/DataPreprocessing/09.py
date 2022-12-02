import numpy as np
from sklearn.preprocessing import OrdinalEncoder

features = np.array([["Low",10], ["High", 50], ["Medium", 3]])
ordinal_ec = OrdinalEncoder()
ordinal_ec.fit_transform(features)
ordinal_ec_data = ordinal_ec.categories_

print("ordinal_encoding_category", ordinal_ec_data)
print(ordinal_ec.fit_transform(features))