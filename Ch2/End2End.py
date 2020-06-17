
# DATA FETCHING
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH  = os.path.join("datasets","housing")
HOUSING_URL   = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path,exist_ok = True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

fetch_housing_data()
    
# DATA LOADING
import pandas as pd
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()
housing.info()
# Find out what categories exist and how many districts belong to each category
housing["ocean_proximity"].value_counts()
# a = housing["total_rooms"].value_counts()
a = housing.describe()

# DATA VISUALIZING
# % is ONLY IN JUPYTER NOTEBOOK
#%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize = (20,15))
plt.show()


# CREATING TEST SETS
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size    = int(len(data) * test_ratio)
    test_indices     = shuffled_indices[:test_set_size]
    train_indices    = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

# A POSSIBLE SOLUTION TO GET A "CERTAIN" (AVOID "LEAKING" WHEN RELOAD) DATA SET
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
# HOUSING DOESN'T HAVE IDENTIFIER, SO WE USE ROW INDEX AS ID
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
'''
If you use the row index as a unique identifier, you need to make sure that new
data gets appended to the end of the dataset and that no row ever gets deleted. If
this is not possible, then you can try to use the most stable features to build a
unique identifier. For example, a district’s latitude and longitude are guaranteed
to be stable for a few million years, so you could combine them into an ID like
so:
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")'''

# USING SKLEARN TO ACHIEVE THE SAME GOAL
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
print(len(train_set))
print(len(test_set))

# STRATIFIED SAMPLING BASED ON INCOME CATEGORY
# CREATING AN INCOME CATEGORY ATTRIBUTE
housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0., 1.5, 3., 4.5, 6., np.inf],
                              labels = [1, 2, 3, 4, 5])
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit as SSS

split = SSS(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set  = housing.loc[test_index]
print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))
# Remove "income_cat" attribute
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis = 1, inplace = True)

# VISUALIZING THE DATA
housing = strat_train_set.copy()
housing.plot(kind = "scatter", x = "longitude", y = "latitude")
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4,
             s = housing["population"]/100, label = "population", figsize = (10,7),
             c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True,)
plt.legend()

# LOOKING FOR CORRELATIONS
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)
# Another way to check for correlation (pandas)
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize = (15,10))
# NOW FOCUS ON MEDIAN INCOME
housing.plot(kind = "scatter", x = "median_income", y = "median_house_value",
             alpha = 0.1)

# TRY OUR VARIOUS ATTRIBUTE COMBINATION
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"]   = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
# NEW CORRELATION MATRIX
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# PREPARE DATA FOR ML ALGO

housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()
# DATA CLEANING
# Takcle missing total_bedrooms
housing.dropna(subset = ["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis = 1)    # option 2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace = True) # option 3
# IF: Option 3, Need to KEEP the MEDIAN for TEST SET
from sklearn.impute import SimpleImputer # sklearn can take care of it
imputer = SimpleImputer(strategy = "median")
housing_num = housing.drop("ocean_proximity", axis =1) #Since the median can only be computed on numerical attributes, 
# you need to create a copy of the data without the text attribute ocean_proximity
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)# transform training set
housing_tr = pd.DataFrame(X, columns = housing_num.columns,
                          index = housing_num.index) # Convert to pd DataFrame

# Numerical part has done, NOW IS THE TEXT ATTRIBUTES
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
# Convert CATEGORY to NUMBER
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_
# Modify CATEGORY since One issue with this representation is that 
# ML ALGO will assume that two nearby values are more similar than two distant values.
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot # OUTPUT IS A SCIPY SPARSE MATRIX
housing_cat_1hot.toarray()
cat_encoder.categories_

# CUSTOM TRANSFORMAER
'''
You will want your transformer to work seamlessly with
Scikit-Learn functionalities (such as pipelines), and since Scikit-Learn relies on
duck typing (not inheritance), all you need to do is create a class and implement
three methods: fit() (returning self), transform(), and fit_transform().
You can get the last one for free by simply adding TransformerMixin as a base
class. If you add BaseEstimator as a base class (and avoid *args and **kargs
in your constructor), you will also get two extra methods (get_params() and
set_params()) that will be useful for automatic hyperparameter tuning.'''
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self # nothing else to do
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)

# Transformation Pipeline for numerical attr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
        ])

# Using ColumnTransformer to handle all columns with diff pipelines
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs),
        ])
                
housing_prepared = full_pipeline.fit_transform(housing)

# SELECT AND TRAIN A MODEL
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Try a few instances
some_data = housing.iloc[:10]
some_labels = housing_labels.iloc[:10]
some_data_prepared = full_pipeline.fit_transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))
# RMSE
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse  = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse # 68628.19819848923 BAD: UNDERFITTING

# TRY ANOTHER MODEL
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse  = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse # 0.0 BAD: OVERFITTING

# BETTER EVALUATION USING CROSS-VALIDATION
from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring = "neg_mean_squared_error", cv = 10)
tree_rmse_scores = np.sqrt(-tree_scores)

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
display_scores(tree_rmse_scores)

# TRY LINEAR REG USING CROSS-VALIDATION
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring = "neg_mean_squared_error", cv = 10)
lin_rsme_scores = np.sqrt(-lin_scores)
display_scores(lin_rsme_scores)

# TRY RANDOMFOESTREGRESSOR
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse  = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse # 18749.929216326935 BETTER
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring = "neg_mean_squared_error", cv = 10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# SAVE MODELS (HYPERPARAMETERS, TRAINED PANAMETERS, SCORES, PREDICTIONS, ETC.)
import joblib
joblib.dump(forest_reg, "my_model.pkl")
# LATER...
my_model_loaded = joblib.load("my_model.pkl")

# FINETUNE MODEL
# GridSearch for Forest
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]

forest_reg  = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv = 5,
                           scoring = 'neg_mean_squared_error',
                           return_train_score = True)

grid_search.fit(housing_prepared, housing_labels)
grid_search.best_estimator_ 
grid_search.best_params_ # {'max_features': 6, 'n_estimators': 30}
# Notice taht 8 is the maxism, which means you might need try bigger value
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# ANALYZE the BEST MODELS and THEIR ERRORS
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

# Display importances score next to corresponding attributes
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder   = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes    = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse = True)
# With These IMPORTANCES, can MODIFY (Drop or Add) features
# and TRY

# EVALUATE SYSTEM ON TEST SET
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis =1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
#X_test_prepared = full_pipeline.fit_transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse  = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) -1,
                       loc = squared_errors.mean(),
                       scale = stats.sem(squared_errors)))
