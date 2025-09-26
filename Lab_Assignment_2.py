import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV



# --- Load Data ---
housing = pd.read_csv('handson-ml2/datasets/housing/housing.csv')

# --- Create Stratified Split based on Median Income ---
housing['income_cat'] = pd.cut(
    housing['median_income'],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

# Perform the stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Now that we have our sets, we can drop the 'income_cat' column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# --- Separate Features and Labels ---
# We only work with the training set from now on to avoid data snooping
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# --- Define Custom Transformer ---
# Get column indices from the training features DataFrame. These are used inside the transformer.
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# --- Build the Preprocessing Pipeline ---
housing_num = housing.select_dtypes(include=[np.number])
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# Pipeline for numerical attributes
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

# Full pipeline to handle all columns
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# --- Run the Pipeline ---
housing_prepared = full_pipeline.fit_transform(housing)
print("Shape of the prepared data:", housing_prepared.shape)
print("\nSample of prepared data (first 5 rows):")
print(housing_prepared[:5])

# --- Linear Regression ---
linear_reg = LinearRegression()
linear_reg.fit(housing_prepared, housing_labels)

# Testing on some data
some_data = housing.iloc[:10]
some_labels = housing_labels.iloc[:10]
some_data_prepared  = full_pipeline.transform(some_data)
print(f"\nPredictions using Linear Regression: {linear_reg.predict(some_data_prepared)}")
print(f'\nlist of labels in linear regression: {list(some_labels)}')

# Calculating RMSE for the linear regression
housing_predictions = linear_reg.predict(housing_prepared)
linear_mse = mean_squared_error(housing_labels,housing_predictions)
linear_rmse = np.sqrt(linear_mse)
print(f"\nRMSE for the Linear regression: {linear_rmse}")

#calculate MAE for linear regression
linear_mae = mean_absolute_error(housing_labels,housing_predictions)
print(f"\nMAE for the Linear regression: {linear_mae}")

# --- Using Decision Tree Regressor ---
dec_tree_reg = DecisionTreeRegressor()
dec_tree_reg.fit(housing_prepared, housing_labels)
print(f"\nPredictions using Decision Tree Regressor: {dec_tree_reg.predict(some_data_prepared)}")
print(f'\nlist of labels in decision tree regression: {list(some_labels)}')

# Calculating RMSE for the decision tree regressor
housing_predictions = dec_tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(f"\nRMSE for the Decision Tree Regressor: {tree_rmse}")

# --- Cross-Validation for Decision Tree Regressor ---
scores = cross_val_score(dec_tree_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print(f'\nscores: {scores}')
    print(f'mean: {scores.mean()}')
    print(f'standard deviation: {scores.std()}')
print("\nDecision Tree Regressor Cross-Validation Scores:")
display_scores(tree_rmse_scores)

# --- Cross-Validation for Linear Regression ---
scores = cross_val_score(linear_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
linear_rmse_scores = np.sqrt(-scores)
print("\nLinear Regression Cross-Validation Scores:")
display_scores(linear_rmse_scores)

# --- Using Random Forest Regressor ---
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
print(f"random forest for some data: {forest_reg.predict(some_data_prepared)}")
print(f'\nlist of labels in random forest regression: {list(some_labels)}')

#calculating RMSE for random forest regressor
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(f"\nRMSE for the Random Forest Regressor: {forest_rmse}")

#using cv to evaluate random forest regressor
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                          scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores)
print("\nRandom Forest Regressor Cross-Validation Scores:")
display_scores(forest_rmse_scores)


# --- Using Support Vector Machine Regressor & calculating it's RMSE ---
svm_reg = SVR(kernel="linear")
# svm_reg = SVR(kernel="linear", C=30000)
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print(f"\nRMSE for the Support Vector Machine Regressor: {svm_rmse}")

# Using CV to evaluate Support vector Regressor
scores = cross_val_score(svm_reg, housing_prepared, housing_labels, 
                         scoring="neg_mean_squared_error", cv=10)
svr_rmse_scores = np.sqrt(-scores)
print(f'\n Svr support Vector Cross-Validation scores:')
display_scores(svr_rmse_scores)

# --- Compare Models with Graphs ---

## --- Final Model Comparison ---

# (Assuming you have calculated these values for all four models)
model_names = ['Linear Reg', 'Decision Tree', 'Random Forest', 'SVR']

# Use Mean from cross-validation scores for the main comparison
cv_mean_scores = [
    linear_rmse_scores.mean(),
    tree_rmse_scores.mean(),
    forest_rmse_scores.mean(),
    svr_rmse_scores.mean() # Use scores from your best SVR
]

# Using the standard deviation to show the variance in performance
cv_std_scores = [
    linear_rmse_scores.std(),
    tree_rmse_scores.std(),
    forest_rmse_scores.std(),
    svr_rmse_scores.std() # Use scores from your best SVR
]

# --- Creating the Comparison Bar Chart ---
plt.figure(figsize=(12, 7))
bars = plt.bar(model_names, cv_mean_scores, capsize=5, 
               color=['blue', 'green', 'red', 'purple'], alpha=0.8)

plt.xlabel("Machine Learning Models", fontsize=12)
plt.ylabel("Mean RMSE from 10-Fold Cross-Validation", fontsize=12)
plt.title("Comparison of Model Performance before using best for svr", fontsize=14)

# Adding the RMSE value on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f'${round(yval)}', 
             ha='center', va='bottom')

plt.show()


# --- experimental ---

# --- using CV to test SVR ---
# param_grid = [
#         {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
#         {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
#          'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
#     ]

# svm_reg = SVR()
# grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
# grid_search.fit(housing_prepared, housing_labels)
# print(f'\n SVR Best_params: {grid_search.best_params_}')
# print(f"\n SVR Best_estimator: {grid_search.best_estimator_}")