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
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import reciprocal, uniform, expon




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
# scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                           scoring='neg_mean_squared_error', cv=10)
# forest_rmse_scores = np.sqrt(-scores)
# print("\nRandom Forest Regressor Cross-Validation Scores:")
# display_scores(forest_rmse_scores)


# --- Using Support Vector Machine Regressor & calculating it's RMSE ---
svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print(f"\nRMSE for the Support Vector Machine Regressor: {svm_rmse}")

# Using CV to evaluate Support vector Regressor
# scores = cross_val_score(svm_reg, housing_prepared, housing_labels, 
#                          scoring="neg_mean_squared_error", cv=10)
# svr_rmse_scores = np.sqrt(-scores)
# print(f'\n Svr support Vector Cross-Validation scores:')
# display_scores(svr_rmse_scores)

# --- Compare Models with Graphs ---

# Graph 1: Bar Chart of RMSE on the full training set
# model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR']
# training_rmse_scores = [linear_rmse, tree_rmse, forest_rmse, svm_rmse]

# plt.figure(figsize=(10, 6))
# plt.bar(model_names, training_rmse_scores, color=['blue', 'green', 'red', 'purple'])
# plt.xlabel("Models")
# plt.ylabel("RMSE on Training Data")
# plt.title("Model Comparison: RMSE on Full Training Set")
# plt.ylim(0, 150000) # Set a limit to make the bars comparable
# plt.show()


# --- Lab 3 ---
print(f"###--- LAB 3 ---###")
# --- Grid search for Decision Tree regressor ---
param_grid = [{'max_depth': list(range(1, 20)),'min_samples_leaf': list(range(1,20))}]
dec_tree_reg = DecisionTreeRegressor()
grid_search = GridSearchCV(dec_tree_reg, param_grid, cv=5, verbose=1,return_train_score=True, n_jobs= -1)
grid_search.fit(housing_prepared,housing_labels)
dt_grid_best_param = grid_search.best_params_
dt_grid_best_estimator = grid_search.best_estimator_
print(f'\ndt_grid_best_param: {dt_grid_best_param}')
print(f'\ndt_grid_best_estimator  : {dt_grid_best_estimator} ')
cv_res = grid_search.cv_results_
# for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
#     print(f'Decision tree grid RMSE: {np.sqrt(-mean_score), params}')
    
# --- Random search for decision tree ---
param_distribs = {'max_depth':randint(low=1, high=20), 'min_samples_leaf':randint(low = 1, high = 20)}

dec_tree_reg = DecisionTreeRegressor()
rnd_search = RandomizedSearchCV(dec_tree_reg, param_distribs, cv=5,n_iter=10, scoring="neg_mean_squared_error",n_jobs= -1,return_train_score=True)
rnd_search.fit(housing_prepared,housing_labels)
dt_rnd_best_param = rnd_search.best_params_
dt_rnd_best_estimator = rnd_search.best_estimator_
print(f'rnd_search_best_param  for decision tree: {dt_rnd_best_param}')
print(f'rnd_search_best_estimator for : {dt_rnd_best_estimator}')
cv_res = rnd_search.cv_results_
# for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
#     print(f'Decision tree random RMSE: {np.sqrt(-mean_score), params}')

# --- Grid search for Random forest regressor ---
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True, n_jobs= -1)
grid_search.fit(housing_prepared, housing_labels)
rf_grid_best_param = grid_search.best_params_
rf_grid_best_estimator = grid_search.best_estimator_
print(f'\nrf_grid_best_param: {rf_grid_best_param}')
print(f"\nrf_grid_best_estimator: {rf_grid_best_estimator}")
cv_res = grid_search.cv_results_
# for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
#     print(f'Random forest grid RMSE: {np.sqrt(-mean_score), params}')

# --- Random search for Random Forest Regressor ---
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
forest_reg = RandomForestRegressor(random_state=42)
rnd_search= RandomizedSearchCV(forest_reg, param_distribs, cv=5, n_iter=10, scoring='neg_mean_squared_error',n_jobs= -1)
rnd_search.fit(housing_prepared,housing_labels)
rf_rnd_best_param = rnd_search.best_params_
rf_rnd_best_estimator = rnd_search.best_estimator_
print(f'rf_rnd_best_param: {rf_grid_best_param}')
print(f'rf_rnd_best_estimator  : {rf_rnd_best_estimator}')
cv_res = rnd_search.cv_results_
# for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
#     print(f'Random forest random RMSE: {np.sqrt(-mean_score), params}')

# --- Grid search for SVR ---
param_grid = [    {
        'kernel': ['rbf'],
        'C': [1, 10, 100, 1000],
        'gamma': ['scale', 0.1, 0.01],
        'epsilon': [0.1, 0.2, 0.5]
    },{
        "kernel": ['linear'],
        "C" : [0.1, 1, 10, 100, 1000],
    }]
svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv= 5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs= -1)
grid_search.fit(housing_prepared,housing_labels)
svm_reg_grid_param = grid_search.best_params_
svm_reg_grid_estimator = grid_search.best_estimator_
print(f'\nsvm_reg_grid_param: {svm_reg_grid_param}')
print(f"\nsvm_reg_grid_param: {svm_reg_grid_estimator}")
cv_res = grid_search.cv_results_
# for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
#     print(f'SVR grid RMSE: {np.sqrt(-mean_score), params}')

# --- Random search for SVR ---
param_distributions = {
    "kernel": ["rbf", "linear"],   # we can still give fixed options
    "C": reciprocal(1, 1000),      # random values between 1 and 1000
    "gamma": expon(scale = 1.0), # random values between 0.001 and 0.1
    "epsilon": uniform(0.01, 1.0)  # random values between 0.01 and 1.01
}
svm_reg = SVR()
rnd_search = RandomizedSearchCV(
    svm_reg,
    param_distributions,
    n_iter=10,   # how many random combos to try
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    return_train_score=True
)
rnd_search.fit(housing_prepared, housing_labels)
svm_reg_rnd_param = rnd_search.best_params_
svm_reg_rnd_estimator = rnd_search.best_estimator_
print("\nsvm_reg_rnd_param", svm_reg_rnd_param )
print("\nsvm_reg_rnd_param:",svm_reg_rnd_estimator )
cv_res = rnd_search.cv_results_
# for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
#     print(f'SVR random RMSE: {np.sqrt(-mean_score), params}')

# --- Grid Search for Linear Regression ---
# Linear Regression has very few hyperparameters, but we can still test:
# - fit_intercept: Whether to calculate the intercept
# - copy_X: Whether to copy input data
param_grid = {
    "fit_intercept": [True, False],
    "copy_X": [True, False]
}

linear_reg = LinearRegression()
grid_search_lin = GridSearchCV(
    linear_reg,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',  # Use negative MSE for scoring
    return_train_score=True,
    n_jobs=-1
)
grid_search_lin.fit(housing_prepared, housing_labels)

# Best params and estimator from Grid Search
lin_grid_best_param = grid_search_lin.best_params_
lin_grid_best_estimator = grid_search_lin.best_estimator_
print(f'\nLinear Regression Grid Search Best Params: {lin_grid_best_param}')
print(f'Linear Regression Grid Search Best Estimator: {lin_grid_best_estimator}')
cv_res = grid_search_lin.cv_results_
print("\nGrid Search Results (RMSE for each combination):")
# for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
#     print(f'linear grid RMSE: {np.sqrt(-mean_score), params}')

# --- Randomized Search for Linear Regression ---
# Random search is usually overkill for Linear Regression, but for practice:
from scipy.stats import randint

param_distribs = {
    "fit_intercept": [True, False],
    "copy_X": [True, False]
}

rnd_search_lin = RandomizedSearchCV(
    linear_reg,
    param_distributions=param_distribs,
    n_iter=4,  # only 4 possible combinations exist
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    return_train_score=True,
    n_jobs=-1
)
rnd_search_lin.fit(housing_prepared, housing_labels)
lin_rnd_best_param = rnd_search_lin.best_params_
lin_rnd_best_estimator = rnd_search_lin.best_estimator_
print(f'\nLinear Regression Random Search Best Params: {lin_rnd_best_param}')
print(f'Linear Regression Random Search Best Estimator: {lin_rnd_best_estimator}')
cv_res = rnd_search_lin.cv_results_
print("\nRandomized Search Results (RMSE for each combination):")
# for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
#     print(f'linear random RMSE: {np.sqrt(-mean_score), params}')

# --- Step 1: Evaluate and Compare Best Models ---

# We'll use the best estimators found from our hyperparameter tuning.
# Let's pick the ones from Randomized Search as it's often more efficient.
best_models_all = {
    "Linear Regression (Random)": lin_rnd_best_estimator,
    "Decision Tree (Random)": dt_rnd_best_estimator,
    "Random Forest (Random)": rf_rnd_best_estimator,
    "SVR (Random)": svm_reg_rnd_estimator,
    
    "Linear Regression (Grid)": lin_grid_best_estimator,
    "Decision Tree (Grid)": dt_grid_best_estimator,
    "Random Forest (Grid)": rf_grid_best_estimator,
    "SVR (Grid)": svm_reg_grid_estimator
}

model_cv_scores = {}

print("\n--- Evaluating Best Models with Cross-Validation (Random + Grid) ---")
for name, model in best_models_all.items():
    # Perform 10-fold cross-validation
    scores = cross_val_score(model, housing_prepared, housing_labels,
                             scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    model_cv_scores[name] = rmse_scores
    print(f"\n Results for {name}")
    display_scores(rmse_scores)

# 1. Calculate the mean RMSE for each model    
mean_rmse_scores = {name: scores.mean() for name, scores in model_cv_scores.items()}
# Find the model with the best (minimum) mean RMSE
best_model_name = min(mean_rmse_scores, key=mean_rmse_scores.get)
least_rmse_value = mean_rmse_scores[best_model_name]
# Print the results
print("\n--- Best Performing Model ---")
print(f"Model with the least RMSE: {best_model_name}")
print(f"Least Mean RMSE value: {least_rmse_value:.2f}") 

# --- Visualize the Comparison ---

model_names = list(model_cv_scores.keys())
cv_mean_scores = [scores.mean() for scores in model_cv_scores.values()]

# --- Visualize the Comparison ---

# Shortened names for readability
short_model_names = [
    "LinReg (Rnd)", "DecTree (Rnd)", "RF (Rnd)", "SVR (Rnd)",
    "LinReg (Grid)", "DecTree (Grid)", "RF (Grid)", "SVR (Grid)"
]
cv_mean_scores = [scores.mean() for scores in model_cv_scores.values()]

plt.figure(figsize=(14, 8))  # Bigger figure
plt.bar(short_model_names, cv_mean_scores, color=['blue', 'green', 'red', 'purple', 'blue', 'green', 'red', 'purple'])
# Labels and Title with larger fonts
plt.xlabel("Machine Learning Models (Tuned)", fontsize=16)
plt.ylabel("Mean RMSE (5-Fold CV)", fontsize=16)
plt.title("Comparison of Tuned Model Performance", fontsize=20)
# Increase tick label size
plt.tick_params(axis='x', labelsize=14, rotation=30)  # angled x labels
plt.tick_params(axis='y', labelsize=14)

plt.ylim(0, max(cv_mean_scores) * 1.1)

plt.tight_layout()  # ensure no overlap
plt.show()
# # --- Final Evaluation on the Test Set ---

# # From the cross-validation results, Random Forest is the best model.
final_model = rf_rnd_best_estimator

# # Separate features and labels from the test set
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# # Prepare the test data using the *already fitted* pipeline
# # IMPORTANT: Use transform() only, not fit_transform(), to avoid data snooping!
X_test_prepared = full_pipeline.transform(X_test)

# # Make predictions on the test set
final_predictions = final_model.predict(X_test_prepared)

# # Calculate the final RMSE
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(f"\n--- Final Model Evaluation on Test Set ---")
print(f"Best Performing Model: Random Forest Regressor")
print(f"Final RMSE on Test Set: {final_rmse}")
