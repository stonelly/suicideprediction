from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor
import pandas as pd
import joblib

# Define the hyperparameters grid
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def preprocess_and_train_model(data):
    label_encoder = LabelEncoder()
    data['RegionName'] = label_encoder.fit_transform(data['RegionName'])

    X = data.drop(['SuicideCount', 'CountryName','Year','CauseSpecificDeathPercentage','DeathRatePer100K','GDPPerCapita','GNIPerCapita'], axis=1)
    y = data['SuicideCount']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Define the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    model = RandomForestRegressor()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply the preprocessor and fit on training data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Perform Grid Search with 5-fold cross-validation for MSE
    grid_search = GridSearchCV(estimator=model, param_grid=rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Train the model with the best hyperparameters
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    return best_model


# Load data
data = pd.read_csv('../data/data_clean.csv').copy()
model = preprocess_and_train_model(data)
# Save the trained model
joblib.dump(model, 'suicide_count_prediction_model_RF.pkl')