import mlflow
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from .data_processing import load_pitcher_data, create_labels

def train_velocity_model():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.xgboost.autolog()
    
    # Load data (2021-2024 seasons)
    df = load_pitcher_data("2021-03-01", "2024-10-01")
    labeled_df = create_labels(df)
    
    # Temporal split
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(labeled_df):
        train = labeled_df.iloc[train_index]
        test = labeled_df.iloc[test_index]
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=200,
            max_depth=5
        )
        
        model.fit(
            train[['avg_velo', 'max_velo', 'spin_diff', 'stress_pitches']],
            train['velo_drop']
        )
        
        # Log model performance
        with mlflow.start_run():
            preds = model.predict_proba(test[features])[:,1]
            mlflow.log_metric("roc_auc", roc_auc_score(test['velo_drop'], preds))
            mlflow.xgboost.log_model(model, "velocity_model")
