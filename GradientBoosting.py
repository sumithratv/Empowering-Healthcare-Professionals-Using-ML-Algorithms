from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

def createModelAndPredictGradientBoosting(X_train, y_train, X_test, preprocessor):
    # Create the RandomForestClassifier pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', GradientBoostingClassifier())])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return y_pred
