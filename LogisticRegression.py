from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
def createModelAndPredictLogisticRegression(X_train, y_train, X_test, preprocessor):
    # Create the RandomForestClassifier pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(max_iter=1000))])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return y_pred



