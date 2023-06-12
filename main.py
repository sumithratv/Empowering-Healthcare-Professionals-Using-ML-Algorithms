
from DataPreProcessing import FilteringFeaturesAndTarget
from DataPreProcessing import DataPreparation
from LogisticRegression import createModelAndPredictLogisticRegression
from ReportUtil import generateReport
from GradientBoosting import createModelAndPredictGradientBoosting
from RandomForestAlgorithm import createModelAndPredictRandomForestClassifier


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    FilteringFeaturesAndTarget()
    cleanedFileName = 'cleaned_data.csv'
    preprocessor,X_train, X_test, y_train, y_test =DataPreparation(cleanedFileName)
    y_pred = createModelAndPredictLogisticRegression(X_train, y_train, X_test, preprocessor)
    print('****************LogisticRegression Algorithm Report Start *********************** ')
    generateReport(y_test, y_pred)
    print('****************LogisticRegression Algorithm Report End *********************** ')
    print('')
    print('')
    print('****************GradientBoosting Algorithm Report Start *********************** ')
    y_pred = createModelAndPredictGradientBoosting(X_train, y_train, X_test, preprocessor)
    generateReport(y_test, y_pred)
    print('****************GradientBoosting Algorithm Report End *********************** ')
    print('')
    print('')
    print('****************RandomForestClassifier Algorithm Report Start *********************** ')
    y_pred = createModelAndPredictRandomForestClassifier(X_train, y_train, X_test, preprocessor)
    generateReport(y_test, y_pred)
    print('****************RandomForestClassifier Algorithm Report End *********************** ')
