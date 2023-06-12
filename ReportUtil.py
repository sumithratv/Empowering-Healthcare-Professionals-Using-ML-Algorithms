from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report

def generateReport(y_test, y_pred):
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1 Score:", f1)

    # Calculate precision
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    print("Precision:", precision)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Extract true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculate specificity
    specificity = TN / (TN + FP)
    print("Specificity:", specificity)

    # Calculate sensitivity
    sensitivity = TP / (TP + FN)
    print("Sensitivity:", sensitivity)

    # Generate classification report
    report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
    recall = report['weighted avg']['recall']
    print('recall :', report['weighted avg']['recall'])
    print('support :', report['weighted avg']['support'])

    # Calculate G-measure
    g_measure = 2 * (precision * recall) / (precision + recall)
    print("G-measure:", g_measure)

    # Calculate Matthew's Correlation Coefficient
    mcc = matthews_corrcoef(y_test, y_pred)
    print("Matthew's Correlation Coefficient:", mcc)