import numpy as np
def confusion_matrix(predY, trueY):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    length = len(predY)
    for i in range(length):
        if trueY[i] == 1:
            if predY[i] == 1:
                true_positive = true_positive + 1
            else:
                false_negative = false_negative + 1
        else:
            if predY[i] == 1:
                false_positive = false_positive + 1
            else:
                true_negative = true_negative + 1
    accuracy = float(true_positive + true_negative)/float(length)
    positive_precision = float(true_positive)/float(true_positive + false_positive)
    negative_precision = float(true_negative)/float(true_negative + false_negative)
    positive_recall = float(true_positive)/float(true_positive + false_negative)
    negative_recall = float(true_negative)/float(true_negative + false_positive)
    conf_matrix = np.array([[true_positive, false_negative], [false_positive, true_negative]])
    return accuracy, positive_precision, negative_precision, positive_recall, negative_recall, conf_matrix
