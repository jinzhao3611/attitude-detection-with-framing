import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    # can be used in trainer or independently
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc_score = accuracy.compute(predictions=predictions, references=labels)
    # which average method to use for PRF1?
    prec_score = precision.compute(predictions=predictions, references=labels, average="macro")
    recall_score = recall.compute(predictions=predictions, references=labels, average="macro")
    f1_score = f1.compute(predictions=predictions, references=labels, average="macro")
    return {**acc_score, **prec_score, **recall_score, **f1_score}
