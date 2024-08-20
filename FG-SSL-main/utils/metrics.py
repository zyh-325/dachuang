from sklearn.metrics import f1_score, precision_score, recall_score

def get_metrics(args, labels, preds):
    """
    计算F1分数、精确率和召回率。

    参数:
    - args: 参数对象（在这里可能不使用，但为了与其他函数保持一致）。
    - labels: 真实标签。
    - preds: 预测标签。

    返回值:
    - f1: F1分数。
    - precision: 精确率。
    - recall: 召回率。
    """
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()

    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')

    return f1, precision, recall
