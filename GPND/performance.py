import random, torch, os, pickle
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics


def GetF1(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)

def get_valid_result_tuple_list_and_inout_classes_and_percent(val_result_path, is_mrx):
    just_file_name = val_result_path.split("/")[1]
    if not is_mrx:
        inliner_classes = [int(just_file_name.split("_")[1])]
        outlier_classes = list(range(10))
        outlier_classes.remove(inliner_classes[0])
        outlier_percent = int(just_file_name.split("_")[2])
    else:
        outlier_classes = [int(just_file_name.split("_")[1])]
        inliner_classes = list(range(10))
        inliner_classes.remove(outlier_classes[0])
        outlier_percent = None

    with open(val_result_path, 'rb') as pkl:
        # (is_inlier, p value)  // is_inlier, is_outlier 따로 구분해줘야하는데 못해줌
        valid_result_tuple_list = pickle.load(pkl)

    return valid_result_tuple_list, inliner_classes, outlier_classes, outlier_percent


def get_confusion_components(test_result_tuple_list, threshold, target):
    # true_positive, true_negative, false_negative, false_positive
    target_confusion_list = [0]*4

    for test_is_in_target, p in test_result_tuple_list:
        if target == "inlier":
            positive_condition = p > threshold
        elif target == "outlier":
            positive_condition = p < threshold
        else:
            raise Exception("Wrong target")

        if test_is_in_target == positive_condition:
            if test_is_in_target:
                target_confusion_list[0] += 1
            else:
                target_confusion_list[1] += 1
        else:
            if test_is_in_target:
                target_confusion_list[2] += 1
            else:
                target_confusion_list[3] += 1

    return target_confusion_list


def calculate_performance_metrics(
    target_confusion_components, test_result_tuple_list,
    inliner_classes, outlier_classes, outlier_percent, target, is_mrx, subject_folder, threshold_percent=None,
):
    true_positive, true_negative, false_negative, false_positive = target_confusion_components

    try:
        f1 = GetF1(true_positive, false_positive, false_negative)
    except Exception:
        f1 = 0
    count = len(test_result_tuple_list)
    accuracy = 100 * (true_positive + true_negative) / count

    # X1: valid set threshold 기준으로 True라고 간주한 test set의 p들
    # Y1: valid set threshold 기준으로 False라고 간주한 test set의 p들
    X1, Y1 = [], []
    y_true, y_scores = [], []

    for x in test_result_tuple_list:
        if x[0]:
            X1.append(x[1])
        else:
            Y1.append(x[1])

        y_true.append(x[0])
        y_scores.append(x[1])

    # x, y, thres = roc_curve(y_true, y_scores)
    # import matplotlib.pyplot as plt
    # plt.plot(x, y)
    # plt.savefig("a.png")

    try:
        if target == "outlier":
            y_scores = [score*-1 for score in y_scores]
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0

    test_result_tuple_list = sorted(test_result_tuple_list, key=lambda x:x[1])
    minP = test_result_tuple_list[0][1] - 1
    maxP = test_result_tuple_list[-1][1] + 1

    # FPR at TPR 95
    fpr95 = 0.0
    clothest_tpr = 1.0
    dist_tpr = 1.0

    # Detection error
    detection_error = 1.0

    # AUPR IN
    auprin = 0.0
    recallTemp = 1.0

    auprin_recall = []
    auprin_precision = []
    for e in np.arange(minP, maxP, 0.2):
        # FPR at TPR 95
        if target == "inlier":
            tp = np.sum(np.greater_equal(X1, e))
            fp = np.sum(np.greater_equal(Y1, e))
            tpr = tp / np.float(len(X1))
            fpr = fp / np.float(len(Y1))
        else:
            tp = np.sum(np.less(X1, e))
            fp = np.sum(np.less(Y1, e))
            tpr = tp / np.float(len(X1))
            fpr = fp / np.float(len(Y1))
        if abs(tpr - 0.95) < dist_tpr:
            dist_tpr = abs(tpr - 0.95)
            clothest_tpr = tpr
            fpr95 = fpr

        # Detection error
        if target == "inlier":
            fnr = np.sum(np.less(X1, e)) / np.float(len(X1))
            fpr = np.sum(np.greater_equal(Y1, e)) / np.float(len(Y1))
        else:
            fnr = np.sum(np.greater_equal(X1, e)) / np.float(len(X1))
            fpr = np.sum(np.less(Y1, e)) / np.float(len(Y1))
        detection_error = np.minimum(detection_error, (fnr + fpr) / 2.0)

        # AUPR IN
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        auprin_precision.append(precision)
        auprin_recall.append(recall)
    auprin = metrics.auc(auprin_recall, auprin_precision)


    # AUPR OUT
    auprout = 0.0
    minP, maxP = -maxP, -minP
    X1 = [-x for x in X1]
    Y1 = [-x for x in Y1]
    recallTemp = 1.0

    auprout_recall = []
    auprout_precision = []
    for e in np.arange(minP, maxP, 0.2):
        if target == "inlier":
            tp = np.sum(np.greater_equal(Y1, e))
            fp = np.sum(np.greater_equal(X1, e))
        else:
            tp = np.sum(np.less(Y1, e))
            fp = np.sum(np.less(X1, e))

        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(Y1))
        auprout_precision.append(precision)
        auprout_recall.append(recall)
    auprout = metrics.auc(auprout_recall, auprout_precision)

    msg = (
        "Target: {}\nThreshold Percent: {}\n"
        "Class: {}\nOutlier Percent: {}\nAccuracy: {}\n"
        "F1: {}\nAUC: {}\nfpr95: {}\n"
        "Detection error: {}\nauprin: {}\nauprout: {}\n\n"
    ).format(
        target, threshold_percent,
        outlier_classes[0] if is_mrx else inliner_classes[0],
        outlier_percent, accuracy,
        f1, auc, fpr95,
        detection_error, auprin, auprout
    )
    print(msg)

    with open(os.path.join(subject_folder, "results_new.txt"), "a") as file:
        file.write(msg)


def compute_threshold(valid_result_tuple_list, target):
    result = valid_result_tuple_list
    random.shuffle(result)

    result_p = np.asarray([r[1] for r in result], dtype=np.float32)
    minP = min(result_p) - 1
    maxP = max(result_p) + 1

    if target == "inlier":
        is_inliers_np = np.asarray([r[0] for r in result], dtype=np.float32)
        is_outliers_np = np.logical_not(is_inliers_np)
    elif target == "outlier":
        is_outliers_np = np.asarray([r[0] for r in result], dtype=np.float32)
        is_inliers_np = np.logical_not(is_outliers_np)
    else:
        raise Exception("Wrong target")

    best_threshold = 0
    best_f = 0
    best_threshold_ = 0
    best_f_ = 0

    for threshold in np.arange(minP, maxP, 0.1):
        if target == "inlier":
            y = np.greater(result_p, threshold)
            true_positive = np.sum(np.logical_and(y, is_inliers_np))
            false_positive = np.sum(np.logical_and(y, is_outliers_np))
            false_negative = np.sum(np.logical_and(np.logical_not(y), is_inliers_np))
        elif target == "outlier":
            y = np.less(result_p, threshold)
            true_positive = np.sum(np.logical_and(y, is_outliers_np))
            false_positive = np.sum(np.logical_and(y, is_inliers_np))
            false_negative = np.sum(np.logical_and(np.logical_not(y), is_outliers_np))

        if true_positive > 0:
            try:
                f = GetF1(true_positive, false_positive, false_negative)
            except:
                f = 0

            if f > best_f:
                best_f = f
                best_threshold = threshold
            if f >= best_f_:
                best_f_ = f
                best_threshold_ = threshold

    best_threshold = (best_threshold + best_threshold_) / 2.0

    print("Best threshold: ", best_threshold)

    return best_threshold


if __name__ == "__main__":
    import sys
    arg_list = sys.argv[1:]

    try:
        test_fold_id = int(arg_list[0])
        if test_fold_id >= 5:
            raise Exception("Invalid test_fold_id (should be lower than 5)")
    except:
        raise Exception("You missed test_fold_id!")

    try:
        is_fashion = "fashion" in arg_list[1].lower()
    except:
        raise Exception("You missed is_fashion!")

    try:
        is_mrx = "mrx" in arg_list[2].lower()
    except:
        raise Exception("You missed is_mrx!")

    np.random.seed(2019)
    random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    subject_folder = "test_fold_{}".format(test_fold_id)
    if is_fashion:
        subject_folder = "f_" + subject_folder

    if is_mrx:
        threshold_percent_list = [0.5, 2, 3.5, 5]

        #
        # MRX은 validation에 outlier percentage가 필요없음 -> validation result file outlier percentage 빼는 전처리 안한경우, 여기서 됨
        #
        val_result_file_list = []
        for f in os.listdir(subject_folder):
            if "validation" in f:
                components = f.split("_")
                if len(components) == 4:
                    f = "_".join(components[:2] + components[3:])
                val_result_file_list.append(os.path.join(subject_folder, f))
        val_result_file_list = sorted(val_result_file_list)

        for val_result_file in val_result_file_list:
            # val_result_file 내에 in or outlier digit이 포함 안되어있음
            valid_result_tuple_list, inliner_classes, outlier_classes, _ = get_valid_result_tuple_list_and_inout_classes_and_percent(
                val_result_file, is_mrx
            )
            p_list = [p for _, p in valid_result_tuple_list]
            thres_and_thres_p_list = [(np.quantile(p_list, thres_p/100), thres_p) for thres_p in threshold_percent_list]

            #
            # "in/outlier class"만 같은 test set만 불러오기(outlier percentage는 필요없음 -> 어차피 validation_set에 outlier가 없기 때문)
            #
            test_result_file_path = [
                os.path.join(subject_folder, f) for f in os.listdir(subject_folder)
                if f.startswith("result_{}".format(outlier_classes[0] if is_mrx else inliner_classes[0]))
                if "validation" not in f
            ]
            test_result_file_path = sorted(test_result_file_path)

            for test_result_path in test_result_file_path:
                outlier_percent = int(test_result_path.split("/")[1].split("_")[2])
                with open(test_result_path, 'rb') as pkl:
                    # (is_inlier, p value)  // 모델 재활용으로인해 is_inlier로 저장되어있음!
                    test_result_tuple_list = pickle.load(pkl)
                    inlier_target_test_result_tuple_list = test_result_tuple_list
                    outlier_target_test_result_tuple_list = [
                        (not is_inlier, p) for is_inlier, p in inlier_target_test_result_tuple_list
                    ]

                for threshold, threshold_p in thres_and_thres_p_list:
                    inlier_target_confusion_components = get_confusion_components(
                        inlier_target_test_result_tuple_list, threshold, "inlier",
                    )
                    outlier_target_confusion_components = get_confusion_components(
                        outlier_target_test_result_tuple_list, threshold, "outlier",
                    )

                    calculate_performance_metrics(
                        inlier_target_confusion_components, inlier_target_test_result_tuple_list,
                        inliner_classes, outlier_classes, outlier_percent, 'inlier', is_mrx, subject_folder, threshold_p
                    )
                    calculate_performance_metrics(
                        outlier_target_confusion_components, outlier_target_test_result_tuple_list,
                        inliner_classes, outlier_classes, outlier_percent, 'outlier', is_mrx, subject_folder, threshold_p
                    )
    else:
        # 2. Get threshold using best f1
        subject_folder = "original_" + subject_folder
        val_result_file_list = sorted([os.path.join(subject_folder, f) for f in os.listdir(subject_folder) if "validat" in f])
        for val_result_file in val_result_file_list:
            # val_result_file 내에 in or outlier digit이 포함되어있음
            inlier_target_valid_result_tuple_list, inliner_classes, outlier_classes, outlier_percent = get_valid_result_tuple_list_and_inout_classes_and_percent(
                val_result_file, is_mrx
            )
            outlier_target_valid_result_tuple_list = [(not is_inlier, p) for is_inlier, p in inlier_target_valid_result_tuple_list]

            #
            # "in/outlier class & outlier percentage"가 같은 test set만 불러오기 (1개일것임)
            #
            test_result_file_path = [
                os.path.join(subject_folder, f) for f in os.listdir(subject_folder)
                if f.startswith("result_{}_{}".format(outlier_classes[0] if is_mrx else inliner_classes[0], outlier_percent))
                if "validation" not in f
            ]
            test_result_file_path = sorted(test_result_file_path)

            assert len(test_result_file_path) == 1
            test_result_path = test_result_file_path[0]
            with open(test_result_path, 'rb') as pkl:
                # (is_inlier, p value)  // 모델 재활용으로인해 is_inlier로 저장되어있음!
                inlier_target_test_result_tuple_list = pickle.load(pkl)
                outlier_target_test_result_tuple_list = [(not is_inlier, p) for is_inlier, p in inlier_target_test_result_tuple_list]

            inlier_threshold = compute_threshold(inlier_target_valid_result_tuple_list, "inlier")
            outlier_threshold = compute_threshold(outlier_target_valid_result_tuple_list, "outlier")

            inlier_target_confusion_components = get_confusion_components(
                inlier_target_test_result_tuple_list, inlier_threshold, "inlier"
            )
            outlier_target_confusion_components = get_confusion_components(
                outlier_target_test_result_tuple_list, outlier_threshold, "outlier"
            )

            calculate_performance_metrics(
                inlier_target_confusion_components, inlier_target_test_result_tuple_list,
                inliner_classes, outlier_classes, outlier_percent, 'inlier', is_mrx, subject_folder, None
            )
            calculate_performance_metrics(
                outlier_target_confusion_components, outlier_target_test_result_tuple_list,
                inliner_classes, outlier_classes, outlier_percent, 'outlier', is_mrx, subject_folder, None
            )
