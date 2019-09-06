import train_AAE
import novelty_detector
import csv
import random
import torch


def save_results(subject_folder, results, percentages):
    f = open(
        os.path.join(subject_folder, "results.csv"),
        'wt'
    )
    columns = ['Percentage {}'.format(p) for p in percentages]
    writer = csv.writer(f)
    writer.writerow(('F1',))
    writer.writerow(columns)
    maxlength = 0
    for percentage in percentages:
        list = results[percentage]
        maxlength = max(maxlength, len(list))

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_f1 = [f1 for auc, f1, fpr95, accuracy, auprin, auprout in list]
            row.append(res_f1[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('AUC',))
    writer.writerow(columns)

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_auc = [auc for auc, f1, fpr95, accuracy, auprin, auprout in list]
            row.append(res_auc[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('FPR',))
    writer.writerow(columns)

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_fpr95 = [fpr95 for auc, f1, fpr95, accuracy, auprin, auprout in list]
            row.append(res_fpr95[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('Accuracy',))
    writer.writerow(columns)

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_accuracy = [accuracy for auc, f1, fpr95, accuracy, auprin, auprout in list]
            row.append(res_accuracy[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('auprin',))
    writer.writerow(columns)

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_auprin = [auprin for auc, f1, fpr95, accuracy, auprin, auprout in list]
            row.append(res_auprin[r] if len(list) > r else '')
        writer.writerow(tuple(row))

    writer.writerow(('auprout',))
    writer.writerow(columns)

    for r in range(maxlength):
        row = []
        for percentage in percentages:
            list = results[percentage]
            res_auprout = [auprout for auc, f1, fpr95, accuracy, auprin, auprout in list]
            row.append(res_auprout[r] if len(list) > r else '')
        writer.writerow(tuple(row))
    f.close()


if __name__ == '__main__':
    import sys, os
    arg_list = sys.argv[1:]
    print('arg_list : ',arg_list)
    try:
        subject_folder = arg_list[0]
    except:
        raise Exception("You missed subject folder name!")
    if not os.path.exists(subject_folder):
        os.mkdir(subject_folder)
    else:
        raise Exception("Give subject folder name already exists!")

    try:
        gpu_num = int(arg_list[1])
        if gpu_num >= torch.cuda.device_count():
            raise Exception("Invalid gpu number")
    except:
        raise Exception("You missed gpu number!")

    try:
        test_fold_id = int(arg_list[2])
        if test_fold_id >= 5:
            raise Exception("Invalid test_fold_id (should be lower than 5)")
    except:
        raise Exception("You missed test_fold_id!")

    try:
        is_fashion = "fashion" in arg_list[3].lower() #fashion mnist : True
    except:
        raise Exception("You missed is_fashion!")

    try:
        num_epoch = int(arg_list[4])
    except:
        num_epoch = None


    random.seed(2019)
    full_run = True
    results = {}

    percentages = [35]
    for percentage in percentages:
        results[percentage] = []

    valid_fold_id = random.randint(0, 4 if full_run else 1)
    while valid_fold_id == test_fold_id:
        valid_fold_id = random.randint(0, 4 if full_run else 1)

    for i in range(10):
        # print('test_fold_id:',test_fold_id)
        # print('valid_fold_id:',valid_fold_id)
        train_AAE.main(
            test_fold_id, valid_fold_id, subject_folder, gpu_num,
            is_fashion=is_fashion, num_epoch=num_epoch, inliner_classes=[i], total_classes=10, batch_size=256, zsize=32
        )
        res = novelty_detector.main(
            test_fold_id, valid_fold_id, subject_folder, gpu_num,
            is_fashion=is_fashion, inliner_classes=[i], total_classes=10, batch_size=256, zsize=32, percentages=percentages
        )
        for k, v in res.items():
            results[k].append(v)

        save_results(subject_folder, results, percentages)
