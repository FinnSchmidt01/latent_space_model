from collections import Counter
import numpy as np
import os
from pandas import read_parquet
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# install('fastparquet')
install('pyarrow')


CHALLENGE = 'main'  # could be ood


group_main = {'dev': {'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'ssIew7vsoDwR9wnG3uWk': [40, 109, 130, 131, 150, 176, 268, 290, 547, 670],
    '/FbOh2ePgMqrdXpJC07P': [7,  53,  84, 194, 226, 264, 286, 323, 360, 692],
    'QdH3ZwzfCrhC3hw15J29': [17,  73, 173, 197, 301, 429, 544, 552, 593, 650],
    'MAF2j097Q4BHKMgtWlWe': [58, 168, 221, 413, 436, 555, 602, 609, 639, 663],
    'oIC+bpSFZVqMyXuw7FzU': [5,  51, 135, 183, 225, 285, 287, 394, 619, 689],
    'B1hl1Lfb+NWPIqMXK4Ya': [19,  22, 158, 233, 374, 422, 575, 605, 712, 719]},
    'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'aLRPqBKV3Hf5siSSHmeC': [102, 114, 181, 250, 342, 385, 393, 439, 487, 586],
    'A/SBLsELSUsBelyv+BWg': [20,  60,  75, 109, 190, 233, 529, 599, 631, 648],
    'x5IhS8u87xOEcMtLynwD': [101, 211, 212, 298, 313, 360, 372, 390, 468, 520],
    'af1Vnuj0UJIxBh2LduXb': [28,  76,  99, 139, 222, 225, 333, 377, 539, 642],
    'usKAtMHbsh05G8B6aBjK': [24, 136, 320, 329, 365, 423, 496, 504, 552, 574],
    'eBkKTNkWPAxeDI7ALWkE': [110, 133, 184, 339, 530, 548, 572, 595, 637, 644]},
    'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'yGA23llYm0Ftzh9bwyy5': [70,  82, 223, 412, 427, 443, 516, 550, 559, 628],
    '0+1ayawyVQ4Akj2DSqbo': [18,  21, 146, 215, 343, 386, 526, 554, 656, 661],
    'edHuKTOVWNPxaMAAhu/O': [7,  49,  77, 176, 208, 242, 262, 297, 331, 636],
    'W98svESnqbEhvA27hl36': [9,  53, 139, 150, 226, 269, 351, 359, 541, 639],
    'RlSN8nU2OBR57BvGKozk': [40,  93, 175, 233, 330, 388, 492, 530, 545, 662],
    'SMu+HtH1qDOoPc4LWCQH': [87,  92, 192, 210, 260, 314, 352, 587, 615, 673]},
    'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'uODHIDAmzGEdNIR/7CU+': [4,  19, 141, 180, 191, 198, 263, 338, 461, 526],
    'BWYoVyTZmAAeGb/+FAN5': [131, 149, 231, 241, 291, 345, 418, 440, 570, 653],
    '4f2CDNju0w5mNFOkcHh6': [110, 133, 184, 339, 530, 548, 572, 595, 637, 644],
    'iJDjTZ10zhFV+ILMn1ZP': [89, 146, 193, 273, 353, 355, 443, 544, 560, 613],
    'G1FQEbvKcBqLVy10JNC6': [82, 189, 195, 252, 295, 321, 470, 531, 573, 604],
    'qJyt7U28CfgWpTGF66Lb': [9,  48, 118, 153, 221, 406, 433, 449, 467, 486]},
    'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce': {
    'OUnuqMzSI/k8J4O7ao/+': [7,  53,  84, 194, 226, 264, 286, 323, 360, 691],
    'EN+BEEINDVoYTogjZuSu': [77,  89, 242, 453, 468, 486, 565, 600, 610, 683],
    '97DG676R2J1wTHpad6uq': [4,  20,  50, 162, 260, 355, 481, 591, 671, 685],
    'TC6qD2MF/2yNeQ1lJrXQ': [40, 109, 130, 131, 150, 176, 268, 290, 547, 669],
    '1onnD5tvnWs55ui1fddC': [33,  63, 122, 139, 178, 184, 190, 265, 377, 507],
    '78vrQM1WMAroVSRnSo1k': [13,  90, 146, 175, 205, 460, 482, 664, 686, 695]}},

    'final': {'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce': {
        'CZ+DhtMljSpNJoGC7uee': [74,  79, 114, 116, 227, 231, 298, 415, 551, 641],
        'Jeu3DMpQlmPiqhMW1SKl': [152, 201, 219, 371, 428, 487, 529, 580, 638, 643],
        '760N+YRVyXPlY7zG6Dcs': [33,  63, 122, 139, 178, 184, 190, 265, 377, 507],
        'oIHPXud6r92YFKjXxvz2': [95, 101, 210, 228, 284, 342, 383, 640, 668, 733],
        '9NUudHK3AYroIBWxG5ny': [4,  20,  50, 162, 260, 355, 481, 592, 672, 686],
        '30rqODn0zoFKgVy4rcbr': [34, 213, 246, 257, 350, 378, 423, 466, 612, 705]},
    'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce': {
        'lyuefUNnplbYN+QkTXon': [11,  15,  29,  36, 173, 177, 260, 387, 389, 432],
        '7HuSlLAHKBHL+xMlI/hT': [131, 149, 231, 241, 291, 345, 418, 440, 570, 653],
        'wWJ/Ve5GIXFs9p+FMjYJ': [18, 107, 201, 304, 391, 414, 519, 545, 569, 646],
        'ui1SwoX4WBvQtAA4Kbge': [89, 146, 193, 273, 353, 355, 443, 544, 560, 613],
        'knL1XdNd2VTq5yCQKuYI': [13,  31,  45,  67, 223, 425, 476, 481, 641, 655],
        'F4lMA0Un+T22TkvpduV2': [9,  48, 118, 153, 221, 406, 433, 449, 467, 486]},
    'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce': {
        'Sb2Jjz5IbmYlromle+Qj': [4,  19,  46, 149, 238, 326, 438, 542, 618, 630],
        'GKJOalC+FWQTU8mSxXvB': [178, 197, 202, 232, 309, 450, 459, 490, 593, 678],
        'RKYc2wP07YUs0OgX9o7B': [16,  67, 158, 179, 277, 393, 495, 503, 543, 597],
        'hxYOd5cdaRTPLQRt+Fw0': [5,  47, 123, 166, 207, 261, 263, 360, 566, 633],
        'h3xGHFGClfLgJSyzuoCl': [31,  57, 112, 127, 163, 167, 172, 243, 346, 462],
        'Cn+BFK0m9afMoXg5Rsd5': [37, 100, 118, 119, 138, 161, 246, 266, 498, 616]},
    'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce': {
        'IzeO9dDRaB9s5jEem3NS': [152, 240, 246, 303, 450, 458, 534, 571, 581, 617],
        'ru1ZEKHmUXylYmbkH/wr': [28,  76,  99, 139, 222, 225, 333, 377, 539, 642],
        'Iw9luXyddOe6iA1gViVr': [13,  31,  45,  67, 223, 425, 476, 481, 641, 655],
        '8CNUwkIDKg8J640AIzPQ': [102, 114, 181, 250, 342, 385, 393, 439, 487, 586],
        'f624gHhnXnp8YBXSU2IP': [20,  60,  75, 109, 190, 233, 529, 599, 631, 648],
        'YmfSjn7YgOfV2k6RbHMX': [18, 107, 201, 304, 391, 414, 519, 545, 569, 646]},
    'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce': {
        '6E86vxKi5MnZieARC5KE': [9,  59, 151, 163, 245, 293, 382, 393, 590, 694],
        'IH4V2RFaQAvgYnlqV25d': [43, 102, 193, 254, 359, 424, 541, 578, 594, 719],
        'X4ON5q2WFxEawHsaLiRx': [196, 215, 220, 252, 337, 493, 504, 539, 645, 737],
        '1pZGWbf83r7c60rZ+GWY': [218, 263, 349, 364, 404, 494, 516, 533, 598, 655],
        'EUDYW+miI0Ha9F4KTzFp': [17,  73, 173, 197, 301, 429, 544, 552, 592, 649],
        'IREheNuMTpATrV9/euLP': [19,  22, 158, 233, 374, 422, 575, 604, 711, 718]}}}





def corr(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """
    adopted from here https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/measures/np_functions.py#LL13C1-L31C47
    Compute the correlation between two NumPy arrays along the specified dimension(s).

    Args:
        y1:      first NumPy array
        y2:      second NumPy array
        axis:    dimension(s) along which the correlation is computed. Any valid NumPy axis spec works here
        eps:     offset to the standard deviation to avoid exploding the correlation due to small division (default 1e-8)
        **kwargs: passed to final numpy.mean operation over standardized y1 * y2

    Returns: correlation array
    """
    print(y1.shape, y2.shape)
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / \
        (y1.std(axis=axis, keepdims=True, ddof=0) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / \
        (y2.std(axis=axis, keepdims=True, ddof=0) + eps)
    corrs = (y1 * y2).mean(axis=axis, **kwargs)
    print(corrs.shape)
    if np.any(np.isnan(corrs)):
        print(f'{np.isnan(corrs).mean() * 100}% NaNs , NaNs will be set to Zero.')
        corrs[np.isnan(corrs)] = 0
    return np.mean(corrs)


def single_trial_correlation_from_submissions(predictions, solutions, track=CHALLENGE, mode='dev'):

    def correlation_per_mouse(solutions, predictions, mouse):
        mouse_sol = solutions[solutions['mouse'] ==
                              mouse].sort_values(by=['trial_indices'])
        mouse_pred = predictions[predictions['mouse']
                                 == mouse].sort_values(by=['trial_indices'])
        # here after concatenation its like (all_frames, number of neurons) like (13944, 7440)
        mouse_sol = np.concatenate(mouse_sol['prediction'].tolist())
        mouse_pred = np.concatenate(mouse_pred['prediction'].tolist())
        c = corr(mouse_sol, mouse_pred, axis=0)
        print(f'correlation shape and value: {m}, {c.shape}, {np.mean(c)}')
        return corr(mouse_sol, mouse_pred, axis=0)

    assert track == 'main' or track == 'ood', 'track should be either "main" or "ood"'
    assert mode == 'dev' or mode == 'final', 'mode should be either "dev" or "final"'

    mice = solutions['mouse'].unique().tolist()
    if track == 'main':
        correlations = 0
        for mouse in mice:
            correlations += correlation_per_mouse(
                solutions, predictions, mouse)
        return correlations / len(mice)
    else:
        # compute independently each mice and each ood
        # average accross mice per oods
        correlations = {}
        if mode == 'dev':
            # no need to average per mice
            for mouse in mice:
                correlations[live_test_mouse_per_ood[mouse]] = correlation_per_mouse(
                    solutions, predictions, mouse)
            return correlations
        else:

            for ood in final_test_ood_mouse_files.keys():
                correlations[ood] = 0
                for mouse in final_test_ood_mouse_files[ood].keys():
                    mouse_sol = solutions[solutions['mouse'] == mouse].sort_values(by=[
                                                                                   'trial_indices'])
                    mouse_pred = predictions[predictions['mouse'] == mouse].sort_values(by=[
                                                                                        'trial_indices'])

                    # subselect correct indexes for the ood
                    idxs = final_test_ood_mouse_files[ood][mouse]
                    mouse_sol = mouse_sol[mouse_sol['trial_indices'].isin(
                        idxs)]
                    mouse_pred = mouse_pred[mouse_pred['trial_indices'].isin(
                        idxs)]

                    mouse_sol = np.concatenate(
                        mouse_sol['prediction'].tolist())
                    mouse_pred = np.concatenate(
                        mouse_pred['prediction'].tolist())
                    correlations[ood] += corr(mouse_sol, mouse_pred, axis=0)

                correlations[ood] = correlations[ood] / \
                    len(final_test_ood_mouse_files[ood].keys())
            return correlations


def permute_mice_responces(mouse, pred, permutations):
    if permutations[mouse] is not None:
        pred = pred[:, permutations[mouse]]
    else:
        return pred


def correlation_to_average_from_submissions(predictions, solutions, track=CHALLENGE, mode='dev'):
    assert track == 'main' or track == 'ood', 'track should be either "main" or "ood"'
    assert mode == 'dev' or mode == 'final', 'mode should be either "dev" or "final"'

    # Todo - load submission and solutions

    if track == 'main':
        correlations = 0
        for mouse in group_main[mode].keys():
            averaged_target = []
            averaged_submission = []
            per_mouse_pred = predictions[predictions['mouse'] == mouse]
            per_mouse_sol = solutions[solutions['mouse'] == mouse]
            # tier_idx = [x[0] for x in per_mouse_pred['trial_indices'].tolist()]
            for s in group_main[mode][mouse].keys():
                file_ids = group_main[mode][mouse][s]

                # select file ids from submisssion csv per mouse, align and average across responses
                pred_to_avg = np.asarray(per_mouse_pred[per_mouse_pred['trial_indices'].isin(
                    file_ids)].sort_values(by=['trial_indices'])['prediction'].tolist())

                # TODO - remove in actuall full size competition
                if pred_to_avg.size > 0:
                    averaged_submission.append(np.mean(pred_to_avg, axis=0))
#                 averaged_submission.append(np.mean(pred_to_avg, axis=0))

                # select file ids from SOLUTIONS and average accross responses
                sol_to_avg = np.asarray(per_mouse_sol[per_mouse_sol['trial_indices'].isin(
                    file_ids)].sort_values(by=['trial_indices'])['prediction'].tolist())

                # TODO - remove in actuall full size competition
                if sol_to_avg.size > 0:
                    averaged_target.append(np.mean(sol_to_avg, axis=0))
#                 averaged_target.append(np.mean(sol_to_avg, axis=0))

            # stack along averaged submissions and targets
            target = np.concatenate(averaged_target)
            # target = np.mean(target, axis=0)
            output = np.concatenate(averaged_submission)
            # output = np.mean(output, axis=0)
            # compute correlations
            correlations += corr(target, output, axis=0)
        return correlations / len(group_main[mode].keys())

    else:
        correlations = {}
        if mode == 'dev':
            for mouse in group_ood_live.keys():
                averaged_target = []
                averaged_submission = []
                per_mouse_pred = predictions[predictions['mouse'] == mouse]
                per_mouse_sol = solutions[solutions['mouse'] == mouse]
                for stim_hash in group_ood_live[mouse].keys():
                    file_ids = group_ood_live[mouse][stim_hash]

                    # select file ids from submisssion csv per mouse, align and average accross responses
                    pred_to_avg = np.asarray(per_mouse_pred[per_mouse_pred['trial_indices'].isin(
                        file_ids)].sort_values(by=['trial_indices'])['prediction'].tolist())

                    # TODO - remove in actuall full size competition
                    if pred_to_avg.size > 0:
                        averaged_submission.append(
                            np.mean(pred_to_avg, axis=0))
                    # averaged_submission.append(np.mean(pred_to_avg, axis=0))

                    # select file ids from SOLUTIONS and average accross responses
                    sol_to_avg = np.asarray(per_mouse_sol[per_mouse_sol['trial_indices'].isin(
                        file_ids)].sort_values(by=['trial_indices'])['prediction'].tolist())

                    # TODO - remove in actuall full size competition
                    if sol_to_avg.size > 0:
                        averaged_target.append(np.mean(sol_to_avg, axis=0))
                    # averaged_target.append(np.mean(sol_to_avg, axis=0))

                # stack along averaged submissions and targets
                target = np.concatenate(averaged_target)
                output = np.concatenate(averaged_submission)
                # compute correlations
                correlations[live_test_mouse_per_ood[mouse]] = corr(
                    target, output, axis=0)
            return correlations
        else:
            for ood in group_ood_final.keys():
                correlations[ood] = 0
                for mouse in group_ood_final[ood].keys():
                    averaged_target = []
                    averaged_submission = []
                    per_mouse_pred = predictions[predictions['mouse'] == mouse]
                    per_mouse_sol = solutions[solutions['mouse'] == mouse]

                    for stim_hash in group_ood_final[ood][mouse].keys():
                        file_ids = group_ood_final[ood][mouse][stim_hash]
                        # select file ids from submisssion csv per mouse, align and average accross responses
                        pred_to_avg = np.asarray(per_mouse_pred[per_mouse_pred['trial_indices'].isin(
                            file_ids)].sort_values(by=['trial_indices'])['prediction'].tolist())

                        # TODO - remove in actuall full size competition
                        if pred_to_avg.size > 0:
                            averaged_submission.append(
                                np.mean(pred_to_avg, axis=0))
#                         averaged_submission.append(np.mean(pred_to_avg, axis=0))

                        # select file ids from SOLUTIONS and average accross responses
                        # should be sth like (10, 249, 7400)
                        sol_to_avg = np.asarray(per_mouse_sol[per_mouse_sol['trial_indices'].isin(
                            file_ids)].sort_values(by=['trial_indices'])['prediction'].tolist())

                        # TODO - remove in actuall full size competition
                        if sol_to_avg.size > 0:
                            averaged_target.append(np.mean(sol_to_avg, axis=0))
#                         averaged_target.append(np.mean(sol_to_avg, axis=0))

                    # todo - check dimensions
                    target = np.concatenate(averaged_target)
                    output = np.concatenate(averaged_submission)
                    correlations[ood] += corr(target, output, axis=0)
                correlations[ood] = correlations[ood] / \
                    len(group_ood_final[ood].keys())
            return correlations


if __name__ == "__main__":
    # Parse arguments.
    assert len(
        os.sys.argv) == 3, "Usage: score.py <predictions and solutions directory> <output directory>"
    solutions_dir = os.path.join(os.sys.argv[1], "ref")
    predictions_dir = os.path.join(os.sys.argv[1], "res")
    output_dir = os.sys.argv[2]

    # load files
    predictions = read_parquet(os.path.join(
        predictions_dir, 'predictions_live_main.parquet.brotli'))
    solutions = read_parquet(os.path.join(
        solutions_dir, 'solutions_live_main.parquet.brotli'))

    # transform predictions column to np arrays
    # after here in each line of ['prediction'] there is a np array of size (frames, number_of_neurons) like (249, 7440)

    # UNCOMMENT IN REAL scoring
    print(solutions['prediction'].head())
    print(predictions['prediction'].head())

    print(predictions.dtypes)
    print(solutions.dtypes)
    
    # solutions['prediction'] = solutions['prediction'].apply(lambda x: np.asarray(x.tolist()).T)
    # predictions['prediction'] = predictions['prediction'].apply(lambda x: np.asarray(x.tolist()).T)

    # ## COMMENT IN REAL scoring
    solutions['prediction'] = solutions['prediction'].apply(lambda x : np.asarray(x.tolist()).T)
    predictions['prediction'] = predictions['prediction'].apply(lambda x : np.asarray(x.tolist()).T)

    # assert mice are same in prediction and solutions
    assert set(solutions['mouse'].unique().tolist()) == set(predictions['mouse'].unique().tolist()), \
        f'Mice are different in predictions. Mice column should contain all of the following values {solutions["mouse"].unique().tolist()}'

    assert Counter(solutions['mouse'].tolist()) == Counter(predictions['mouse'].tolist()), \
        f'Make sure you have predicted all files for per mouse, should be {Counter(solutions["mouse"].tolist())} instead {Counter(predictions["mouse"].tolist())}'

    solutions['pairs'] = solutions.apply(lambda x: (x['mouse'], x['trial_indices']), axis=1)
    predictions['pairs'] = predictions.apply(lambda x: (x['mouse'], x['trial_indices']), axis=1)

    assert Counter(solutions['pairs'].tolist()) == Counter(predictions['pairs'].tolist()), \
        f'Make sure you have predicted all files for per mouse, {set(solutions["pairs"].tolist()).difference(set(predictions["pairs"].tolist()))} is missing' + \
        f'{set(predictions["pairs"].tolist()).difference(set(solutions["pairs"].tolist()))} are redundant'
    predictions.drop(['pairs'], axis=1, inplace=True)
    solutions.drop(['pairs'], axis=1, inplace=True)

     # assert all neurons are odered same withing mice
    permutations = {}
    predictions['neuron_ids_check'] = predictions['neuron_ids'].apply(lambda x: str(x))
    for m in predictions['mouse'].unique().tolist():
        sub = predictions[predictions['mouse'] == m]
        assert len(sub['neuron_ids_check'].unique(
            )) == 1, f'Please assure the order of the neurons within single mouse always stay the same, we see issues with mouse {m}'
        neuron_ids_pred_list = sub['neuron_ids'].tolist()[0]
        neuron_ids_orig_list = solutions[solutions['mouse'] == m]['neuron_ids'].tolist()[0]

        neuron_ids_pred = set(neuron_ids_pred_list)
        neuron_ids_orig = set(neuron_ids_orig_list)
        assert neuron_ids_orig == neuron_ids_pred, f'Make sure all of the neurons are predicted for mouse {m}, ' +\
            f'{neuron_ids_orig.difference(neuron_ids_pred)} are missing, {neuron_ids_pred.difference(neuron_ids_orig)} are redundant' + \
            'Make sure your neuron_ids is a list of int'

        assert len(neuron_ids_pred_list) == len(
            neuron_ids_pred), 'Check your neuronal ids for duplicates / missing neurons'

        neuron_ids_orig_list = np.asarray(neuron_ids_orig_list)
        neuron_ids_pred_list = np.asarray(neuron_ids_pred_list)
        if (neuron_ids_orig_list == neuron_ids_pred_list).all():
            permutations[m] = None
        else:
            perm = []
            for i in neuron_ids_pred_list:
                perm.append(np.where(neuron_ids_orig_list == i)[0][0])
            permutations[m] = perm

    predictions.drop(['neuron_ids_check'], axis=1, inplace=True)

    # TODO add checks for the correct input sizes
    assert predictions.shape == solutions.shape, f'Input file has wrong size, should be {solutions.shape} (rows, columns), get ({predictions.shape}) instead'
    assert set(predictions.columns.values) == set(solutions.columns.values), \
        f'Columns names are wrong, should be {set(solutions.columns.values)} got {set(predictions.columns.values)} instead'

    # check sizes in prediction
    print(solutions['prediction'][0])
    print(predictions['prediction'][0])
    solutions['prediction_shape'] = solutions['prediction'].apply(lambda x: x.shape)
    predictions['prediction_shape'] = predictions['prediction'].apply(lambda x: x.shape)
    solutions.sort_values(by=['mouse', 'trial_indices'], inplace=True)
    predictions.sort_values(by=['mouse', 'trial_indices'], inplace=True)
    sol_pred_shape = solutions['prediction_shape'].tolist()
    pred_pred_shape = predictions['prediction_shape'].tolist()
    for i, j in zip(sol_pred_shape, pred_pred_shape):
        assert i == j, 'Make sure the predictions have correct shapes. Should be np.array with shape (num of neurons, video_frames - 50), like (7400, 250) for a video with 300 frames'
    predictions.drop(['prediction_shape'], axis=1, inplace=True)
    solutions.drop(['prediction_shape'], axis=1, inplace=True)

    predictions['prediction'] = predictions.apply(lambda x: permute_mice_responces(
        x['mouse'], x['prediction'], permutations), axis=1)

    scores = {}
    scores['single_trial_correlation'] = single_trial_correlation_from_submissions(
        predictions, solutions, track=CHALLENGE, mode='dev') + 0.00001

    scores['correlation_to_average'] = correlation_to_average_from_submissions(
        predictions, solutions, track=CHALLENGE, mode='dev')

    # saving results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scores.txt"), "w") as f:
        if CHALLENGE == 'main':
            for metric, value in scores.items():
                f.write(f'{metric}: {value}\n')
        else:
            # ood track
            for metric, ood_dict in scores.items():
                for ood_type, ood_value in ood_dict.items():
                    f.write(f'{metric}-{ood_type}: {ood_value}\n')
                f.write(
                    f'average-{metric}: {sum(ood_dict.values()) / len(ood_dict.values())}')
