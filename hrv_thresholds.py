import os
import sys

sys.path.insert(1, os.path.dirname(sys.path[0]))

import fitparse
from plotnine import ggplot, aes, geom_line, labs, geom_point, scale_y_continuous, scale_color_gradient
import numpy as np
import pandas as pd
import math


def DFA(pp_values, lower_scale_limit, upper_scale_limit):
    scaleDensity = 30 # scales DFA is conducted between lower_scale_limit and upper_scale_limit
    m = 1 # order of polynomial fit (linear = 1, quadratic m = 2, cubic m = 3, etc...)

    # initialize, we use logarithmic scales
    start = np.log(lower_scale_limit) / np.log(10)
    stop = np.log(upper_scale_limit) / np.log(10)
    scales = np.floor(np.logspace(np.log10(math.pow(10, start)), np.log10(math.pow(10, stop)), scaleDensity))
    F = np.zeros(len(scales))
    count = 0

    for s in scales:
        rms = []
        # Step 1: Determine the "profile" (integrated signal with subtracted offset)
        x = pp_values
        y_n = np.cumsum(x - np.mean(x))
        # Step 2: Divide the profile into N non-overlapping segments of equal length s
        L = len(x)
        shape = [int(s), int(np.floor(L/s))]
        nwSize = int(shape[0]) * int(shape[1])
        # beginning to end, here we reshape so that we have a number of segments based on the scale used at this cycle
        Y_n1 = np.reshape(y_n[0:nwSize], shape, order="F")
        Y_n1 = Y_n1.T
        # end to beginning
        Y_n2 = np.reshape(y_n[len(y_n) - (nwSize):len(y_n)], shape, order="F")
        Y_n2 = Y_n2.T
        # concatenate
        Y_n = np.vstack((Y_n1, Y_n2))

        # Step 3: Calculate the local trend for each 2Ns segments by a least squares fit of the series
        for cut in np.arange(0, 2 * shape[1]):
            xcut = np.arange(0, shape[0])
            pl = np.polyfit(xcut, Y_n[cut,:], m)
            Yfit = np.polyval(pl, xcut)
            arr = Yfit - Y_n[cut,:]
            rms.append(np.sqrt(np.mean(arr * arr)))

        if (len(rms) > 0):
            F[count] = np.power((1 / (shape[1] * 2)) * np.sum(np.power(rms, 2)), 1/2)
        count = count + 1

    pl2 = np.polyfit(np.log2(scales), np.log2(F), 1)
    alpha = pl2[0]
    return alpha


def DFA1(pp_values, lower_scale_limit, upper_scale_limit):
    scaleDensity = 30 # scales DFA is conducted between lower_scale_limit and upper_scale_limit
    m = 1 # order of polynomial fit (linear = 1, quadratic m = 2, cubic m = 3, etc...)

    # initialize, we use logarithmic scales
    scales = np.floor(np.logspace(np.log10(lower_scale_limit), np.log10(upper_scale_limit), scaleDensity)).astype('i4')
    F = np.zeros(len(scales))
    # Step 1: Determine the "profile" (integrated signal with subtracted offset)
    y_n = np.cumsum(pp_values - np.mean(pp_values))

    for i, s in enumerate(scales):
        # Step 2: Divide the profile into N non-overlapping segments of equal length s
        n0 = len(pp_values) // s
        shape = (n0, s)
        # beginning to end, here we reshape so that we have a number of segments based on the scale used at this cycle
        Y_n1 = np.reshape(y_n[:n0*s], shape)
        # end to beginning
        Y_n2 = np.reshape(y_n[-n0*s:], shape)
        # concatenate
        Y_n = np.vstack((Y_n1, Y_n2)).T

        # Step 3: Calculate the local trend for each 2Ns segments by a least squares fit of the series
        xcut = np.arange(0, s)
        x = np.vstack(tuple([np.ones_like(xcut),]+ [np.power(xcut, i) for i in range(1, m+1)]))
        beta = np.linalg.solve(x.dot(x.T), x.dot(Y_n))
        arr = x.T.dot(beta)-Y_n
        smse = np.sum(arr * arr)
        F[i] = np.sqrt(smse / (n0 *s* 2))

    x = np.vstack(tuple([np.ones_like(scales), np.log2(scales)]))
    alpha = np.linalg.solve(x.dot(x.T), x.dot(np.log2(F)))[1]
    return alpha


def computeFeatures(df, step=120):
    features = []
    for index in range(0, int(round(np.max(df['timestamp']) / step))):
        array_rr = df.loc[(df['timestamp'] >= (index * step)) & (df['timestamp'] <= (index + 1) * step), 'RR'] * 1000
        # compute heart rate
        heartrate = round(60000 / np.mean(array_rr), 2)
        # compute rmssd
        NNdiff = np.abs(np.diff(array_rr))
        rmssd = round(np.sqrt(np.sum((NNdiff * NNdiff) / len(NNdiff))), 2)
        # compute sdnn
        sdnn = round(np.std(array_rr), 2)
        # dfa, alpha 1
        alpha1 = DFA(array_rr.to_list(), 4, 16)
        # alpha11 = DFA1(array_rr.to_list(), 4, 16)
        # print(f'DFA: {alpha1} {alpha11}')
        # print(f'DFA equal: {alpha11 == alpha1}')

        curr_features = {
            'timestamp': index,
            'heartrate': heartrate,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'alpha1': alpha1,
        }

        features.append(curr_features)

    features_df = pd.DataFrame(features)
    return features_df


def computeFeatures1(RRs, step=120):
    bins = int(np.floor(RRs.sum())) // step
    ind = np.searchsorted(np.cumsum(RRs), np.arange(1, bins+1)*step)
    features = []
    for i, x in enumerate(np.split(RRs, ind)):
        if x.sum() < step/2.:
            continue
        # compute heart rate
        heartrate = 60/x.mean()
        x *= 1000
        # compute rmssd
        NNdiff = np.abs(np.diff(x))
        rmssd = np.sqrt(np.sum((NNdiff * NNdiff) / len(NNdiff)))
        # compute sdnn
        sdnn = np.std(x)
        # dfa, alpha 1
        alpha1 = DFA1(x, 4, 16)

        curr_features = {
            'timestamp': i,
            'heartrate': heartrate,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'alpha1': alpha1,
        }

        features.append(curr_features)

    features_df = pd.DataFrame(features)
    return features_df


def run(entry, out_dir):
    f_dt = entry.name.split('.')[0]
    in_file = fitparse.FitFile(entry.path)

    def path(f_name, f_dt=f_dt, out_dir=out_dir):
        return os.path.join(out_dir, '.'.join(['_'.join([f_dt, f_name]), 'pdf']))

    # load RR intervals from the fit file
    RRs = np.array([RR_interval for record in in_file.get_messages('hrv') for record_data in record for RR_interval in
         record_data.value if RR_interval is not None])

    artifact_correction_threshold = 0.05
    log_ratio = np.ediff1d(np.log(RRs), to_begin=np.ones(1))
    cond = (log_ratio < np.log(1 + artifact_correction_threshold)) & (
                log_ratio > np.log(1 - artifact_correction_threshold))
    RRs = RRs[cond]

    df = pd.DataFrame()
    df['timestamp'] = np.cumsum(RRs)
    df['RR'] = RRs

    (ggplot(df)
     + aes(x='timestamp', y='RR')
     + geom_line()
     + labs(title="RR intervals", x='Seconds', y="Milliseconds")
     ).save(path('rr_intervals'), dpi=600)

    # orig code
    # features_df = computeFeatures(df)
    # print(features_df.head(50))
    features_df = computeFeatures1(RRs)
    print(features_df.head(50))

    # update & cleanup
    threshold_sdnn = 10  # rather arbitrary, based on visual inspection of the data
    features_df = features_df.loc[(0 < features_df['sdnn']) & (features_df['sdnn'] < threshold_sdnn),]

    (ggplot(features_df)
     + aes(x='timestamp', y='alpha1', color='alpha1')
     + geom_point()
     + geom_line()
     + scale_y_continuous(limits=[0, 1.5])
     + scale_color_gradient(low="red", high="yellow", limits=[0, 1.5])
     + labs(title="Alpha 1 as derived from DFA for aerobic threshold estimation. Average alpha 1: " + str(
                round(np.mean(features_df['alpha1']), 2)), x='Window', y="alpha 1")).save(
        path('Alpha1filtered'))

    from sklearn.linear_model import LinearRegression

    length = len(features_df['alpha1'])
    reg = LinearRegression().fit(features_df['alpha1'].values.reshape(length, 1),
                                 features_df['heartrate'].values.reshape(length, 1))
    prediction = reg.predict(np.array(0.75).reshape(1, 1))
    print(math.floor(prediction))

    (ggplot(features_df)
     + aes(x='heartrate', y='alpha1', color='alpha1')
     + geom_point()
     + geom_line()
     + scale_color_gradient(low="red", high="yellow", limits=[0, 1.5])
     + labs(title='Estimated aerobic threshold heart rate (alpha 1 = 0.75): ' + (
                str(math.floor(prediction[0].item()))) + " bpm", x='bpm', y="alpha 1")).save(path('LT1HRprediction'),
                                                                                             dpi=600)


if __name__ == '__main__':
    with os.scandir('C:\\Temp\\Activities\\') as it:
        for entry in it:
            if entry.is_file():
                run(entry, out_dir='C:\\Temp\\HrvAnalysis')
            # break

