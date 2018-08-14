#!/usr/bin/python
# coding: utf-8
# @Time    : 2018/8/14 16:11
# @Author  : Ye Wang (Wane)
# @Email   : y.wang@newdegreetech.com
# @File    : simulate.py
# @Software: PyCharm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def generate_fake_signal(sigma, drift_max, drift_momentum):
    m = 1000
    # sigma = 3
    # drift_max = 0.1
    # drift_momentum = 0.8
    noise = np.squeeze(np.random.randn(m, 1) * sigma)
    drift_slp = np.squeeze(np.random.rand(m, 1) * 2 * drift_max - drift_max)
    for ii in range(1, len(drift_slp)):
        drift_slp[ii] += (drift_slp[ii - 1] - drift_slp[ii]) * drift_momentum
    drift_integral = np.cumsum(drift_slp)
    # plt.subplot(311)
    # plt.plot(noise)
    # plt.ylabel('noise: ADC')
    # plt.subplot(312)
    # plt.plot(drift_integral)
    # plt.ylabel('drift: ADC')
    # plt.subplot(313)
    plt.plot(drift_integral + noise)
    plt.ylabel('signal: ADC')
    # plt.title('Signal Decomposition')
    # plt.show()
    return drift_integral + noise


def calc_upper_slp_and_integral(rawdata, slp_order, integral_order):
    raw_slp_history = rawdata[:-slp_order]
    raw_slp_new = rawdata[slp_order:]
    raw_integral_history = rawdata[:-integral_order]
    raw_integral_new = rawdata[integral_order:]
    slp_arr = np.abs(raw_slp_new - raw_slp_history)
    integral_arr = np.abs(raw_integral_new - raw_integral_history)
    # plt.subplot(211)
    # plt.hist(slp_arr, 100)
    # plt.vlines(np.percentile(slp_arr, 99), 0, 400)
    # plt.ylabel('slp')
    # plt.subplot(212)
    # plt.hist(integral_arr, 100)
    # plt.vlines(np.percentile(integral_arr, 99), 0, 400)
    # plt.ylabel('integral')
    # plt.suptitle('Slp and Integral almost i.i.d.')
    # plt.show()
    return np.percentile(integral_arr, 99)


def generate_fake_model(lf, rf, decay, gap, wn):
    # lf = 60
    # rf = 200
    # decay = 5
    # gap = 8
    # wn = 5  # std
    gaussian = lambda x, p, wn: \
        p[0] * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2)) + \
        np.squeeze(np.random.randn(len(x), 1) * wn)
    xx = np.arange(-gap, 4 * gap, 0.1)
    sensor1 = gaussian(xx, (lf, 0, decay), wn)
    sensor2 = gaussian(xx, (rf, gap, decay), wn)
    sensor3 = gaussian(xx, (rf, 2 * gap, decay), wn)
    sensor4 = gaussian(xx, (lf, 3 * gap, decay), wn)
    plt.plot(xx, sensor1)
    plt.plot(xx, sensor2)
    plt.plot(xx, sensor3)
    plt.plot(xx, sensor4)
    plt.vlines(0, 0, np.max((sensor1, sensor2)), linestyles='--')
    plt.vlines(3 * gap, 0, np.max((sensor1, sensor2)), linestyles='--')
    plt.title('Signal vs. Position')
    plt.xlabel('position: mm')
    plt.ylabel('signal: ADC')
    # plt.show()
    model_xx = xx
    # model_xx = np.arange(0, 3 * gap, 0.1)
    model = np.hstack((gaussian(model_xx, (lf, 0, decay), 0)[:, np.newaxis],
                       gaussian(model_xx, (rf, gap, decay), 0)[:, np.newaxis],
                       gaussian(model_xx, (rf, 2 * gap, decay), 0)[:, np.newaxis],
                       gaussian(model_xx, (lf, 3 * gap, decay), 0)[:, np.newaxis],
                       ))
    real_sensor = np.hstack((gaussian(model_xx, (lf, 0, decay), wn)[:, np.newaxis],
                             gaussian(model_xx, (rf, gap, decay), wn)[:, np.newaxis],
                             gaussian(model_xx, (rf, 2 * gap, decay), wn)[:, np.newaxis],
                             gaussian(model_xx, (lf, 3 * gap, decay), wn)[:, np.newaxis],
                             ))
    normalization = lambda y: np.apply_along_axis(lambda x: x / np.max(np.abs(x)), 1, y)
    return model_xx, \
           normalization(model), \
           normalization(real_sensor)


def calc_pos(model, data):
    m = np.shape(model)[0]
    dist_mat = np.tile(data, (m, 1))
    dist_mat = (dist_mat - model)
    dist_mat_square_sum = np.sum(dist_mat ** 2, axis=1)
    min_dist_10_val = np.sort(dist_mat_square_sum)[:10]
    min_dist_10_val = min_dist_10_val / (np.min(min_dist_10_val) + 1)
    W_min_10 = np.exp(-200 * (min_dist_10_val - 1) ** 2)
    min_dist_10_ind = np.argsort(dist_mat_square_sum)[:10]
    weighted_pos = np.sum(min_dist_10_ind * W_min_10) / np.sum(W_min_10)
    return weighted_pos


def map_pos(rgA_l, rgA_r, rgB_l, rgB_r, pos):
    return rgB_l + pos * (rgB_r - rgB_l) / (rgA_r - rgA_l)


def calc_positions(real_x, model, data):
    # plt.subplot(211)
    # plt.plot(model)
    # plt.subplot(212)
    # plt.plot(data)
    # plt.show()
    postions = np.apply_along_axis(
        lambda x: calc_pos(model, x),
        1, data)
    postions = map_pos(0, np.shape(data)[0],
                       np.min(real_x), np.max(real_x),
                       postions)
    plt.plot(real_x, postions)
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('real position')
    plt.ylabel('calc position')
    plt.title('Real position vs. Calculated position')
    # plt.show()


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Consolas'
    plt.rcParams['font.size'] = 20
    fig = plt.figure()
    fig.set_size_inches(60, 10)
    while True:
        params = pd.read_table('hyper_params.txt', delimiter='\t').ix[:, :-1]
        module_noise_sigma = np.squeeze(params['module_noise_sigma'])
        module_drift_max = np.squeeze(params['module_drift_max'])
        module_drift_momentum = np.squeeze(params['module_drift_momentum'])
        structure_little_finger = np.squeeze(params['structure_little_finger'])
        structure_ring_finger = np.squeeze(params['structure_ring_finger'])
        structure_decay = np.squeeze(params['structure_decay'])
        structure_gap = np.squeeze(params['structure_gap'])
        structure_wn_of_str = np.squeeze(params['structure_noise_WN'])

        plt.subplot(221)
        rawdata = generate_fake_signal(module_noise_sigma, module_drift_max, module_drift_momentum)
        min_max_integral = calc_upper_slp_and_integral(rawdata, 3, 15)
        plt.title('Signal Decomposition ' + '%.2f' % min_max_integral)
        # print(min_max_integral)
        plt.subplot(223)
        model_x, model, real_data = generate_fake_model(
            structure_little_finger,
            structure_ring_finger,
            structure_decay,
            structure_gap,
            structure_wn_of_str
        )
        plt.subplot(122)
        calc_positions(model_x, model, real_data)
        plt.suptitle('Intuition on how hyper parameters influence performance')
        plt.pause(0.2)
        plt.clf()
        # time.sleep(0.1)
