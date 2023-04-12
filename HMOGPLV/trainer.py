import sys
import os
import time
# sys.path.append('/home/acp18cm/Second_project/')
import tensorflow as tf
import random
random.seed(3)
tf.random.set_seed(24)
import argparse
from HMOGPLV.utils import plot_gp
from HMOGPLV.utils import NMSE_test_bar, MNLP, MSE_test_bar_normall
import pandas as pd
import matplotlib.pyplot as plt
import gpflow
import HMOGPLV
import HMOGPLV.models as MODEL
from gpflow.config import default_float
import numpy as np
import tensorflow as tf
from HMOGPLV.utils import run_adam_fulldata, save_plot, LMC_data_set, LMCsum_data,new_format_for_X_Y, LMCsum_data_Missing_part_of_one_output_in_Whole, new_format_for_X_Y_Missing_part_of_one_output_in_Whole
from gpflow.utilities import parameter_dict
from gpflow.ci_utils import ci_niter ## for the number of training
from HMOGPLV.models import SHGP_replicated_within_data, SVGP_MOGP, SVGP_MOGP_sum
from gpflow.inducing_variables import InducingPoints, SeparateIndependentInducingVariables
from HMOGPLV.kernels import lmc_kernel
import GPy

from tensorflow import keras
from tensorflow.keras import layers

class Trainer(object):
    def __init__(self, X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test,X_all_outputs_with_replicates,Y_list,training_id, test_id, seed_index, cfg, **kwargs):
        '''
        :param X_list_missing: Training input data set
        :param X_list_missing_test: Test input data set
        :param Y_list_missing: Training output data set
        :param Y_list_missing_test: Test output data set
        :param X_all_outputs_with_replicates:All training input data set, we mainly use this variable in Missing data set for plot
        :param Y_list: All training output data set
        :param training_id: The training index is used in Missing data set for plot
        :param test_id: The test index is used in Missing data set for plot. Sometime it can be used for output index.
        :param seed_index: The index for different repetition in cross-validation
        :param cfg: the configure parameters for all our model
        :param kwargs: potential parameters
        '''
        self.X_list_missing = X_list_missing
        self.X_list_missing_test = X_list_missing_test
        self.Y_list_missing = Y_list_missing
        self.Y_list_missing_test = Y_list_missing_test
        self.X_all_outputs_with_replicates = X_all_outputs_with_replicates
        self.Y_list = Y_list
        self.training_id = training_id
        self.test_id = test_id
        self.seed_index = seed_index
        self.cfg = cfg
        ### Synthetis Data parameter
        self.num_replicates = self.cfg.SYN.NUM_REPLICATES
        self.num_data_each_replicate = self.cfg.SYN.NUM_DATA_IN_REPLICATES
        self.D = self.cfg.SYN.NUM_OUTPUTS
        self.train_percentage = self.cfg.SYN.TRAIN_PERCENTAGE
        ## Model Parameters
        self.gap = self.cfg.MODEL.GAP  ## This is for the inducing variables
        self.Q = self.cfg.MODEL.Q
        ## Optimization Step parameter
        self.Training_step = self.cfg.OPTIMIZATION.TRAINING_NUM_EACH_STEP
        ## Path
        self.my_path = self.cfg.PATH.SAVING_GENERAL
        self.Our_path_loss = self.cfg.PATH.LOSS
        self.Our_path_parameter = self.cfg.PATH.PARAMETERS
        self.Our_path_plot = self.cfg.PATH.PLOT
        self.Our_path_result = cfg.PATH.RESULT

        ## Misc options
        self.Num_repetition = self.cfg.MISC.NUM_REPETITION
        self.Model_name = self.cfg.MISC.MODEL_NAME
        self.Data_name = self.cfg.MISC.DATA_NAME
        self.Mr = self.cfg.MISC.MR
        self.Experiment_type = self.cfg.MISC.EXPERIMENTTYPE
        self.variance_lower = self.cfg.MISC.VARIANCE_BOUND
        self.Num_ker = self.cfg.MISC.NUM_KERNEL

    def set_up_model(self):
        '''
        We set up the model. In each prediciton, we only set up one model
        :return: m_test, data_set, x_input_all_index
        '''
        print('number_of_outputs', self.D)
        if self.Model_name == 'HMOGPLV':
            ## Our model
            m_test, data_set, x_input_all_index = self.set_up_HMOGPLV()
        elif self.Model_name == 'HGPInd':
            ## Single output Gaussian processes with inducing variables
            m_test, data_set, x_input_all_index = self.set_up_HGPInd()
        elif self.Model_name == 'LMC':
            ## LMC model: consider all replicas in the same output as one output
            m_test, data_set, x_input_all_index = self.set_up_LMC()
        elif self.Model_name == 'LMC2':
            ## LMC model: consider each replica as each output
            m_test, data_set, x_input_all_index = self.set_up_LMC2()
        elif self.Model_name == 'LMC3':
            ## LMC model: consider each replica as each output and LMC3 run on one output
            m_test, data_set, x_input_all_index = self.set_up_LMC2()
        elif self.Model_name == 'LMCsum':
            ## LMC model: consider each replica as each output and LMC3 run on one output
            m_test, data_set, x_input_all_index = self.set_up_LMCsum()
        elif self.Model_name == 'HGP':
            ## Single ouptut Gaussian processes (James' paper)
            m_test, data_set, x_input_all_index = self.set_up_HGP()
        elif self.Model_name == 'DNN':
            m_test, data_set, x_input_all_index = self.set_up_DNN()
        elif self.Model_name == 'SGP':
            ## Single ouptut Gaussian processes
            m_test, data_set, x_input_all_index = self.set_up_SGP()
        elif self.Model_name == 'SGP2':
            ## Single ouptut Gaussian processes
            m_test, data_set, x_input_all_index = self.set_up_SGP2()
        elif self.Model_name == 'DHGP':
            ## Deep hierarchial kernel in Gaussian processes (James' paper)
            m_test, data_set, x_input_all_index = self.set_up_DHGP()
        elif self.Model_name == 'LVMOGP':
            ## LVMOGP model: consider all replicas in the same output as one output (Dai's paper)
            m_test, data_set, x_input_all_index = self.set_up_LVMOGP()
        elif self.Model_name == 'LVMOGP2':
            ## LVMOGP model: consider each replica as each output (Dai's paper)
            m_test, data_set, x_input_all_index = self.set_up_LVMOGP2()
        elif self.Model_name == 'LVMOGP3':
            ## LVMOGP model: consider each replica as each output LVMOGP3 run on one output (Dai's paper)
            m_test, data_set, x_input_all_index = self.set_up_LVMOGP2()
        else:
            print('prince correct model')
        return m_test, data_set, x_input_all_index

    def optimization(self, m_test, data_set, x_input_all_index):
        '''
        This function is used for optimizing a model
        :param m_test: the model is used
        :param data_set: the data set for the model
        :param x_input_all_index: input data set with index. We use N = x_input_all_index.shape[0]
        :return:
        '''
        max_run = self.Training_step
        if self.Model_name == 'HMOGPLV' or self.Model_name == 'LMC' or self.Model_name == 'LMCsum' or self.Model_name == 'LMC2' or self.Model_name == 'LMC3':
            ### Adam optimization in tensorflow ###
            N = x_input_all_index.shape[0]
            maxiter = ci_niter(max_run)
            a = time.time()
            logf = run_adam_fulldata(m_test, maxiter, data_set, N, N)
            b = time.time()
            Total_time = b - a
        elif self.Model_name == 'HGPInd':
            ### L-BFGS-B in Scipy for GPflow code ###
            opt = gpflow.optimizers.Scipy()
            logf = []
            def callback(step, variables, values):
                if step % 100 == 0:
                    obj = -m_test.training_loss().numpy()
                    print(step, obj)
                    logf.append(obj)
            a = time.time()
            opt_log = opt.minimize(m_test.training_loss, m_test.trainable_variables, step_callback=callback,
                                   options=dict(maxiter=max_run), compile=True)
            b = time.time()
            Total_time = b - a
        elif self.Model_name == 'HGP' or self.Model_name == 'DHGP':
            ### L-BFGS-B in Scipy for GPy code ###
            a = time.time()
            m_test.optimize(messages=1)
            b = time.time()
            Total_time = b - a
            logf = [0, 0, 0, 0 ,0]
        elif self.Model_name == 'DNN':
            a = time.time()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            m_test.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
            m_test.fit(data_set, x_input_all_index, epochs=max_run, verbose=0, batch_size=data_set.shape[0])
            b = time.time()
            Total_time = b - a
            logf = [0, 0, 0, 0 ,0]
        elif self.Model_name == 'SGP' or self.Model_name == 'SGP2':
            ### L-BFGS-B in Scipy for GPy code ###
            a = time.time()
            m_test.optimize_restarts(5, robust=True)
            b = time.time()
            Total_time = b - a
            logf = [0, 0, 0, 0 ,0]
        elif self.Model_name == 'LVMOGP' or self.Model_name == 'LVMOGP2' or self.Model_name == 'LVMOGP3':
            ### L-BFGS-B in Scipy for GPy code ###
            a = time.time()
            m_test.optimize_auto(max_iters=max_run)
            b = time.time()
            Total_time = b - a
            logf = [0, 0, 0, 0 ,0]
        return Total_time, logf, m_test

    def save_elbo(self, logf):
        '''
        Saving elbo for checking whether converge for those model that is build in tensorflow
        '''
        ## make a floder
        newpath_loss = self.my_path + self.Our_path_loss
        if not os.path.exists(newpath_loss):
            os.makedirs(newpath_loss)
        ## save eblo
        np.savetxt(newpath_loss + '/' + self.Experiment_type + self.Data_name + '-%i Output' % self.D
                   + ' with %i replicates' % self.num_replicates + 'loss' + self.Model_name +
                   '-%ith-Run.txt' % self.seed_index, logf, fmt='%d')
        ## plot elbo
        plt.figure(figsize=(20, 9))
        plt.plot(logf[1:])
        plt.xlabel("Training setp"), plt.ylabel("Elbo")
        plt.title("Elbo")
        plt.savefig(newpath_loss + '/' + self.Experiment_type + self.Data_name + '-%i Output' % self.D
                    + ' with %i replicates' % self.num_replicates + ' ELBO' + self.Model_name
                    + '-%ith-Run.eps' % self.seed_index, format='eps', bbox_inches='tight')

    def save_parameter(self, m_test):
        '''
        save the parameter of model that is built in tensorflow
        '''
        ## make a floder
        newpath_parameter = self.my_path + self.Our_path_parameter
        if not os.path.exists(newpath_parameter):
            os.makedirs(newpath_parameter)
        ## save parameters
        params_dict = parameter_dict(m_test)
        for p in params_dict:
            params_dict[p] = params_dict[p].numpy()
        np.savez(newpath_parameter + f"/{self.Data_name}-{self.Experiment_type}-{self.D}Output-"
                                     f"{self.num_replicates}Replicates{self.num_replicates}-output-"
                                     f"{self.gap}gap{self.train_percentage}-trainingpercentage-"
                                     f"{self.Q}-Q-"
                                     f"{self.Model_name}-%ith-Missing-Run.npz" % self.seed_index, **params_dict)


    def plot_index(self, d):
        '''
        Finding indexs for test and training in each output with experiment type: Train_test_in_each_replica
        '''
        idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
        idx_train_d = [self.X_list_missing[d][:, -1] == i for i in range(self.num_replicates)]
        return idx_test_d, idx_train_d

    def plot_train_test(self, ax, d, idx_train_d, r, idx_test_d, newpath_plot, index):
        '''
        Plot with experiment type: Train_test_in_each_replica
        '''
        if self.Model_name == 'HGP' or self.Model_name == 'HGPInd' or self.Model_name == 'SGP' or self.Model_name == 'LMC3' or self.Model_name == 'LVMOGP3':
            ax.plot(self.X_list_missing[:, :-1][idx_train_d[r]], self.Y_list_missing[idx_train_d[r]], 'ks', mew=5.2,
                    ms=4, label='Train')
            ax.plot(self.X_list_missing_test[:, :-1][idx_test_d[r]], self.Y_list_missing_test[idx_test_d[r]], 'rd',
                    mew=8.2, ms=4, label='Test')
        else:
            ax.plot(self.X_list_missing[d][:, :-1][idx_train_d[r]],
                    self.Y_list_missing[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')
            ax.plot(self.X_list_missing_test[d][:, :-1][idx_test_d[r]],
                    self.Y_list_missing_test[d][idx_test_d[r]], 'rd', mew=8.2, ms=4, label='Test')
        plt.title('%i-th output ' % d + '%ith replicates' % r)
        ax.legend()
        save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, index, d, self.Model_name,
                  self.Q, self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)

    def save_plot_aistats(self, m_test):
        '''
        This function plot the prediction with
        '''
        ## save path
        newpath_plot = self.my_path + self.Our_path_plot
        if self.Data_name == 'GUSTO':
            x_raw_plot = np.arange(0, 48, 0.2)[:, None]
        elif self.Data_name == 'Gene':
            x_raw_plot = np.arange(0, 20, 0.2)[:, None]

        Num_plot = x_raw_plot.shape[0]
        if self.Model_name == 'HMOGPLV':  ## HMOGPLV
            ## create the x for prediction
            XX_plot = []
            for r in range(self.num_replicates):
                XX_plot.append(np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)])
            X_r = np.vstack(XX_plot)
            ## prediction
            mean, variance = m_test.predict_f(X_r)
            mu, var = mean.numpy(), variance.numpy() + m_test.Heter_GaussianNoise.numpy()
            ## plot
            for d in range(self.D):
                idx_test_d, idx_train_d = self.plot_index(d)
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates, r + 1)
                    mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                    var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

        elif self.Model_name == 'LMCsum':
            x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                               self.Y_list_missing,
                                                                                               self.X_list_missing_test,
                                                                                               self.Y_list_missing_test,
                                                                                               self.num_replicates,
                                                                                               self.D)
            for r in range(self.num_replicates):
                x_test_all = x_test_all_pre[r]
                y_test_all = y_test_all_pre[r]
                x_train_all = x_train_all_pre[r]
                y_train_all = y_train_all_pre[r]
                plt.figure(figsize=(20, 9))
                for d in range(self.D):
                    ax = plt.subplot(1, self.D, d+1)
                    X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                    mu, var = m_test.predict_f(X_r)
                    var = var + m_test.likelihood.likelihoods[d].variance
                    plot_gp(x_raw_plot, mu, var)

                    index_replica_test = x_test_all[:, -1][:, None] == d
                    index_replica_train = x_train_all[:, -1][:, None] == d

                    ax.plot(x_train_all[:, :-1][index_replica_train.squeeze()],
                                y_train_all[:, :-1][index_replica_train.squeeze()], 'ks', mew=5.2, ms=4, label='Train')
                    ax.plot(x_test_all[:, :-1][index_replica_test.squeeze()],
                                y_test_all[:, :-1][index_replica_test.squeeze()], 'rd', mew=8.2, ms=4, label='Test')
                    plt.title('%i-th output' % d + ' %i-th replicate' % r)
                    ax.legend()
                    save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, self.D, self.Model_name, self.Q,
                                  self.train_percentage, r, self.gap, self.Num_repetition, self.seed_index)

        # elif self.Model_name == 'LMC':
        #     # We assume all replica in the same output as one output.
        #     idx_train, idx_test, x_train_all, x_test_all, y_train_all, y_test_all = LMC_data_set(self.D, self.Y_list_missing, self.X_list_missing,
        #                                                                                              self.num_replicates, self.Y_list_missing_test,
        #                                                                                              self.X_list_missing_test)
        #     for d in range(self.D):
        #         plt.figure(figsize=(20, 9))
        #         for r in range(self.num_replicates):
        #             ax = plt.subplot(1, self.num_replicates, r + 1)
        #             X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
        #             mu, var = m_test.predict_f(X_r)
        #             var = var + m_test.likelihood.likelihoods[d].variance
        #             plot_gp(x_raw_plot, mu, var)
        #             ax.plot(x_train_all[:, :-1][idx_train[r + d * self.num_replicates]],
        #                         y_train_all[:, :-1][idx_train[r + d * self.num_replicates]], 'ks', mew=5.2, ms=4, label='Train')
        #             ax.plot(x_test_all[:, :-1][idx_test[r + d * self.num_replicates]],
        #                         y_test_all[:, :-1][idx_test[r + d * self.num_replicates]], 'rd', mew=8.2, ms=4, label='Test')
        #             plt.title('%i-th output' % d + ' %i-th replicate' % r)
        #             ax.legend()
        #             save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, d, self.Model_name, self.Q,
        #                           self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)

        elif self.Model_name == 'HGPInd' or self.Model_name == 'HGP' or self.Model_name == 'SGP':
            d = self.test_id
            ## Consider one by one ouptut: we consider all the same replica as the same input
            ## Index for training and test
            idx_train_d = [self.X_list_missing[:, -1] == i for i in range(self.num_replicates)]
            idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
            plt.figure(figsize=(20, 9))
            for r in range(self.num_replicates):
                ax = plt.subplot(1, self.num_replicates + 1, r + 2)
                if self.Model_name == 'HGPInd':
                    X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                    mu, var = m_test.predict_y(X_r)
                    mu, var = mu.numpy(), var.numpy()
                elif self.Model_name == 'HGP':
                    X_r = np.c_[x_raw_plot, (r+1) * np.ones_like(x_raw_plot)]
                    mu, var = m_test.predict(X_r)
                elif self.Model_name == 'SGP':
                    X_r = x_raw_plot
                    mu, var = m_test.predict(X_r)
                plot_gp(x_raw_plot, mu, var)
                self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

        elif self.Model_name == 'LVMOGP':
            # We assume all replica in the same output as one output.
            mu, var = m_test.predict(x_raw_plot)
            for d in range(self.D):
                idx_test_d, idx_train_d = self.plot_index(d)
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates, r + 1)
                    mu_on = mu[:, d][:, None]
                    var_on = var[:, d][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

        elif self.Model_name == 'DHGP':
            for d in range(self.D):
                ## create x for ploting
                XX_plot = []
                for r in range(self.num_replicates):
                    XX_plot.append(np.c_[x_raw_plot, d * np.ones_like(x_raw_plot), r * np.ones_like(x_raw_plot) + d * self.num_replicates + 1])
                X_r = np.vstack(XX_plot)
                mu, var = m_test.predict(X_r)
                idx_test_d, idx_train_d = self.plot_index(d)
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates, r + 1)
                    mu_on = mu[Num_plot * r:Num_plot * (r + 1)]
                    var_on = var[Num_plot * r:Num_plot * (r + 1)]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

    def save_plot_synthetic(self, m_test):
        '''
        This function plot the prediction for synthetic data set. It plot in two main types: Train_test_in_each_replica and others
        '''
        ## save path
        newpath_plot = self.my_path + self.Our_path_plot
        if self.Data_name == 'Synthetic_different_input':
            x_raw_plot = np.arange(0, 10, 0.2)[:, None]
        elif self.Data_name == 'Gene':
            x_raw_plot = np.arange(0, 20, 0.2)[:, None]
        elif self.Data_name == 'MOCAP9':
            x_raw_plot = np.arange(-2, 2, 0.01)[:, None]
        # x_raw_plot = np.arange(0, 10, 0.2)[:, None]
        Num_plot = x_raw_plot.shape[0]
        if self.Experiment_type == 'Train_test_in_each_replica':
            if self.Model_name == 'HMOGPLV': ## HMOGPLV
                ## create the x for prediction
                XX_plot = []
                for r in range(self.num_replicates):
                    XX_plot.append(np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)])
                X_r = np.vstack(XX_plot)
                ## prediction
                mean, variance = m_test.predict_f(X_r)
                mu, var = mean.numpy(), variance.numpy() + m_test.Heter_GaussianNoise.numpy()
                ## plot
                for d in range(self.D):
                    idx_test_d, idx_train_d = self.plot_index(d)
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, self.num_replicates, 1)
                    for r in range(self.num_replicates):
                        ax = plt.subplot(1, self.num_replicates, r + 1)
                        mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                        var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]
                        plot_gp(x_raw_plot, mu_on, var_on)
                        self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

            elif self.Model_name == 'HGPInd' or self.Model_name == 'HGP' or self.Model_name == 'SGP':
                d = self.test_id
                ## Consider one by one ouptut: we consider all the same replica as the same input
                ## Index for training and test
                idx_train_d = [self.X_list_missing[:, -1] == i for i in range(self.num_replicates)]
                idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
                plt.figure(figsize=(20, 9))
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates + 1, r + 2)
                    if self.Model_name == 'HGPInd':
                        X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                        mu, var = m_test.predict_y(X_r)
                        mu, var = mu.numpy(), var.numpy()
                    elif self.Model_name == 'HGP':
                        X_r = np.c_[x_raw_plot, (r+1) * np.ones_like(x_raw_plot)]
                        mu, var = m_test.predict(X_r)
                    elif self.Model_name == 'SGP':
                        X_r = x_raw_plot
                        mu, var = m_test.predict(X_r)
                    plot_gp(x_raw_plot, mu, var)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)
            elif self.Model_name == 'SGP2':
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, 1, 1)
                mu, var = m_test.predict(x_raw_plot)
                plot_gp(x_raw_plot, mu, var)
                ax.plot(self.X_list_missing[:, :-1], self.Y_list_missing, 'ks', mew=5.2, ms=4, label='Train')
                ax.plot(self.X_list_missing_test[:, :-1], self.Y_list_missing_test, 'rd', mew=8.2, ms=4, label='Test')
                plt.title('%i-th output ' % self.test_id + '%ith replicates' % self.seed_index)
                ax.legend()
                save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.test_id, self.test_id, self.Model_name,
                      self.Q, self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)

            elif self.Model_name == 'DHGP':
                for d in range(self.D):
                    ## create x for ploting
                    XX_plot = []
                    for r in range(self.num_replicates):
                        XX_plot.append(np.c_[x_raw_plot, d * np.ones_like(x_raw_plot), r * np.ones_like(x_raw_plot) + d * self.num_replicates + 1])
                    X_r = np.vstack(XX_plot)
                    mu, var = m_test.predict(X_r)
                    idx_test_d, idx_train_d = self.plot_index(d)
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, self.num_replicates, 1)
                    for r in range(self.num_replicates):
                        ax = plt.subplot(1, self.num_replicates, r + 1)
                        mu_on = mu[Num_plot * r:Num_plot * (r + 1)]
                        var_on = var[Num_plot * r:Num_plot * (r + 1)]
                        plot_gp(x_raw_plot, mu_on, var_on)
                        self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

            elif self.Model_name == 'LMC':
                # We assume all replica in the same output as one output.
                idx_train, idx_test, x_train_all, x_test_all, y_train_all, y_test_all = LMC_data_set(self.D, self.Y_list_missing, self.X_list_missing,
                                                                                                     self.num_replicates, self.Y_list_missing_test,
                                                                                                     self.X_list_missing_test)
                for d in range(self.D):
                    plt.figure(figsize=(20, 9))
                    for r in range(self.num_replicates):
                        ax = plt.subplot(1, self.num_replicates, r + 1)
                        X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                        mu, var = m_test.predict_f(X_r)
                        var = var + m_test.likelihood.likelihoods[d].variance
                        plot_gp(x_raw_plot, mu, var)
                        ax.plot(x_train_all[:, :-1][idx_train[r + d * self.num_replicates]],
                                y_train_all[:, :-1][idx_train[r + d * self.num_replicates]], 'ks', mew=5.2, ms=4, label='Train')
                        ax.plot(x_test_all[:, :-1][idx_test[r + d * self.num_replicates]],
                                y_test_all[:, :-1][idx_test[r + d * self.num_replicates]], 'rd', mew=8.2, ms=4, label='Test')
                        plt.title('%i-th output' % d + ' %i-th replicate' % r)
                        ax.legend()
                        save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, d, self.Model_name, self.Q,
                                  self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)
            elif self.Model_name == 'LMCsum':
                x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                               self.Y_list_missing,
                                                                                               self.X_list_missing_test,
                                                                                               self.Y_list_missing_test,
                                                                                               self.num_replicates,
                                                                                               self.D)
                for r in range(self.num_replicates):
                    x_test_all = x_test_all_pre[r]
                    y_test_all = y_test_all_pre[r]
                    x_train_all = x_train_all_pre[r]
                    y_train_all = y_train_all_pre[r]
                    plt.figure(figsize=(20, 9))
                    for d in range(self.D):
                        ax = plt.subplot(1, self.D, d+1)
                        X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                        mu, var = m_test.predict_f(X_r)
                        var = var + m_test.likelihood.likelihoods[d].variance
                        plot_gp(x_raw_plot, mu, var)

                        index_replica_test = x_test_all[:, -1][:, None] == d
                        index_replica_train = x_train_all[:, -1][:, None] == d

                        ax.plot(x_train_all[:, :-1][index_replica_train.squeeze()],
                                y_train_all[:, :-1][index_replica_train.squeeze()], 'ks', mew=5.2, ms=4, label='Train')
                        ax.plot(x_test_all[:, :-1][index_replica_test.squeeze()],
                                y_test_all[:, :-1][index_replica_test.squeeze()], 'rd', mew=8.2, ms=4, label='Test')
                        plt.title('%i-th output' % d + ' %i-th replicate' % r)
                        ax.legend()
                        save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, self.D, self.Model_name, self.Q,
                                  self.train_percentage, r, self.gap, self.Num_repetition, self.seed_index)

            elif self.Model_name == 'LMC2':
                # We assume each replica as each output.
                for d in range(self.D):
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, 1, 1)
                    X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                    idx_test_d, idx_train_d = self.plot_index(d)
                    mu, var = m_test.predict_f(X_r)
                    var = var + m_test.likelihood.likelihoods[d].variance
                    plot_gp(x_raw_plot, mu, var)
                    self.plot_train_test(ax, d, idx_train_d, self.test_id, idx_test_d, newpath_plot, self.test_id)
            elif self.Model_name == 'LMC3':
                d = self.test_id
                # We assume each replica as each output and we only prediction for one output
                idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
                idx_train_d = [self.X_list_missing[:, -1] == i for i in range(self.num_replicates)]
                for r in range(self.num_replicates):
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, 1, 1)
                    X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                    mu, var = m_test.predict_f(X_r)
                    var = var + m_test.likelihood.likelihoods[r].variance
                    plot_gp(x_raw_plot, mu, var)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)
            elif self.Model_name == 'LVMOGP':
                # We assume all replica in the same output as one output.
                mu, var = m_test.predict(x_raw_plot)
                for d in range(self.D):
                    idx_test_d, idx_train_d = self.plot_index(d)
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, self.num_replicates, 1)
                    for r in range(self.num_replicates):
                        ax = plt.subplot(1, self.num_replicates, r + 1)
                        mu_on = mu[:, d][:, None]
                        var_on = var[:, d][:, None]
                        plot_gp(x_raw_plot, mu_on, var_on)
                        self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)
            elif self.Model_name == 'LVMOGP2':
                # We assume each replica as each output.
                mu, var = m_test.predict(x_raw_plot)
                for d in range(self.D):
                    idx_test_d, idx_train_d = self.plot_index(d)
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, 1, 1)
                    mu_on = mu[:, d][:, None]
                    var_on = var[:, d][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, self.test_id, idx_test_d, newpath_plot, self.test_id)
            elif self.Model_name == 'LVMOGP3':
                d = self.test_id
                # We assume each replica as each output and we only prediction for one output
                idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
                idx_train_d = [self.X_list_missing[:, -1] == i for i in range(self.num_replicates)]
                mu, var = m_test.predict(x_raw_plot)
                for r in range(self.num_replicates):
                    plt.figure(figsize=(20, 9))
                    ax = plt.subplot(1, 1, 1)
                    mu_on = mu[:, r][:, None]
                    var_on = var[:, r][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on)
                    self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)
        elif self.Experiment_type == 'Missing_part_of_one_output_in_Whole':
            if self.Model_name == 'HMOGPLV': ## HMOGPLV
                ## create the x for prediction
                XX_plot = []
                for r in range(self.num_replicates):
                    XX_plot.append(np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)])
                X_r = np.vstack(XX_plot)
                ## prediction
                mean, variance = m_test.predict_f(X_r)
                mu, var = mean.numpy(), variance.numpy() + m_test.Heter_GaussianNoise.numpy()
                ## plot
                d = self.D - 1
                idx_test_d = [self.X_list_missing_test[0][:, -1] == i for i in range(self.num_replicates)]
                idx_train_d = [self.X_list_missing[d][:, -1] == i for i in range(self.num_replicates)]
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates, r + 1)
                    mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                    var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]
                    plot_gp(x_raw_plot, mu_on, var_on)

                    ax.plot(self.X_list_missing[d][:, :-1][idx_train_d[r]],
                            self.Y_list_missing[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')
                    ax.plot(self.X_list_missing_test[0][:, :-1][idx_test_d[r]],
                            self.Y_list_missing_test[0][idx_test_d[r]], 'rd', mew=8.2, ms=4, label='Test')
                    plt.title('%i-th output ' % d + '%ith replicates' % r)
                    ax.legend()
                    save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, d, self.Model_name,
                              self.Q, self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition,
                              self.seed_index)

                    # self.plot_train_test(ax, d, idx_train_d, r, idx_test_d, newpath_plot, self.num_replicates)

        else:
            #######################################
            ### Plot for the Missing data type ####
            #######################################
            if self.Model_name == 'SGP':
                d = self.test_id
                idx_train_d = [self.X_all_outputs_with_replicates[d][:, -1] == i for i in range(self.num_replicates)]
                plt.figure(figsize=(20, 9))
                ax = plt.subplot(1, self.num_replicates, 1)
                for r in range(self.num_replicates):
                    ax = plt.subplot(1, self.num_replicates, r + 1)
                    # find the mean and variance
                    mu_on, var_on = m_test.predict(x_raw_plot)

                    # plot for different experiment type
                    plot_gp(x_raw_plot, mu_on, var_on)
                    if r == (d + self.seed_index) % self.num_replicates:
                        ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]], self.Y_list[d][idx_train_d[r]], 'rd', mew=8.2, ms=4, label='Missing')
                    else:
                        ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]], self.Y_list[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')
                    plt.title('%i-th output ' % d + '%ith replicates' % r)
                    ax.legend()
                    save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, d,
                              self.Model_name, self.Q, self.train_percentage, self.num_data_each_replicate,
                              self.gap, self.Num_repetition, self.seed_index)
            else:
                if self.Model_name == 'HMOGPLV':
                    XX_plot = []
                    for r in range(self.num_replicates):
                        XX_plot.append(np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)])
                    X_r = np.vstack(XX_plot)
                    mean, variance = m_test.predict_f(X_r)
                    mu, var = mean.numpy(), variance.numpy() + m_test.Heter_GaussianNoise.numpy()
                elif self.Model_name == 'LVMOGP' or self.Model_name == 'LVMOGP3':
                    mu, var = m_test.predict(x_raw_plot)

                if self.Model_name == 'LMCsum':

                    x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                                   self.Y_list_missing,
                                                                                                   self.X_list_missing_test,
                                                                                                   self.Y_list_missing_test,
                                                                                                   self.num_replicates,
                                                                                                   self.D)
                    for r in range(self.num_replicates):
                        x_test_all = x_test_all_pre[r]
                        y_test_all = y_test_all_pre[r]
                        x_train_all = x_train_all_pre[r]
                        y_train_all = y_train_all_pre[r]
                        plt.figure(figsize=(50, 9))
                        for d in range(self.D):
                            ax = plt.subplot(1, self.D, d + 1)
                            X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                            mu, var = m_test.predict_f(X_r)
                            var = var + m_test.likelihood.likelihoods[d].variance
                            plot_gp(x_raw_plot, mu, var)

                            index_replica_test = x_test_all[:, -1][:, None] == d
                            index_replica_train = x_train_all[:, -1][:, None] == d

                            ax.plot(x_train_all[:, :-1][index_replica_train.squeeze()],
                                    y_train_all[:, :-1][index_replica_train.squeeze()], 'ks', mew=5.2, ms=4,
                                    label='Train')
                            ax.plot(x_test_all[:, :-1][index_replica_test.squeeze()],
                                    y_test_all[:, :-1][index_replica_test.squeeze()], 'rd', mew=8.2, ms=4, label='Test')
                            plt.title('%i-th output' % d + ' %i-th replicate' % r)
                            ax.legend()
                            save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates,
                                      self.D, self.Model_name, self.Q,
                                      self.train_percentage, r, self.gap, self.Num_repetition, self.seed_index)
                else:
                    for d in range(self.D):
                        idx_train_d = [self.X_all_outputs_with_replicates[d][:, -1] == i for i in range(self.num_replicates)]
                        plt.figure(figsize=(20, 9))
                        ax = plt.subplot(1, self.num_replicates, 1)
                        for r in range(self.num_replicates):
                            ax = plt.subplot(1, self.num_replicates, r + 1)

                            # find the mean and variance
                            if self.Model_name == 'HMOGPLV':
                                mu_on = mu[Num_plot * r:Num_plot * (r + 1), d][:, None]
                                var_on = var[Num_plot * r:Num_plot * (r + 1), d][:, None]
                            elif self.Model_name == 'HGPInd':
                                X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                                mu, var = m_test.predict_y(X_r)
                                mu_on, var_on = mu.numpy(), var.numpy()
                            elif self.Model_name == 'LMC':
                                X_r = np.c_[x_raw_plot, d * np.ones_like(x_raw_plot)]
                                mu_on, var_on = m_test.predict_f(X_r)
                                var_on = var_on + m_test.likelihood.likelihoods[d].variance
                            elif self.Model_name == 'LMC3':
                                X_r = np.c_[x_raw_plot, r * np.ones_like(x_raw_plot)]
                                mu_on, var_on = m_test.predict_f(X_r)
                                var_on = var_on + m_test.likelihood.likelihoods[r].variance
                            elif self.Model_name == 'LVMOGP':
                                mu_on = mu[:, d][:, None]
                                var_on = var[:, d][:, None]
                            elif self.Model_name == 'LVMOGP3':
                                mu_on = mu[:, r][:, None]
                                var_on = var[:, r][:, None]
                            elif self.Model_name == 'HGP':
                                X_r = np.c_[x_raw_plot, (r + 1) * np.ones_like(x_raw_plot)]
                                mu_on, var_on = m_test.predict(X_r)
                            # plot for different experiment types
                            plot_gp(x_raw_plot, mu_on, var_on)
                            if d == self.seed_index and r == (
                                    d + self.seed_index) % self.num_replicates and self.Experiment_type == 'Missing_One_replica_in_Whole':
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]], self.Y_list[d][idx_train_d[r]], 'rd',
                                        mew=8.2, ms=4, label='Missing')
                            elif d == self.seed_index and r == (
                                    d + self.seed_index) % self.num_replicates and self.Experiment_type == 'Missing_part_of_one_replica_in_Whole':
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]][self.training_id],
                                        self.Y_list[d][idx_train_d[r]][self.training_id], 'ks', mew=5.2, ms=4, label='Train')
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]][self.test_id],
                                        self.Y_list[d][idx_train_d[r]][self.test_id], 'rd', mew=8.2, ms=4, label='Missing')
                            if self.Experiment_type == 'Missing_One_replica_in_each_ouput':
                                if r == (d + self.seed_index) % self.num_replicates:
                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]], self.Y_list[d][idx_train_d[r]], 'rd',
                                            mew=8.2, ms=4, label='Missing')
                                else:
                                    ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]], self.Y_list[d][idx_train_d[r]], 'ks', mew=5.2, ms=4, label='Train')
                            elif r == (d + self.seed_index) % self.num_replicates and self.Experiment_type == 'Missing_part_of_one_replica_in_each_ouput':
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]][self.training_id[d]],
                                        self.Y_list[d][idx_train_d[r]][self.training_id[d]], 'ks', mew=5.2, ms=4, label='Train')
                                ax.plot(self.X_all_outputs_with_replicates[d][:, :-1][idx_train_d[r]][self.test_id[d]],
                                        self.Y_list[d][idx_train_d[r]][self.test_id[d]], 'rd', mew=8.2, ms=4, label='Missing')
                            plt.title('%i-th output ' % d + '%ith replicates' % r)
                            ax.legend()
                            save_plot(newpath_plot, self.Experiment_type, self.Data_name, self.D, self.num_replicates, d, self.Model_name, self.Q,
                                      self.train_percentage, self.num_data_each_replicate, self.gap, self.Num_repetition, self.seed_index)



    def prediction_for_model(self, m_test):
        '''
        This function is used for prediction for test data set and calculation for evaluation metric
        :param m_test:
        :return:
        '''
        # evaluation metric
        NMSE_test_bar_missing_replicates = []
        MNLP_missing_replicates = []
        Global_mean_pre = []
        Global_variance_pre = []
        Global_test_pre = []
        if self.Experiment_type == 'Missing_One_replica_in_Whole' or self.Experiment_type == 'Missing_part_of_one_replica_in_Whole':
            for d in range(self.D):
                if d == self.seed_index:
                    if self.Model_name == 'HMOGPLV':
                        mean_missing, variance_missing = m_test.predict_f(self.X_list_missing_test[0])
                        mean_missing, variance_missing = mean_missing.numpy(), variance_missing.numpy() + m_test.Heter_GaussianNoise.numpy()
                        mean_one_missing = mean_missing[:, d][:, None]
                        variance_one_missing = variance_missing[:, d][:, None]
                    elif self.Model_name == 'HGPInd':
                        mean_missing, variance_missing = m_test.predict_y(self.X_list_missing_test[0])
                        mean_one_missing, variance_one_missing = mean_missing.numpy(), variance_missing.numpy()
                    elif self.Model_name == 'LMC':
                        X_pre = np.c_[self.X_list_missing_test[0][:, :-1], np.ones_like(self.X_list_missing_test[0][:, :-1]) * d]
                        mean_missing, variance_missing = m_test.predict_f(X_pre)
                        mean_one_missing = mean_missing.numpy()
                        variance_one_missing = variance_missing.numpy() + m_test.likelihood.likelihoods[d].variance.numpy()
                    Y_one_missing = self.Y_list_missing_test[0]
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
        elif self.Experiment_type == 'Missing_One_replica_in_each_ouput' or self.Experiment_type == 'Missing_part_of_one_replica_in_each_ouput':
            if self.Model_name == 'SGP':
                x_test = self.X_list_missing_test[:, :-1]
                mean_one_missing, variance_one_missing = m_test.predict(x_test)
                Y_one_missing = self.Y_list_missing_test
                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

            elif self.Model_name == 'DNN':
                x_test = self.X_list_missing_test[:, :-1]
                mean_one_missing = m_test.predict(x_test)
                variance_one_missing = abs(mean_one_missing)
                Y_one_missing = self.Y_list_missing_test
                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

            elif self.Model_name == 'LMC3' or self.Model_name == 'LVMOGP3':
                Indexoutput_test = []
                x_all_test = []
                y_all_test = []
                for r in range(self.num_replicates):
                    x_output_d_r = []
                    y_output_d_r = []
                    Indexoutput_d_r = []
                    for d in range(self.D):
                        index_r = self.X_list_missing_test[d][:, -1] == r
                        x_output_d_r.append(self.X_list_missing_test[d][index_r][:, :-1])
                        y_output_d_r.append(self.Y_list_missing_test[d][index_r])
                        Indexoutput_d_r.append(np.ones_like(self.Y_list_missing_test[d][index_r]) * r)
                    x_all_test.append(np.vstack(x_output_d_r))
                    y_all_test.append(np.vstack(y_output_d_r))
                    Indexoutput_test.append(np.vstack(Indexoutput_d_r))
                Indexoutput_test = np.vstack(Indexoutput_test)
                x_all_test = np.vstack(x_all_test)
                x_input_all_index = np.hstack((x_all_test, Indexoutput_test))
                y_all_test = np.vstack(y_all_test)
                idx_test = [x_input_all_index[:, -1] == i for i in range(self.num_replicates)]
                R = []
                for i in range(self.num_replicates):
                    if True in idx_test[i]:
                        R.append(i)

                ## Prediction
                if self.Model_name == 'LMC3':
                    mean_test, var_test = m_test.predict_f(x_input_all_index)
                    mean_test, var_test = mean_test.numpy(), var_test.numpy()
                elif self.Model_name == 'LVMOGP3':
                    mean_test, var_test = m_test.predict(x_all_test)


                for r in R:
                    if self.Model_name == 'LMC3':
                        mean_one_missing = mean_test[idx_test[r]]
                        variance_one_missing = var_test[idx_test[r]] + m_test.likelihood.likelihoods[r].variance.numpy()
                    elif self.Model_name == 'LVMOGP3':
                        mean_one_missing = mean_test[idx_test[r], r][:, None]
                        variance_one_missing = var_test[idx_test[r], r][:, None]
                    Y_one_missing = y_all_test[idx_test[r]]
                    # Evalutaion metric
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

            else:
                for d in range(self.D):
                    if self.Model_name == 'HMOGPLV':
                        mean_missing, variance_missing = m_test.predict_f(self.X_list_missing_test[d])
                        mean_missing, variance_missing = mean_missing.numpy(), variance_missing.numpy() + m_test.Heter_GaussianNoise.numpy()
                        mean_one_missing = mean_missing[:, d][:, None]
                        variance_one_missing = variance_missing[:, d][:, None]
                    elif self.Model_name == 'HGPInd':
                        mean_missing, variance_missing = m_test.predict_y(self.X_list_missing_test[d])
                        mean_one_missing, variance_one_missing = mean_missing.numpy(), variance_missing.numpy()
                    elif self.Model_name == 'LMC':
                        X_pre = np.c_[self.X_list_missing_test[d][:, :-1], np.ones_like(self.X_list_missing_test[d][:, :-1]) * d]

                        mean_missing, variance_missing = m_test.predict_f(X_pre)
                        mean_one_missing = mean_missing.numpy()
                        variance_one_missing = variance_missing.numpy() + m_test.likelihood.likelihoods[d].variance.numpy()
                    elif self.Model_name == 'LMCsum':
                        X_pre = np.c_[self.X_list_missing_test[d][:, :-1], np.ones_like(self.X_list_missing_test[d][:, :-1]) * d]
                        mean_missing, variance_missing = m_test.predict_f(X_pre)
                        mean_one_missing = mean_missing.numpy()
                        variance_one_missing = variance_missing.numpy() + m_test.likelihood.likelihoods[d].variance.numpy()
                    elif self.Model_name == 'LVMOGP':
                        mean_test, variance_test = m_test.predict(self.X_list_missing_test[d][:, :-1])
                        mean_one_missing = mean_test[:, d][:, None]
                        variance_one_missing = variance_test[:, d][:, None]
                    elif self.Model_name == 'HGP':
                        x_test = np.hstack((self.X_list_missing_test[d][:, :-1], self.X_list_missing_test[d][:, -1][:, None] + 1))
                        mean_one_missing, variance_one_missing = m_test.predict(x_test)
                    elif self.Model_name == 'DHGP':

                        x_test = np.hstack((self.X_list_missing_test[d][:, :-1], self.X_list_missing_test[d][:, -1][:, None] * 0 + d,
                                            self.X_list_missing_test[d][:, -1][:, None] * 0))


                        mean_one_missing, variance_one_missing = m_test.predict(x_test)





                    Y_one_missing = self.Y_list_missing_test[d]
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

        elif self.Experiment_type == 'Missing_part_of_one_output_in_Whole':
            if self.Model_name == 'HMOGPLV':
                d = self.D-1
                idx_test_d = [self.X_list_missing_test[0][:, -1] == i for i in range(self.num_replicates)]
                mean_test, variance_test = m_test.predict_f(self.X_list_missing_test[0])
                mean_test, variance_test = mean_test.numpy(), variance_test.numpy() + m_test.Heter_GaussianNoise.numpy()
                for r_test in range(self.num_replicates):
                    mean_one_missing = mean_test[idx_test_d[r_test], d][:, None]
                    variance_one_missing = variance_test[idx_test_d[r_test], d][:, None]
                    if len(variance_one_missing) == 0:
                        continue
                    Y_one_missing = self.Y_list_missing_test[0][idx_test_d[r_test]]
                    # Evalutaion metric
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

            elif self.Model_name == 'LMCsum':

                d = self.D -1
                x_test_missing = self.X_list_missing_test[0]
                x_test_all = np.c_[x_test_missing[:,0][:,None], np.ones_like(x_test_missing[:, -1][:, None]) * d]
                y_test_all = self.Y_list_missing_test[0]
                ## Prediction
                mean_test, var_test = m_test.predict_f(x_test_all)
                mean_test, var_test = mean_test.numpy(), var_test.numpy()

                mean_one_missing = mean_test
                variance_one_missing = var_test + m_test.likelihood.likelihoods[d].variance.numpy()
                Y_one_missing = y_test_all

                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

        elif self.Experiment_type == 'Train_test_in_each_replica':
            if self.Model_name == 'HMOGPLV':
                for d in range(self.D):
                    idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
                    mean_test, variance_test = m_test.predict_f(self.X_list_missing_test[d])
                    mean_test, variance_test = mean_test.numpy(), variance_test.numpy() + m_test.Heter_GaussianNoise.numpy()
                    NMSE_GUSTO_pred = []
                    NMSE_GUSTO_real = []

                    for r_test in range(self.num_replicates):
                        mean_one_missing = mean_test[idx_test_d[r_test], d][:, None]
                        variance_one_missing = variance_test[idx_test_d[r_test], d][:, None]
                        if len(variance_one_missing) == 0:
                            continue
                        Y_one_missing = self.Y_list_missing_test[d][idx_test_d[r_test]]
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        if self.Data_name == 'GUSTO':
                            NMSE_GUSTO_real.append(Y_one_missing)
                            NMSE_GUSTO_pred.append(mean_one_missing)
                            # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        else:
                            NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                        NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))


            elif self.Model_name == 'LVMOGP':
                for d in range(self.D):
                    NMSE_GUSTO_pred = []
                    NMSE_GUSTO_real = []
                    idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
                    mean_test, variance_test = m_test.predict(self.X_list_missing_test[d][:, :-1])
                    for r_test in range(self.num_replicates):
                        mean_one_missing = mean_test[idx_test_d[r_test], d][:, None]
                        variance_one_missing = variance_test[idx_test_d[r_test], d][:, None]
                        if len(variance_one_missing) == 0:
                            continue
                        Y_one_missing = self.Y_list_missing_test[d][idx_test_d[r_test]]
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        if self.Data_name == 'GUSTO':
                            NMSE_GUSTO_real.append(Y_one_missing)
                            NMSE_GUSTO_pred.append(mean_one_missing)
                            # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        else:
                            NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                        NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))
            elif self.Model_name == 'LVMOGP2':
                for d in range(self.D):
                    idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
                    mean_test, variance_test = m_test.predict(self.X_list_missing_test[d][:, :-1])
                    mean_one_missing = mean_test[idx_test_d[self.test_id], d][:, None]
                    variance_one_missing = variance_test[idx_test_d[self.test_id], d][:, None]
                    Y_one_missing = self.Y_list_missing_test[d][idx_test_d[self.test_id]]
                    # Evalutaion metric
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
            elif self.Model_name == 'DHGP':
                for d in range(self.D):
                    NMSE_GUSTO_pred = []
                    NMSE_GUSTO_real = []
                    idx_test_d = [self.X_list_missing_test[d][:, -1] == i for i in range(self.num_replicates)]
                    Indexoutput_d = np.ones_like(self.Y_list_missing_test[d]) * d
                    Index_replica_d = self.X_list_missing_test[d][:, -1][:, None] + 1 + d * self.num_replicates
                    x_all_d = np.hstack((np.vstack(self.X_list_missing_test[d])[:, :-1], Indexoutput_d))
                    X_d = np.hstack((x_all_d, Index_replica_d))
                    mean_test, variance_test = m_test.predict(X_d)
                    for r_test in range(self.num_replicates):
                        mean_one_missing = mean_test[idx_test_d[r_test]]
                        variance_one_missing = variance_test[idx_test_d[r_test]]
                        if len(variance_one_missing) == 0:
                            continue
                        Y_one_missing = self.Y_list_missing_test[d][idx_test_d[r_test]]
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        if self.Data_name == 'GUSTO':
                            NMSE_GUSTO_real.append(Y_one_missing)
                            NMSE_GUSTO_pred.append(mean_one_missing)
                            # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        else:
                            NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                        NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))
            elif self.Model_name == 'LMCsum':
                idx_train, idx_test, x_train_all, x_test_all, y_train_all, y_test_all = LMC_data_set(self.D, self.Y_list_missing, self.X_list_missing,
                                                                                                     self.num_replicates, self.Y_list_missing_test,
                                                                                                     self.X_list_missing_test)
                ## Prediction
                mean_test, var_test = m_test.predict_f(x_test_all)
                mean_test, var_test = mean_test.numpy(), var_test.numpy()
                for d in range(self.D):
                    NMSE_GUSTO_pred = []
                    NMSE_GUSTO_real = []
                    for r in range(self.num_replicates):
                        mean_one_missing = mean_test[idx_test[r + d * self.num_replicates]]
                        variance_one_missing = var_test[idx_test[r + d * self.num_replicates]] + m_test.likelihood.likelihoods[d].variance.numpy()
                        Y_one_missing = y_test_all[:, :-1][idx_test[r + d * self.num_replicates]]
                        if len(variance_one_missing) == 0:
                            continue
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        if self.Data_name == 'GUSTO':
                            NMSE_GUSTO_real.append(Y_one_missing)
                            NMSE_GUSTO_pred.append(mean_one_missing)
                            # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        else:
                            NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                        NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))

            elif self.Model_name == 'LMC2':
                Indexoutput_test = []
                x_test_all = []
                y_test_all = []
                for d in range(self.D):
                    index_r = self.X_list_missing_test[d][:, -1] == self.test_id
                    x_test_all.append(self.X_list_missing_test[d][index_r][:, :-1])
                    y_test_all.append(self.Y_list_missing_test[d][index_r])
                    Indexoutput_test.append(np.ones_like(self.Y_list_missing_test[d][index_r]) * d)
                Indexoutput_test = np.vstack(Indexoutput_test)
                x_test_all = np.vstack(x_test_all)
                y_test_all = np.vstack(y_test_all)
                x_test_all_index = np.hstack((x_test_all, Indexoutput_test))
                ## Prediction
                mean_test, var_test = m_test.predict_f(x_test_all_index)
                mean_test, var_test = mean_test.numpy(), var_test.numpy()
                idx_test = [x_test_all_index[:, -1] == i for i in range(self.D)]
                for d in range(self.D):
                    mean_one_missing = mean_test[idx_test[d]]
                    variance_one_missing = var_test[idx_test[d]] + m_test.likelihood.likelihoods[d].variance.numpy()
                    Y_one_missing = y_test_all[idx_test[d]]
                    # Evalutaion metric
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
            elif self.Model_name == 'LMCsum':
                x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                               self.Y_list_missing,
                                                                                               self.X_list_missing_test,
                                                                                               self.Y_list_missing_test,
                                                                                               self.num_replicates,
                                                                                               self.D)
                ## Prediction
                for r in range(self.num_replicates):
                    x_test_all = x_test_all_pre[r]
                    y_test_all = y_test_all_pre[r]

                    mean_test, var_test = m_test.predict_f(x_test_all)
                    mean_test, var_test = mean_test.numpy(), var_test.numpy()
                    for d in range(self.D):
                        index_replica = x_test_all[:, -1][:, None] == d
                        mean_one_missing = mean_test[index_replica.squeeze()]
                        variance_one_missing = var_test[index_replica.squeeze()] + m_test.likelihood.likelihoods[d].variance.numpy()
                        Y_one_missing = y_test_all[:, :-1][index_replica.squeeze()]
                        if len(variance_one_missing) == 0:
                            continue
                        # Evalutaion metric
                        Global_mean_pre.append(mean_one_missing)
                        Global_variance_pre.append(variance_one_missing)
                        Global_test_pre.append(Y_one_missing)
                        # if self.Data_name == 'GUSTO':
                        #     NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                        # else:
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                        MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))


            elif self.Model_name == 'HGPInd' or self.Model_name == 'HGP' or self.Model_name == 'SGP' or self.Model_name == 'DNN' or self.Model_name == 'LMC3' or self.Model_name == 'LVMOGP3':
                ## Finding the index for each replicate
                idx_test_d = [self.X_list_missing_test[:, -1] == i for i in range(self.num_replicates)]
                ## Prediction

                if self.Model_name == 'HGPInd':
                    mean_test, variance_test = m_test.predict_y(self.X_list_missing_test)
                    mean_test, variance_test = mean_test.numpy(), variance_test.numpy()
                elif self.Model_name == 'HGP':
                    x_test = np.hstack((self.X_list_missing_test[:, :-1], self.X_list_missing_test[:, -1][:, None] + 1))
                    mean_test, variance_test = m_test.predict(x_test)
                elif self.Model_name == 'SGP':
                    x_test = np.vstack(self.X_list_missing_test[:, :-1])
                    mean_test, variance_test = m_test.predict(x_test)
                elif self.Model_name == 'DNN':
                    x_test = np.vstack(self.X_list_missing_test[:, :-1])
                    mean_test = m_test.predict(x_test)
                    variance_test = abs(mean_test)
                elif self.Model_name == 'LMC3':
                    mean_test, variance_test = m_test.predict_f(self.X_list_missing_test)
                    mean_test, variance_test = mean_test.numpy(), variance_test.numpy()
                elif self.Model_name == 'LVMOGP3':
                    x_test = np.vstack(self.X_list_missing_test[:, :-1])
                    mean_test, variance_test = m_test.predict(x_test)
                NMSE_GUSTO_pred = []
                NMSE_GUSTO_real = []
                for r_test in range(self.num_replicates):
                    ## This the mean prediction for the output with the replicate
                    if self.Model_name == 'LMC3':
                        mean_one_missing = mean_test[idx_test_d[r_test]]
                        variance_one_missing = variance_test[idx_test_d[r_test]] + m_test.likelihood.likelihoods[r_test].variance.numpy()
                    elif self.Model_name == 'LVMOGP3':
                        mean_one_missing = mean_test[idx_test_d[r_test], r_test][:, None]
                        variance_one_missing = variance_test[idx_test_d[r_test], r_test][:, None]
                    else:
                        mean_one_missing = mean_test[idx_test_d[r_test]]
                        variance_one_missing = variance_test[idx_test_d[r_test]]
                    Y_one_missing = self.Y_list_missing_test[idx_test_d[r_test]]
                    # Evalutaion metric
                    if len(variance_one_missing) == 0:
                        continue
                    Global_mean_pre.append(mean_one_missing)
                    Global_variance_pre.append(variance_one_missing)
                    Global_test_pre.append(Y_one_missing)
                    if self.Data_name == 'GUSTO':
                        NMSE_GUSTO_real.append(Y_one_missing)
                        NMSE_GUSTO_pred.append(mean_one_missing)
                        # NMSE_test_bar_missing_replicates.append(MSE_test_bar_normall(Y_one_missing, mean_one_missing))
                    else:
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    # NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                    MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))
                if self.Data_name == 'GUSTO':
                    NMSE_GUSTO_real_d_output = np.vstack(NMSE_GUSTO_real)
                    NMSE_GUSTO_pred_d_output = np.vstack(NMSE_GUSTO_pred)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar(NMSE_GUSTO_real_d_output, NMSE_GUSTO_pred_d_output))
            elif self.Model_name == 'SGP2':
                x_test = self.X_list_missing_test[:, :-1]
                mean_one_missing, variance_one_missing = m_test.predict(x_test)
                Y_one_missing = self.Y_list_missing_test
                Global_mean_pre.append(mean_one_missing)
                Global_variance_pre.append(variance_one_missing)
                Global_test_pre.append(Y_one_missing)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar(Y_one_missing, mean_one_missing))
                MNLP_missing_replicates.append(MNLP(Y_one_missing, mean_one_missing, variance_one_missing))

        Global_mean = np.vstack(Global_mean_pre)
        Global_variance = np.vstack(Global_variance_pre)
        Global_test = np.vstack(Global_test_pre)
        Glo_NMSE_test = NMSE_test_bar(Global_test, Global_mean)
        Glo_MNLP = MNLP(Global_test, Global_mean, Global_variance)
        return Glo_NMSE_test, Glo_MNLP, NMSE_test_bar_missing_replicates, MNLP_missing_replicates

    def save_result(self, Glo_NMSE_test_all, Glo_MNLP_all, NMSE_test_bar_missing_replicates_all, MNLP_missing_replicates_all,Time_all):
        '''
        In this function, we want to save all the evaluation metric into a folder
        '''
        Glo_NMSE_test_mean = np.mean(Glo_NMSE_test_all)
        Glo_NMSE_test_std = np.std(Glo_NMSE_test_all)
        Glo_MNLP_mean = np.mean(Glo_MNLP_all)
        Glo_MNLP_std = np.std(Glo_MNLP_all)
        Mean_missing_NMSE_test_bar = np.mean(NMSE_test_bar_missing_replicates_all)
        Std_missing_NMSE_test_bar = np.std(NMSE_test_bar_missing_replicates_all)
        Mean_missing_MNLP = np.mean(MNLP_missing_replicates_all)
        Std_missing_MNLP = np.std(MNLP_missing_replicates_all)

        Result_performance_measure = pd.DataFrame({'Model': [self.Model_name],
                                                   'Total time': [Time_all],
                                                   'Mean of NMSE_test_bar for missing replicates': [
                                                       Mean_missing_NMSE_test_bar],
                                                   'Std of NMSE_test_bar for missing replicates': [
                                                       Std_missing_NMSE_test_bar],
                                                   'Mean of MNLP for missing replicates': [Mean_missing_MNLP],
                                                   'Std of MNLP for missing replicates': [Std_missing_MNLP],
                                                   'Mean of Global NMSE_test for missing replicates': [
                                                       Glo_NMSE_test_mean],
                                                   'Std of Global NMSE_test for missing replicates': [
                                                       Glo_NMSE_test_std],
                                                   'Mean of Global MNLP for missing replicates': [Glo_MNLP_mean],
                                                   'Std of Global MNLP for missing replicates': [Glo_MNLP_std], })

        ## Save our result
        newpath_result = self.my_path + self.Our_path_result
        ## We make a floder
        if not os.path.exists(newpath_result):
            os.makedirs(newpath_result)
        Result_name = self.Experiment_type + self.Model_name + self.Data_name + '-%i-outputs' % self.D + '%i-replicates' % self.num_replicates + '-%i-Q' % self.Q + '-%f-train_percentage' % self.train_percentage + \
                      '-%i-num_data_each_replicate' % self.num_data_each_replicate + '-%i-gap' % self.gap + '-%i-Num_repetition' % self.Num_repetition + '-%i-Num_Inducing_outputs' % self.Mr + '.csv'
        Result_performance_measure.to_csv(newpath_result + Result_name)

    def set_up_HMOGPLV(self):
        '''
        Here we set up HMOGPLV model.
        '''
        ## data set
        Indexoutput = []
        for d in range(self.D):
            Indexoutput.append(np.ones_like(self.Y_list_missing[d]) * d)
        Indexoutput = np.vstack(Indexoutput)
        x_all = np.vstack(self.X_list_missing)
        x_input_all_index = tf.cast(np.hstack((x_all, Indexoutput)), dtype=tf.float64)
        Y_all = np.vstack(self.Y_list_missing)
        indexD = x_input_all_index[..., -1].numpy()
        data_set = (x_input_all_index, Y_all)
        ## model
        lengthscales2 = tf.convert_to_tensor([1.0] * self.Q, dtype=default_float())
        kernel_row = gpflow.kernels.RBF(lengthscales=lengthscales2)
        total_replicated = self.num_replicates
        kern_upper_R = gpflow.kernels.Matern32()
        kern_lower_R = gpflow.kernels.Matern32()
        k_hierarchy_outputs = HMOGPLV.kernels.Hierarchial_kernel_replicated(kernel_g=kern_upper_R,
                                                                            kernel_f=[kern_lower_R],
                                                                            total_replicated=total_replicated)
        # if self.Data_name == 'GUSTO':
            # Z = x_all[:3*self.num_replicates]
            # Z = x_all[::self.gap].copy()
        # else:
        Z = x_all[::self.gap].copy()


        Mc = Z.shape[0]
        Xr_dim = self.Q
        Mr = self.Mr
        Heter_GaussianNoise = np.full(self.D, 1)
        m_test = MODEL.HMOGP_prior_outputs_kronecker_product_Missing_speed_up(kernel=k_hierarchy_outputs,
                                                                              kernel_row=kernel_row,
                                                                              Heter_GaussianNoise=Heter_GaussianNoise,
                                                                              Xr_dim=Xr_dim,
                                                                              Z=Z,
                                                                              num_inducing=(Mc, Mr), indexD=indexD,
                                                                              Initial_parameter='GP',
                                                                              variance_lowerbound=self.variance_lower,
                                                                              x_all=x_all, y_all=Y_all)
        return m_test, data_set, x_input_all_index

    def set_up_HGPInd(self):
        '''
        Here we set up the HGPInd model.
        '''
        ## data set
        x_all = np.vstack(self.X_list_missing)
        Y_all = np.vstack(self.Y_list_missing)
        ## model
        Z = x_all[::self.gap].copy()
        Total_num_replicates = self.num_replicates
        kern_upper_HGP_Inducing = gpflow.kernels.Matern32()
        kern_lower_HGP_Inducing = gpflow.kernels.Matern32()
        k_hierarchy = HMOGPLV.kernels.Hierarchial_kernel_replicated(kernel_g=kern_upper_HGP_Inducing,
                                                                    kernel_f=[kern_lower_HGP_Inducing],
                                                                    total_replicated=Total_num_replicates)
        m_test = SHGP_replicated_within_data(kernel=k_hierarchy, inducing_variable=Z, data=(x_all, Y_all), mean_function=None)
        return m_test, 0, 0

    def set_up_HGP(self):
        '''
        Here we set up the HGP model.
        '''

        ## data set
        if self.Experiment_type == 'Train_test_in_each_replica':
            x_all = np.hstack((self.X_list_missing[:, :-1], self.X_list_missing[:, -1][:, None]+1))
        elif self.Experiment_type == 'Missing_One_replica_in_each_ouput':
            xx = []
            for d in range(self.D):
                xx.append(np.hstack((self.X_list_missing[d][:, :-1], self.X_list_missing[d][:, -1][:, None]+1)))
            x_all = np.vstack(xx)
        Y_all = np.vstack(self.Y_list_missing)

        ## model
        kern_upper = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='upper')
        kern_lower = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='lower')
        k_hierarchy = GPy.kern.Hierarchical(kernels=[kern_upper, kern_lower])
        m_test = GPy.models.GPRegression(X=x_all, Y=Y_all, kernel=k_hierarchy)
        return m_test, 0, 0

    def set_up_DNN(self):
        '''
        Here we set up the HGP model.
        '''

        ## data set
        x_all = self.X_list_missing[:, :-1]

        Y_all = np.vstack(self.Y_list_missing).squeeze()

        ## model
        m_test = keras.Sequential([
            layers.Dense(200, activation='relu', input_shape=[x_all.shape[1]]),
            layers.Dense(200, activation='relu'),
            layers.Dense(1)
        ])

        return m_test, x_all , Y_all

    def set_up_SGP(self):
        '''
        Here we set up the SGP model where we consider all replicas in the same output as one output.
        '''
        ## data set
        x_all = np.vstack(self.X_list_missing[:, :-1])
        Y_all = np.vstack(self.Y_list_missing)
        ## model
        k = GPy.kern.RBF(input_dim=1, active_dims=[0], name="rbf")
        m_test = GPy.models.GPRegression(X=x_all, Y=Y_all, kernel=k)
        return m_test, 0, 0

    def set_up_SGP2(self):
        '''
        Here we set up the SGP model where we consider all replicas in the same output as one output.
        '''
        ## data set
        x_all = self.X_list_missing[:, :-1]
        Y_all = self.Y_list_missing
        ## model
        k = GPy.kern.RBF(input_dim=1, active_dims=[0], name="rbf")
        m_test = GPy.models.GPRegression(X=x_all, Y=Y_all, kernel=k)
        return m_test, 0, 0

    def set_up_DHGP(self):
        '''
        Here we set up the DHGP model.
        '''
        # data set
        Indexoutput = []
        Index_replica = []
        for d in range(self.D):
            Indexoutput.append(np.ones_like(self.Y_list_missing[d]) * d)
            Index_replica.append(self.X_list_missing[d][:, -1][:, None] + 1 + d*self.num_replicates)
        Indexoutput = np.vstack(Indexoutput)
        if self.Experiment_type == 'Missing_One_replica_in_each_ouput':
            Index_replica = np.vstack(Index_replica)
            total_index_replica = self.D * self.num_replicates
            j = 1
            for i in range(total_index_replica+1):
                if sum(Index_replica == i):
                    Index_replica[Index_replica == i] = j
                    j += 1
        else:
            Index_replica = np.vstack(Index_replica)


        x_all = np.hstack((np.vstack(self.X_list_missing)[:, :-1], Indexoutput))
        X = np.hstack((x_all, Index_replica))
        Y = np.vstack(self.Y_list_missing)
        # model
        k_cluster = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='cluster')
        k_gene = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='gene')
        k_replicate = GPy.kern.Matern32(input_dim=1, active_dims=[0], name='replicate')
        k_hierarchy = GPy.kern.Hierarchical([k_cluster, k_gene, k_replicate])
        m_test = GPy.models.GPRegression(X, Y, kernel=k_hierarchy)
        return m_test, 0, 0

    def set_up_LMC(self):
        '''
        Here we set up the LMC model where we consider all replicas in the same output as one output.
        '''

        # data set
        if self.Experiment_type == 'Train_test_in_each_replica':
            _, _, x_input_all_index, _, Y_all, _ = LMC_data_set(self.D, self.Y_list_missing, self.X_list_missing, self.num_replicates,
                                                                self.Y_list_missing_test, self.X_list_missing_test)
        else:
            Indexoutput = []
            for d in range(self.D):
                Indexoutput.append(np.ones_like(self.Y_list_missing[d]) * d)
            Indexoutput = np.vstack(Indexoutput)

            x_all = np.vstack(self.X_list_missing)[:, :-1]
            x_input_all_index = np.hstack((x_all, Indexoutput))
            Y_all = np.hstack((np.vstack(self.Y_list_missing), Indexoutput))
        data_set = (x_input_all_index, Y_all)

        # model
        Z = x_input_all_index[::self.gap, 0][:, None].copy()
        ks = [gpflow.kernels.RBF() for _ in range(self.Num_ker)]
        L = len(ks)
        Zs = [Z.copy() for _ in range(L)]
        iv_list = [InducingPoints(Z) for Z in Zs]
        iv = SeparateIndependentInducingVariables(iv_list)
        N = x_input_all_index.shape[0]
        likelihood_gp = [gpflow.likelihoods.Gaussian() for _ in range(self.D)]
        kern = lmc_kernel(self.D, ks)
        lik = gpflow.likelihoods.SwitchedLikelihood(likelihood_gp)
        m_test = SVGP_MOGP(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L, num_data=N)

        return m_test, data_set, x_input_all_index

    def set_up_LMC2(self):
        '''
        Here we set up the LMC2 model where we consider each replica as each output.
        '''
        # data set
        Indexoutput = []
        x_all = []
        y_all = []
        if self.Model_name == 'LMC2':
            for d in range(self.D):
                index_r = self.X_list_missing[d][:, -1] == self.test_id
                x_all.append(self.X_list_missing[d][index_r][:, :-1])
                y_all.append(self.Y_list_missing[d][index_r])
                Indexoutput.append(np.ones_like(self.Y_list_missing[d][index_r]) * d)
        elif self.Model_name == 'LMC3' and self.Experiment_type == 'Train_test_in_each_replica':
            for r in range(self.num_replicates):
                index_r = self.X_list_missing[:, -1] == r
                x_all.append(self.X_list_missing[index_r][:, :-1])
                y_all.append(self.Y_list_missing[index_r])
                Indexoutput.append(np.ones_like(self.Y_list_missing[index_r]) * r)
        elif self.Model_name == 'LMC3' and self.Experiment_type == 'Missing_One_replica_in_each_ouput':
            for r in range(self.num_replicates):
                x_output_d_r = []
                y_output_d_r = []
                Indexoutput_d_r = []
                for d in range(self.D):
                    index_r = self.X_list_missing[d][:, -1] == r
                    x_output_d_r.append(self.X_list_missing[d][index_r][:, :-1])
                    y_output_d_r.append(self.Y_list_missing[d][index_r])
                    Indexoutput_d_r.append(np.ones_like(self.Y_list_missing[d][index_r]) * r)
                x_all.append(np.vstack(x_output_d_r))
                y_all.append(np.vstack(y_output_d_r))
                Indexoutput.append(np.vstack(Indexoutput_d_r))

        Indexoutput = np.vstack(Indexoutput)
        x_all = np.vstack(x_all)
        x_input_all_index = np.hstack((x_all, Indexoutput))
        y_all = np.vstack(y_all)
        Y_all = np.hstack((y_all, Indexoutput))
        data_set = (x_input_all_index, Y_all)

        # model
        Z = x_input_all_index[::self.gap, 0][:, None].copy()
        ks = [gpflow.kernels.RBF() for _ in range(self.Num_ker)]
        L = len(ks)
        Zs = [Z.copy() for _ in range(L)]
        iv_list = [InducingPoints(Z) for Z in Zs]
        iv = SeparateIndependentInducingVariables(iv_list)
        N = x_input_all_index.shape[0]
        if self.Model_name == 'LMC3':
            likelihood_gp = [gpflow.likelihoods.Gaussian() for _ in range(self.num_replicates)]
            kern = lmc_kernel(self.num_replicates, ks)
        elif self.Model_name == 'LMC2':
            likelihood_gp = [gpflow.likelihoods.Gaussian() for _ in range(self.D)]
            kern = lmc_kernel(self.D, ks)
        lik = gpflow.likelihoods.SwitchedLikelihood(likelihood_gp)
        m_test = SVGP_MOGP(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L, num_data=N)
        return m_test, data_set, x_input_all_index


    def set_up_LMCsum(self):
        '''
        Here we set up the LMC2 model where we consider each replica as each output.
        '''


        if self.Experiment_type == 'Missing_part_of_one_output_in_Whole':
            x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data_Missing_part_of_one_output_in_Whole(self.X_list_missing,
                                                                                           self.Y_list_missing,
                                                                                           self.X_list_missing_test,
                                                                                           self.Y_list_missing_test,
                                                                                           self.num_replicates,
                                                                                           self.D)
            x_input_all_index, All_X_test, Y_all, All_Y_test = new_format_for_X_Y_Missing_part_of_one_output_in_Whole(x_train_all_pre, x_test_all_pre,
                                                                                  y_train_all_pre, y_test_all_pre,
                                                                                  self.num_replicates)

        else:
            # data set
            x_train_all_pre, x_test_all_pre, y_train_all_pre, y_test_all_pre = LMCsum_data(self.X_list_missing,
                                                                                           self.Y_list_missing,
                                                                                           self.X_list_missing_test,
                                                                                           self.Y_list_missing_test,
                                                                                           self.num_replicates,
                                                                                           self.D)
            #
            x_input_all_index, All_X_test, Y_all, All_Y_test = new_format_for_X_Y(x_train_all_pre, x_test_all_pre,y_train_all_pre, y_test_all_pre, self.num_replicates)




        data_set = (x_input_all_index, Y_all)

        # model
        Z = x_input_all_index[::self.gap, 0][:, None].copy()
        ks = [gpflow.kernels.RBF() for _ in range(self.Num_ker)]
        L = len(ks)
        Zs = [Z.copy() for _ in range(L)]
        iv_list = [InducingPoints(Z) for Z in Zs]
        iv = SeparateIndependentInducingVariables(iv_list)
        N = x_input_all_index.shape[0]
        likelihood_gp = [gpflow.likelihoods.Gaussian() for _ in range(self.D)]
        kern = lmc_kernel(self.D, ks)
        lik = gpflow.likelihoods.SwitchedLikelihood(likelihood_gp)
        m_test = SVGP_MOGP_sum(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L, num_data=N, num_replicates=self.num_replicates)



        return m_test, data_set, x_input_all_index



    def set_up_LVMOGP(self):
        '''
        Here we set up the LVMOGP model where we consider all replicas in the same output as one output.
        '''

        # data set
        Indexoutput = []
        for d in range(self.D):
            Indexoutput.append(np.ones_like(self.Y_list_missing[d]) * d)
        Indexoutput = np.vstack(Indexoutput)
        indexD = Indexoutput.squeeze()
        x_all = np.vstack(self.X_list_missing)[:, :-1]
        Y_all = np.vstack(self.Y_list_missing)
        data_set = (x_all, Y_all)
        Mc = x_all[::self.gap].shape[0]
        ## model
        m_test = GPy.models.GPMultioutRegressionMD(x_all, Y_all, indexD, Xr_dim=self.Q, kernel_row=GPy.kern.RBF(self.Q, ARD=True),
                                              num_inducing=(Mc, self.Mr), init='GP')
        return m_test, data_set, x_all

    def set_up_LVMOGP2(self):
        '''
        Here we set up the LVMOGP2 model where we consider each replica data as each output.
        '''

        # data set
        Indexoutput = []
        x_all = []
        y_all = []
        if self.Model_name == 'LVMOGP2':
            for d in range(self.D):
                index_r = self.X_list_missing[d][:, -1] == self.test_id
                x_all.append(self.X_list_missing[d][index_r][:, :-1])
                y_all.append(self.Y_list_missing[d][index_r])
                Indexoutput.append(np.ones_like(self.Y_list_missing[d][index_r]) * d)
        elif self.Model_name == 'LVMOGP3' and self.Experiment_type == 'Train_test_in_each_replica':
            for r in range(self.num_replicates):
                index_r = self.X_list_missing[:, -1] == r
                x_all.append(self.X_list_missing[index_r][:, :-1])
                y_all.append(self.Y_list_missing[index_r])
                Indexoutput.append(np.ones_like(self.Y_list_missing[index_r]) * r)
        elif self.Model_name == 'LVMOGP3' and self.Experiment_type == 'Missing_One_replica_in_each_ouput':
            for r in range(self.num_replicates):
                x_output_d_r = []
                y_output_d_r = []
                Indexoutput_d_r = []
                for d in range(self.D):
                    index_r = self.X_list_missing[d][:, -1] == r
                    x_output_d_r.append(self.X_list_missing[d][index_r][:, :-1])
                    y_output_d_r.append(self.Y_list_missing[d][index_r])
                    Indexoutput_d_r.append(np.ones_like(self.Y_list_missing[d][index_r]) * r)
                x_all.append(np.vstack(x_output_d_r))
                y_all.append(np.vstack(y_output_d_r))
                Indexoutput.append(np.vstack(Indexoutput_d_r))

        Indexoutput = np.vstack(Indexoutput)
        indexD = Indexoutput.squeeze()
        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        data_set = (x_all, y_all)

        Mc = x_all[::self.gap].shape[0]
        ## model
        m_test = GPy.models.GPMultioutRegressionMD(x_all, y_all, indexD, Xr_dim=self.Q, kernel_row=GPy.kern.RBF(self.Q, ARD=True),
                                              num_inducing=(Mc, self.Mr), init='GP')
        return m_test, data_set, x_all