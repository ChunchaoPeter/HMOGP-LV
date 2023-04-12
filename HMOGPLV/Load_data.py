

                                #######################################################
                                #################### Load data set ####################
                                #######################################################
import gpflow
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import tensorflow as tf
from gpflow import default_float
import random
import HMOGPLV
from HMOGPLV.amc_parser import parse_amc
from sklearn.preprocessing import scale


class Data_set(object):
    def __init__(self, cfg, seed):
        self.num_replicates = cfg.SYN.NUM_REPLICATES
        self.num_data_each_replicate = cfg.SYN.NUM_DATA_IN_REPLICATES
        self.D = cfg.SYN.NUM_OUTPUTS
        self.Noise = cfg.SYN.NOISE * self.D
        self.Max_range = cfg.SYN.MAX_RANGE
        self.line_range = cfg.SYN.LINE_RANGE
        ## Model Parameters
        self.Q = cfg.MODEL.Q
        self.data_path = cfg.PATH.DATA_PATH
        ## Misc options
        self.Data_name = cfg.MISC.DATA_NAME
        self.Num_data_sampled = cfg.MISC.NUM_SAMPLED
        self.marker_list = cfg.MISC.MARKER_LIST
        self.Experiment_type = cfg.MISC.EXPERIMENTTYPE
        self.train_percentage = cfg.SYN.TRAIN_PERCENTAGE

        self.seed = seed

    def Load_data_set(self):
        MOCAP = ['MOCAP7', 'MOCAP8', 'MOCAP9', 'MOCAP12', 'MOCAP16', 'MOCAP35', 'MOCAP49', 'MOCAP64','MOCAP118']
        if self.Data_name  == 'Synthetic_different_input':
            X_all_outputs_with_replicates, Y_list = self.synthetic_data_different_inputs_different_seed()
            return X_all_outputs_with_replicates, Y_list
        elif self.Data_name in MOCAP:
            print(self.Data_name)
            X_all_outputs_with_replicates, Y_list = self.MOCAP_data_simulate()
            return X_all_outputs_with_replicates, Y_list
        elif self.Data_name == 'Gene':
            X_all_outputs_with_replicates, Y_list = self.Gene_dataset()
            return X_all_outputs_with_replicates, Y_list
        elif self.Data_name == 'GeneLarge':
            X_all_outputs_with_replicates, Y_list = self.GeneLarge_dataset()
            return X_all_outputs_with_replicates, Y_list


    def find_y_ProbeID_gene(self, dataset, ProbeID_number):
        '''
        Here we assume the ProbeID as an gene, ID as the individual
        '''
        Y = [] ## the whole output
        for i in range(len(ProbeID_number)):
            same_ProbeID = dataset.loc[dataset['ProbeID']==ProbeID_number[i]]
            aa_test = same_ProbeID.sort_values(by=['ID','Timestamp'])
            Y_1 = np.array(aa_test['Input'])[:,None] ## one output
            Y.append(Y_1)
        return Y



    def synthetic_data_different_inputs_different_seed(self):
        '''
        Here we build our sythentic data set
        :param num_replicates:
        num_replicates: the number of replicates in each output
        Noise: a list the noise for each output
        Max_range: it is upper bound of the range
        line_range: it is the upper bound of the line range
        D: the number of outputs
        Q: the dimension of H
        kernel_H: kernel for outputs
        k_hierarchy: kernel for inputs
        seed: it is the random seed

        Return
        X_all_outputs_with_replicates: a list that includes all outputs with index
        Y: a vector where all the output come together
        idx_replicates: it is the index for each replicates. In this case, different replicates has a different index
        index_output: index for the output
        X_outputs: a list that includes all outputs
        Index_all_replicates: a vector that include all replicates index where different replicates that have different index
        '''
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        ### kernel for the input X
        Total_num_replicates = self.num_replicates
        kern_upper = gpflow.kernels.Matern32(variance=1.0, lengthscales=1.0)
        kern_lower = gpflow.kernels.Matern32(variance=1.0, lengthscales=1.0)
        k_hierarchy = HMOGPLV.kernels.Hierarchial_kernel_replicated(kernel_g=kern_upper, kernel_f=[kern_lower],
                                                                    total_replicated=Total_num_replicates)
        ### kernel for the H
        # lengthscales2 = tf.convert_to_tensor([1] * self.Q, dtype=default_float())
        lengthscales2 = tf.convert_to_tensor([1] * 2, dtype=default_float())
        kernel_H = gpflow.kernels.RBF(lengthscales = lengthscales2)

        ### Here we build different inputs
        X_outputs = []
        for d in range(self.D):
            X_outputs.append(np.random.RandomState(seed=d*self.seed).uniform(low=0.0, high=self.Max_range, size=self.num_data_each_replicate))
        ### Here we set up all the output with index
        X_all_outputs_with_replicates = []
        for d in range(self.D):
            X_output_with_replicates = []
            for i in range(self.num_replicates):
                X_output_with_replicates.append(np.c_[X_outputs[d], np.ones_like(X_outputs[d]) * i])
            X_all_outputs_with_replicates.append(np.vstack(X_output_with_replicates))
        ## Calcualte for the hidden variable H
        H = []
        # for i in range(self.Q): we change it into 2 to make sure all the model have same synthetic data
        for i in range(2):
            H.append(np.linspace(0.0, self.line_range, num=self.D)[:, None])
        H = np.hstack(H)
        ## Calculate the covariance
        whole_matrix = []
        for i in range(self.D):
            row_matrix = []
            for j in range(self.D):
                row_matrix.append(k_hierarchy.K(X_all_outputs_with_replicates[i], X_all_outputs_with_replicates[j]))
            whole_matrix.append(np.hstack(row_matrix))
        cov_c_different_input = np.vstack(whole_matrix)
        cov_output = kernel_H.K(H)
        D_list = []
        for d in range(self.D):
            D_list.append(X_all_outputs_with_replicates[d].shape[0])
        cov_all_different_input = np.repeat(np.repeat(cov_output, D_list, axis=0), D_list,
                                                axis=1) * cov_c_different_input
        ## Find the index for each replicates
        Index_all_replicates_raw = [X_all_outputs_with_replicates[d][:, -1] + d * self.num_replicates for d in range(self.D)]
        Index_all_replicates = np.hstack(Index_all_replicates_raw)
        idx_replicates = [Index_all_replicates == i for i in range(self.D * self.num_replicates)]
        ## Find the index for the output
        index_output = np.repeat(np.arange(self.D), D_list, axis=0)
        ## Simulate the dataset
        K_tril = tf.linalg.cholesky(cov_all_different_input)
        F = tfp.distributions.MultivariateNormalTriL(tf.zeros(sum(D_list), dtype=K_tril.dtype), K_tril).sample().numpy()
        Noise_all = np.repeat(self.Noise, D_list, axis=0) * np.random.RandomState(seed=self.seed).randn(*F.shape)
        Y = F + Noise_all
        y = Y.reshape(-1, self.D, order='F')
        Y_list = []
        for d in range(self.D):
            Y_list.append(y[:, d][:, None])
        return X_all_outputs_with_replicates, Y_list

    def MOCAP_data_simulate(self):
        '''
        :param data_path: It is the path of our data
        :return:
        X_all_outputs_with_replicates: It is a list: x with inputs with replicates
        Y_list: It is a list with Y corresponding to x input with replicates
        '''
        if self.Data_name == 'MOCAP7':
            subject = 7
            all_trial = [8, 9, 10]
        elif self.Data_name == 'MOCAP8':
            subject = 8
            all_trial = [2, 3, 8, 9]
        elif self.Data_name == 'MOCAP9':
            subject = 9
            all_trial = [1, 2, 3, 5, 6, 11]
        elif self.Data_name == 'MOCAP12':
            subject = 12
            all_trial = [1, 3]
        elif self.Data_name == 'MOCAP16':
            subject = 16
            all_trial = [15, 31, 58]
        elif self.Data_name == 'MOCAP35':
            subject = 35
            all_trial = [29, 31, 34]
        elif self.Data_name == 'MOCAP49':
            subject = 49
            all_trial = [18, 19, 20]
        elif self.Data_name == 'MOCAP64':
            subject = 64
            all_trial = [3, 4, 5, 7, 8, 9]
        elif self.Data_name == 'MOCAP118':
            subject = 118
            all_trial = [3, 4, 11, 17]
        X_all_trial, Y_all_trail = self.extract_data(all_trial, self.marker_list, self.D, self.data_path, subject)
        X_all_outputs_with_replicates_sampled, Y_list_sampled = self.sample_dataset(self.D, self.num_replicates, self.Num_data_sampled, self.seed, X_all_trial, Y_all_trail)
        return X_all_outputs_with_replicates_sampled, Y_list_sampled

    def sample_dataset(self, D, num_replicates, Num_data_sampled, seed, X_replicates, Y_list):
        ### Here we set up all the output with index
        X_all_outputs_with_replicates = []
        for d in range(D):
            X_output_with_replicates = []
            for i in range(num_replicates):
                X_output_with_replicates.append(np.c_[X_replicates[i], np.ones_like(X_replicates[i]) * i])
            X_all_outputs_with_replicates.append(X_output_with_replicates)
            ## We build all sampled data set
        Whole_all = []
        for d in range(D):
            Whole_each = []
            for i in range(num_replicates):
                Whole_each.append(np.hstack((X_all_outputs_with_replicates[d][i], Y_list[d][i])))
            Whole_all.append(Whole_each)
        ## We sample our data set
        Whole_all_sampled = []
        for d in range(D):
            Whole_each_sampled = []
            for i in range(num_replicates):
                random_indices = np.random.RandomState(seed=seed).choice(Whole_all[d][i].shape[0],
                                                                         size=Num_data_sampled, replace=False)
                random_rows = Whole_all[d][i][random_indices, :]
                Whole_each_sampled.append(random_rows)
            Whole_all_sampled.append(Whole_each_sampled)
        ## We split whole data into X and Y
        X_all_outputs_with_replicates_sampled = []
        Y_list_sampled = []
        for d in range(D):
            X_output_with_replicates_sampled = []
            Y_list_each_sampled = []
            for i in range(num_replicates):
                X_output_with_replicates_sampled.append(Whole_all_sampled[d][i][:, :-1])
                Y_list_each_sampled.append(Whole_all_sampled[d][i][:, -1][:, None])
            X_all_outputs_with_replicates_sampled.append(np.vstack(X_output_with_replicates_sampled))
            Y_list_sampled.append(np.vstack(Y_list_each_sampled))
        return X_all_outputs_with_replicates_sampled, Y_list_sampled

    def extract_data(self, All_trial, All_useful_set, D, data_path, subject):
        ## Return X_replicates_all: a list. List of input of replica. Each part of list means the input
        ## for the all the output in that replica.

        ## Return Y_list_all. Each part of list is a output with all replicates
        X_all_trail = []
        Y_all_trail = []
        for i in range(len(All_trial)):
            Y_trail = []
            for m in All_useful_set:
                Xs_trail, Ys_trail = self.mocap_data(data_path, subject=subject, trial=All_trial[i], marker_list=m[0])
                Y_trail.append(Ys_trail[m[1]])
            X_all_trail.append(Xs_trail)
            Y_all_trail.append(Y_trail)

        ## The Y_list: each list is a output with all replicates
        Y_list_all = []
        for d in range(D):
            Y_output = []
            for i in range(len(All_trial)):
                Y_output.append(Y_all_trail[i][d])
            Y_list_all.append(Y_output)

        ## The replicates
        X_replicates_all = []
        for i in range(len(All_trial)):
            X_replicates_all.append(X_all_trail[i][0])
        return X_replicates_all, Y_list_all


    def Gene_dataset(self):
        '''
        We obtain some gene dataset here
        '''
        ## There are six outputs with eight replicates
        expression = np.loadtxt(self.data_path + 'Gene_data_set/kalinka09_mel.csv', delimiter=',', usecols=range(1, 57))
        gene_names = np.loadtxt(self.data_path + 'Gene_data_set/kalinka09_mel.csv', delimiter=',', usecols=[0], dtype=np.str)
        replicates, times = np.loadtxt(self.data_path + 'Gene_data_set/kalinka09_mel_pdata.csv', delimiter=',').T
        # clustered_gene_names = np.array(['ac', 'bib', 'yellow-e3', 'Tom', 'CG13333', 'tld'], dtype=np.str)
        clustered_gene_names = np.array(['CG12723', 'CG13196', 'CG13627', 'Osi15'], dtype=np.str)
        gene_index_all = [i for i, gn in enumerate(gene_names) if gn in clustered_gene_names]
        # normalize data row-wise
        expression -= expression.mean(1)[:, np.newaxis]
        expression /= expression.std(1)[:, np.newaxis]
        Y_cluster = expression[np.random.RandomState(seed=self.seed).permutation(gene_index_all)]
        Y_list = [y.reshape(-1, 1) for y in Y_cluster]
        X_all_outputs_with_replicates = []
        for i in range(len(gene_index_all)):
            X_all_outputs_with_replicates.append(np.hstack((times.reshape(-1, 1), replicates.reshape(-1, 1) - 1)))
        return X_all_outputs_with_replicates, Y_list


    def GeneLarge_dataset(self):
        '''
        We obtain some gene dataset here
        '''
        ## There are six outputs with eight replicates
        expression = np.loadtxt(self.data_path + 'Gene_data_set/kalinka09_mel.csv', delimiter=',', usecols=range(1, 57))
        gene_names = np.loadtxt(self.data_path + 'Gene_data_set/kalinka09_mel.csv', delimiter=',', usecols=[0], dtype=np.str)
        replicates, times = np.loadtxt(self.data_path + 'Gene_data_set/kalinka09_mel_pdata.csv', delimiter=',').T
        # clustered_gene_names = np.array(['ac', 'bib', 'yellow-e3', 'Tom', 'CG13333', 'tld'], dtype=np.str)
        np.random.seed(0)
        clustered_gene_names = gene_names[np.random.choice(gene_names.shape[0], self.D, replace=False)]
        # clustered_gene_names = np.array(['CG12723', 'CG13196', 'CG13627', 'Osi15'], dtype=np.str)
        gene_index_all = [i for i, gn in enumerate(gene_names) if gn in clustered_gene_names]
        # normalize data row-wise
        expression -= expression.mean(1)[:, np.newaxis]
        expression /= expression.std(1)[:, np.newaxis]
        Y_cluster = expression[np.random.RandomState(seed=self.seed).permutation(gene_index_all)]
        Y_list = [y.reshape(-1, 1) for y in Y_cluster]
        X_all_outputs_with_replicates = []
        for i in range(len(gene_index_all)):
            X_all_outputs_with_replicates.append(np.hstack((times.reshape(-1, 1), replicates.reshape(-1, 1) - 1)))
        return X_all_outputs_with_replicates, Y_list


    def mocap_data(self, path, subject, trial, marker_list):
        cmu_meta = pd.read_csv(path + "CMU_Mocap/cmu_meta.csv", index_col="full_id")
        cmu_meta = cmu_meta.loc[f"{subject:02}_{trial:02}"]

        amc_path = path + cmu_meta["path"]

        motion = parse_amc(amc_path)
        # print("NumFrames:", len(motion))

        markers = np.array(list(motion[0].keys()))[marker_list]
        # print("Markers:", markers)

        # labels = [m + str(i) for m in markers for i in range(len(motion[0][m]))]
        # print("Labels:", labels)

        ## The Y is a matrix.
        ## The row of Y are the values of makres for each recording.
        ## E.g., Y[0]=[head0, head1, head2, lowerneck0,lowerneck1,
        ##             lowerneck2,upperneck0,upperneck1,upperneck2]
        ##     is the first record
        Y = scale(np.hstack([np.array([frame[m] for frame in motion]) for m in markers]))

        Ys = [Yi[:, None] for Yi in Y.T]

        ## We also scale index X with zero mean and 1 for std
        X = scale(np.arange(len(Ys[0]), dtype=float)[:, None])
        Xs = [X for _ in Ys]
        return Xs, Ys

    def obtain_different_experiment_type_data(self, seed_index, X_all_outputs_with_replicates, Y_list):
        '''
        index_seed: this is used for changing seed for each kind of cross-validation
        x_raw:The input data with replicates index
        y: It is a matrix; column is the output
        train_percentage: the percentage of training points
        num_data_each_replicate: the number of data in each replicate
        num_replicates: the number of replicates

        Return:
        x_train, x_test, y_train, y_test
        '''
        X_list_missing = []
        Y_list_missing = []
        X_list_missing_test = []
        Y_list_missing_test = []
        if self.Experiment_type == 'Missing_One_replica_in_Whole':
            for d in range(self.D):
                if d == seed_index:
                    Index_no_missing = X_all_outputs_with_replicates[d][:, -1] != ((d + seed_index) % self.num_replicates)
                    Index_missing = X_all_outputs_with_replicates[d][:, -1] == ((d + seed_index) % self.num_replicates)
                    X_list_missing.append(X_all_outputs_with_replicates[d][Index_no_missing])
                    Y_list_missing.append(Y_list[d][Index_no_missing])
                    X_list_missing_test.append(X_all_outputs_with_replicates[d][Index_missing])
                    Y_list_missing_test.append(Y_list[d][Index_missing])
                else:
                    X_list_missing.append(X_all_outputs_with_replicates[d])
                    Y_list_missing.append(Y_list[d])
            return X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test, 1, 1
        elif self.Experiment_type == 'Missing_One_replica_in_each_ouput':
            for d in range(self.D):
                Index_no_missing = X_all_outputs_with_replicates[d][:, -1] != ((d + seed_index) % self.num_replicates)
                Index_missing = X_all_outputs_with_replicates[d][:, -1] == ((d + seed_index) % self.num_replicates)
                X_list_missing.append(X_all_outputs_with_replicates[d][Index_no_missing])
                Y_list_missing.append(Y_list[d][Index_no_missing])
                X_list_missing_test.append(X_all_outputs_with_replicates[d][Index_missing])
                Y_list_missing_test.append(Y_list[d][Index_missing])
            return X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test, 1, 1
        elif self.Experiment_type == 'Missing_part_of_one_replica_in_Whole':
            for d in range(self.D):
                if d == seed_index:
                    Index_no_missing = X_all_outputs_with_replicates[d][:, -1] != ((d + seed_index) % self.num_replicates)
                    Index_missing = X_all_outputs_with_replicates[d][:, -1] == ((d + seed_index) % self.num_replicates)
                    X_no_missing_train = X_all_outputs_with_replicates[d][Index_no_missing]
                    Y_no_missing_train = Y_list[d][Index_no_missing]

                    ## We find the training and test data in the missing replica in X
                    X_missing = X_all_outputs_with_replicates[d][Index_missing]
                    indices = np.random.RandomState(seed=(d + seed_index)).permutation(self.num_data_each_replicate)
                    num_train = round(self.train_percentage * self.num_data_each_replicate)
                    training_id, test_id = indices[:num_train], indices[num_train:]
                    X_missing_train = X_missing[training_id]
                    X_missing_test = X_missing[test_id]

                    ## We find the training and test data in the missing replica in y
                    Y_missing = Y_list[d][Index_missing]
                    Y_missing_train = Y_missing[training_id]
                    Y_missing_test = Y_missing[test_id]

                    ## The training data and test data
                    X_train_each_output = np.vstack((X_no_missing_train, X_missing_train))
                    Y_train_each_output = np.vstack((Y_no_missing_train, Y_missing_train))

                    X_list_missing.append(X_train_each_output)
                    Y_list_missing.append(Y_train_each_output)
                    X_list_missing_test.append(X_missing_test)
                    Y_list_missing_test.append(Y_missing_test)
                else:
                    X_list_missing.append(X_all_outputs_with_replicates[d])
                    Y_list_missing.append(Y_list[d])
            return X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test, training_id, test_id
        elif self.Experiment_type == 'Missing_part_of_one_output_in_Whole':
            for d in range(self.D):
                if d == (self.D-1):
                    ## We find the index for each output
                    Num_data_replicates_in_output = int(X_all_outputs_with_replicates[d].shape[0] / self.num_replicates)
                    indices = np.random.RandomState(seed=(d + seed_index)).permutation(Num_data_replicates_in_output)
                    num_train = round(self.train_percentage * Num_data_replicates_in_output)
                    training_idx, test_idx = indices[:num_train], indices[num_train:]
                    Train_index_y = np.hstack(
                        [training_idx + num * Num_data_replicates_in_output for num in np.arange(self.num_replicates)])
                    Test_index_y = np.hstack(
                        [test_idx + num * Num_data_replicates_in_output for num in np.arange(self.num_replicates)])
                    ### We hand made an train/test split for each output
                    x_train, x_test = X_all_outputs_with_replicates[d][Train_index_y], X_all_outputs_with_replicates[d][
                        Test_index_y]
                    y_train, y_test = Y_list[d][Train_index_y], Y_list[d][Test_index_y]
                    X_list_missing.append(x_train)
                    X_list_missing_test.append(x_test)
                    Y_list_missing.append(y_train)
                    Y_list_missing_test.append(y_test)
                else:
                    X_list_missing.append(X_all_outputs_with_replicates[d])
                    Y_list_missing.append(Y_list[d])
            return X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test, 1, 1
        elif self.Experiment_type == 'Missing_part_of_one_replica_in_each_ouput':
            training_id = []
            test_id = []
            for d in range(self.D):
                Index_no_missing = X_all_outputs_with_replicates[d][:, -1] != ((d + seed_index) % self.num_replicates)
                Index_missing = X_all_outputs_with_replicates[d][:, -1] == ((d + seed_index) % self.num_replicates)
                X_no_missing_train = X_all_outputs_with_replicates[d][Index_no_missing]
                Y_no_missing_train = Y_list[d][Index_no_missing]

                ## We find the training and test data in the missing replica in X
                X_missing = X_all_outputs_with_replicates[d][Index_missing]
                indices = np.random.RandomState(seed=(d + seed_index)).permutation(self.num_data_each_replicate)
                num_train = round(self.train_percentage * self.num_data_each_replicate)
                training_idx, test_idx = indices[:num_train], indices[num_train:]
                X_missing_train = X_missing[training_idx]
                X_missing_test = X_missing[test_idx]
                training_id.append(training_idx)
                test_id.append(test_idx)
                ## We find the training and test data in the missing replica in y
                Y_missing = Y_list[d][Index_missing]
                Y_missing_train = Y_missing[training_idx]
                Y_missing_test = Y_missing[test_idx]
                ## The training data and test data
                X_train_each_output = np.vstack((X_no_missing_train, X_missing_train))
                Y_train_each_output = np.vstack((Y_no_missing_train, Y_missing_train))
                X_list_missing.append(X_train_each_output)
                Y_list_missing.append(Y_train_each_output)
                X_list_missing_test.append(X_missing_test)
                Y_list_missing_test.append(Y_missing_test)
            return X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test, training_id, test_id
        elif self.Experiment_type == 'Train_test_in_each_replica':
            seed_index = seed_index * 1000
            if self.Data_name == 'Gene':
                for d in range(self.D):
                    idx_d = [X_all_outputs_with_replicates[d][:, -1] == i for i in range(self.num_replicates)]
                    Num_data_replicas_all = []
                    for i in range(self.num_replicates):
                        Num_data_replicas_all.append(sum(idx_d[i]))
                    training_ix = []
                    test_ix = []
                    ## find training and test for all replica
                    for j in range(self.num_replicates):
                        indices = np.random.RandomState(seed=(d + seed_index)).permutation(Num_data_replicas_all[j])
                        num_train = int(round(self.train_percentage * Num_data_replicas_all[j]))
                        # print(num_train)
                        training_ix.append(indices[:num_train])
                        test_ix.append(indices[num_train:])
                    ## 0 with cumsum index
                    Num_data_replicas_new_index = Num_data_replicas_all[:-1]
                    Num_data_replicas_new_index.insert(0, 0)
                    index_for_each_D = np.cumsum(Num_data_replicas_new_index)

                    ## Find all index for all y
                    Train_index_y = np.hstack([training_ix[i] + add_index for i, add_index in enumerate(index_for_each_D)])
                    Test_index_y = np.hstack([test_ix[i] + add_index for i, add_index in enumerate(index_for_each_D)])

                    ### We hand made an train/test split for each output
                    x_train, x_test = X_all_outputs_with_replicates[d][Train_index_y], X_all_outputs_with_replicates[d][Test_index_y]
                    y_train, y_test = Y_list[d][Train_index_y], Y_list[d][Test_index_y]
                    X_list_missing.append(x_train)
                    X_list_missing_test.append(x_test)
                    Y_list_missing.append(y_train)
                    Y_list_missing_test.append(y_test)
            else:
                for d in range(self.D):
                    ## We find the index for each output
                    Num_data_replicates_in_output = int(X_all_outputs_with_replicates[d].shape[0] / self.num_replicates)
                    indices = np.random.RandomState(seed=(d + seed_index)).permutation(Num_data_replicates_in_output)
                    num_train = round(self.train_percentage * Num_data_replicates_in_output)
                    training_idx, test_idx = indices[:num_train], indices[num_train:]
                    Train_index_y = np.hstack(
                        [training_idx + num * Num_data_replicates_in_output for num in np.arange(self.num_replicates)])
                    Test_index_y = np.hstack(
                        [test_idx + num * Num_data_replicates_in_output for num in np.arange(self.num_replicates)])
                    ### We hand made an train/test split for each output
                    x_train, x_test = X_all_outputs_with_replicates[d][Train_index_y], X_all_outputs_with_replicates[d][
                        Test_index_y]
                    y_train, y_test = Y_list[d][Train_index_y], Y_list[d][Test_index_y]
                    X_list_missing.append(x_train)
                    X_list_missing_test.append(x_test)
                    Y_list_missing.append(y_train)
                    Y_list_missing_test.append(y_test)
            return X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test, 1, 1
        else:
            print('Please entre the correct experiment type')
