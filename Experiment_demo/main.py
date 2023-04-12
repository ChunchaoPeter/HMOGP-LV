import sys

import tensorflow as tf
import random

random.seed(3)
tf.random.set_seed(24)
import argparse
from HMOGPLV.trainer import Trainer
from HMOGPLV.config import get_cfg_defaults_all
from HMOGPLV.Load_data import Data_set
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def arg_parse():
    """
    Parsing arguments
    This function is used for pass the parameter from the yaml file.
    """
    parser = argparse.ArgumentParser(description="Configure files")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()
    return args


def main():
    # ---- setup configures ----
    args = arg_parse()
    cfg = get_cfg_defaults_all()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # ---- setup dataset ----
    setup_data = Data_set(cfg, 1)
    X_all_outputs_with_replicates, Y_list = setup_data.Load_data_set()
    print('X', X_all_outputs_with_replicates)
    print('Y', Y_list)
    # ---- all the result ----
    Glo_NMSE_test_all = []
    Glo_MNLP_all = []
    NMSE_test_bar_missing_replicates_all = []
    MNLP_missing_replicates_all = []
    Time_all = []

    MOCAP = ['MOCAP7', 'MOCAP8', 'MOCAP12', 'MOCAP16', 'MOCAP35', 'MOCAP49', 'MOCAP64']

    # cross-validation for our model
    for i in range(cfg.MISC.NUM_REPETITION):

        # ---- setup dataset format ----
        X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test, training_id, test_id = setup_data.obtain_different_experiment_type_data(
            i, X_all_outputs_with_replicates, Y_list)

        # --- evaluation metric ---
        Glo_NMSE_test = []
        Glo_MNLP = []
        NMSE_test_bar_missing_replicates = []
        MNLP_missing_replicates = []

        if cfg.MISC.EXPERIMENTTYPE == 'Train_test_in_each_replica' and (
                cfg.MISC.MODEL_NAME == 'HGPInd' or cfg.MISC.MODEL_NAME == 'HGP' or cfg.MISC.MODEL_NAME == 'DNN' or cfg.MISC.MODEL_NAME == 'SGP' or cfg.MISC.MODEL_NAME == 'SGP2' or cfg.MISC.MODEL_NAME == 'LMC3' or cfg.MISC.MODEL_NAME == 'LVMOGP3'):
            ## In this part, we run two single output Gaussian processes. There are training and test data set in each replica.
            ## We thus run each single Gaussian process for each output.
            for d in range(cfg.SYN.NUM_OUTPUTS):
                if cfg.MISC.MODEL_NAME == 'SGP2':
                    idx_train_d = [X_list_missing[d][:, -1] == i for i in range(cfg.SYN.NUM_REPLICATES)]
                    idx_test_d = [X_list_missing_test[d][:, -1] == i for i in range(cfg.SYN.NUM_REPLICATES)]
                    for r in range(cfg.SYN.NUM_REPLICATES):
                        # ---- setup trainer ---- (we let d be test_id in trainer)
                        trainer = Trainer(X_list_missing[d][idx_train_d[r]], X_list_missing_test[d][idx_test_d[r]],
                                          Y_list_missing[d][idx_train_d[r]], Y_list_missing_test[d][idx_test_d[r]],
                                          X_all_outputs_with_replicates, Y_list, training_id, d, i + r * 100, cfg)

                        # ---- setup model ----
                        m_test, data_set, x_input_all_index = trainer.set_up_model()

                        # ---- optimization ----
                        Total_time, logf, m_test = trainer.optimization(m_test, data_set, x_input_all_index)

                        # ---- plot ----
                        # if cfg.MISC.DATA_NAME == 'Synthetic_different_input' or cfg.MISC.DATA_NAME == 'Gene' or cfg.MISC.DATA_NAME == 'MOCAP8':
                        # trainer.save_plot_synthetic(m_test)

                        # ---- prediction ----
                        Glo_NMSE_test_d, Glo_MNLP_d, NMSE_test_bar_missing_replicates_d, MNLP_missing_replicates_d = trainer.prediction_for_model(
                            m_test)

                        Glo_NMSE_test.append(Glo_NMSE_test_d)
                        Glo_MNLP.append(Glo_MNLP_d)
                        NMSE_test_bar_missing_replicates.append(NMSE_test_bar_missing_replicates_d)
                        MNLP_missing_replicates.append(MNLP_missing_replicates_d)
                else:
                    # ---- setup trainer ---- (we let d be test_id in trainer)
                    trainer = Trainer(X_list_missing[d], X_list_missing_test[d], Y_list_missing[d],
                                      Y_list_missing_test[d], X_all_outputs_with_replicates, Y_list, training_id, d, i,
                                      cfg)

                    # ---- setup model ----
                    m_test, data_set, x_input_all_index = trainer.set_up_model()

                    # ---- optimization ----
                    Total_time, logf, m_test = trainer.optimization(m_test, data_set, x_input_all_index)

                    # ---- save elbo, parameters, plot ----
                    if cfg.MISC.DATA_NAME == 'Gene' or cfg.MISC.DATA_NAME == 'MOCAP8':
                        if cfg.MISC.MODEL_NAME == 'HGP' or cfg.MISC.MODEL_NAME == 'SGP' or cfg.MISC.MODEL_NAME == 'LVMOGP3':
                            trainer.save_plot_synthetic(m_test)
                    elif cfg.MISC.MODEL_NAME == 'DNN':
                        print('DNN')
                    else:
                        # trainer.save_elbo(logf)
                        trainer.save_parameter(m_test)
                        # trainer.save_plot_synthetic(m_test)

                    # ---- prediction ----
                    Glo_NMSE_test_d, Glo_MNLP_d, NMSE_test_bar_missing_replicates_d, MNLP_missing_replicates_d = trainer.prediction_for_model(
                        m_test)

                    Glo_NMSE_test.append(Glo_NMSE_test_d)
                    Glo_MNLP.append(Glo_MNLP_d)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar_missing_replicates_d)
                    MNLP_missing_replicates.append(MNLP_missing_replicates_d)

        elif cfg.MISC.EXPERIMENTTYPE == 'Train_test_in_each_replica' and (
                cfg.MISC.MODEL_NAME == 'LVMOGP2' or cfg.MISC.MODEL_NAME == 'LMC2'):
            ## In this part, we run two multi-output Gaussian processes. We consider each replica data as each output.
            ## We thus run each multi-output Gaussian process for each replica.
            for r in range(cfg.SYN.NUM_REPLICATES):
                # ---- setup trainer ----
                trainer = Trainer(X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test,
                                  X_all_outputs_with_replicates, Y_list, training_id, r, i, cfg)

                # ---- setup model ----
                m_test, data_set, x_input_all_index = trainer.set_up_model()

                # ---- optimization ----
                Total_time, logf, m_test = trainer.optimization(m_test, data_set, x_input_all_index)

                # ---- save elbo, parameters, plot ----
                # trainer.save_plot_synthetic(m_test)

                # ---- prediction ----
                Glo_NMSE_test_d, Glo_MNLP_d, NMSE_test_bar_missing_replicates_d, MNLP_missing_replicates_d = trainer.prediction_for_model(
                    m_test)

                Glo_NMSE_test.append(Glo_NMSE_test_d)
                Glo_MNLP.append(Glo_MNLP_d)
                NMSE_test_bar_missing_replicates.append(NMSE_test_bar_missing_replicates_d)
                MNLP_missing_replicates.append(MNLP_missing_replicates_d)
        else:
            #  the missing data set
            if cfg.MISC.EXPERIMENTTYPE == 'Missing_One_replica_in_each_ouput' and (
                    cfg.MISC.MODEL_NAME == 'SGP' or cfg.MISC.MODEL_NAME == 'DNN'):
                for d in range(cfg.SYN.NUM_OUTPUTS):
                    # ---- setup trainer ---- (we let d be test_id in trainer)
                    trainer = Trainer(X_list_missing[d], X_list_missing_test[d], Y_list_missing[d],
                                      Y_list_missing_test[d], X_all_outputs_with_replicates, Y_list, training_id, d, i,
                                      cfg)

                    # ---- setup model ----
                    m_test, data_set, x_input_all_index = trainer.set_up_model()

                    # ---- optimization ----
                    Total_time, logf, m_test = trainer.optimization(m_test, data_set, x_input_all_index)

                    # ---- plot ----
                    # if cfg.MISC.DATA_NAME == 'Synthetic_different_input':
                    # trainer.save_plot_synthetic(m_test)

                    # ---- prediction ----
                    Glo_NMSE_test_d, Glo_MNLP_d, NMSE_test_bar_missing_replicates_d, MNLP_missing_replicates_d = trainer.prediction_for_model(
                        m_test)

                    Glo_NMSE_test.append(Glo_NMSE_test_d)
                    Glo_MNLP.append(Glo_MNLP_d)
                    NMSE_test_bar_missing_replicates.append(NMSE_test_bar_missing_replicates_d)
                    MNLP_missing_replicates.append(MNLP_missing_replicates_d)

            else:
                # ---- setup trainer ----
                trainer = Trainer(X_list_missing, X_list_missing_test, Y_list_missing, Y_list_missing_test,
                                  X_all_outputs_with_replicates, Y_list, training_id, test_id, i, cfg)

                # ---- setup model ----
                m_test, data_set, x_input_all_index = trainer.set_up_model()

                # ---- optimization ----
                Total_time, logf, m_test = trainer.optimization(m_test, data_set, x_input_all_index)

                # ---- save elbo, parameters, plot ----
                if cfg.MISC.MODEL_NAME == 'DHGP' or cfg.MISC.MODEL_NAME == 'LVMOGP' or cfg.MISC.MODEL_NAME == 'LVMOGP3' or cfg.MISC.MODEL_NAME == 'HGP' or cfg.MISC.MODEL_NAME == 'HGPInd':
                    if cfg.MISC.DATA_NAME == 'Synthetic_different_input' or cfg.MISC.DATA_NAME == 'Gene' or cfg.MISC.DATA_NAME == 'MOCAP8':
                        # trainer.save_plot_synthetic(m_test)
                        trainer.save_parameter(m_test)
                else:
                    # trainer.save_elbo(logf)
                    trainer.save_parameter(m_test)
                    # if cfg.MISC.DATA_NAME == 'Synthetic_different_input' or cfg.MISC.DATA_NAME == 'Gene'or cfg.MISC.DATA_NAME == 'MOCAP8':
                    #     trainer.save_plot_synthetic(m_test)

                # ---- prediction ----
                Glo_NMSE_test, Glo_MNLP, NMSE_test_bar_missing_replicates, MNLP_missing_replicates = trainer.prediction_for_model(
                    m_test)

        # ---- output ----
        Time_all.append(Total_time)
        Glo_NMSE_test_all.append(Glo_NMSE_test)
        Glo_MNLP_all.append(Glo_MNLP)
        NMSE_test_bar_missing_replicates_all.append(NMSE_test_bar_missing_replicates)
        MNLP_missing_replicates_all.append(MNLP_missing_replicates)

    # ---- save output result ----
    trainer.save_result(Glo_NMSE_test_all, Glo_MNLP_all, NMSE_test_bar_missing_replicates_all,
                        MNLP_missing_replicates_all, Time_all)


if __name__ == '__main__':
    print('We begin to run')
    main()
