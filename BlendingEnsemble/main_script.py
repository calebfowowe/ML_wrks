"""
Blending Ensemble Model - CQF Final Project
Using Streamlit to deploy the ML app. | Caleb Fowowe
"""
from BlendingEnsemble.src.Blending import (step1LoadData, step2Features, step3ModelParams,
                                           step4RunModel, step5TuneModel, step6RunTunedModel,
                                           step7RunBacktest1, step8RunBacktest2, config_to_dict)


import streamlit as st
from datetime import datetime

# params_dict = config_to_dict('config_file.xlsx')
# print(params_dict['tuning_ntrials'])


def main():
    #params_dict = config_to_dict('config_file') #config file

    try:
        #Upload parameters config file
        params_dict = config_to_dict('config_file.xlsx')
        data_files = {'files': [item.strip() for item in
                                params_dict['files'].split(",")]}  #extract the files & remove whitespaces

        time_period = [item.strip() for item in params_dict['time_period'].split(",")]  #extract config time range
        company_name = params_dict['company_name']  #Extract the company name or ticker into the company_name variable

        df1 = step1LoadData(data_files, time_period, plotdata=True)  #Load Data and print charts step

        df2 = step2Features(df1)  #Features engineering steps
        basemod, blend, basemod_params, blend_params = step3ModelParams(df2)  #Deriving the base and meta model step

        # Initial model run step/get initial output
        X_fin, y_fin, acc, f1score, ypred, yprob, yfull = step4RunModel(df2, basemod, blend)

        t_vals = step5TuneModel(X_fin, y_fin, n_trials=params_dict['tuning_ntrials'])  #Tuning the Hyperparameters step

        basemod2, blend2 = step6RunTunedModel(basemod_params, blend_params, t_vals)  #Run tuned model step

        X_fin2, y_fin2, acc2, f1score2, ypred2, yprob2, yfull2 = step4RunModel(df2, basemod2,
                                                                               blend2)  #Get final output

        report = step7RunBacktest1(df1, 1, ypred2, company_name)  #Backtesting step
        print()
        report2 = step8RunBacktest2(df1, ypred2)  #Backtesting2 step


    except Exception as e:
        print("Exception occurred: ", e)


if __name__ == '__main__':
    main()
