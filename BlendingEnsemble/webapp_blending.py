from BlendingEnsemble.src.Blending import (step1LoadData, step2Features, step3ModelParams,
                                           step4RunModel, step5TuneModel, step6RunTunedModel,
                                           step7RunBacktest1, step8RunBacktest2)


def main():
    try:

        data_files = {'files': ['NVDA', 'VVIX_History', 'USCPI', 'USGDP', 'FedFundRate', '2yrTreasury', '10yrTreasury']}

        time_period = ['2008', '2024']  #specifies a period range, in the case provided data is goes back than required.
        company_name = data_files['files'][0]  #Extract the company name or ticker into the company_name variable

        df1 = step1LoadData(data_files, time_period, plotdata=False) #Load Data and print charts step
        df2 = step2Features(df1) #Features engineering steps
        basemod, blend, basemod_params, blend_params = step3ModelParams(df2) #Deriving the base and meta model step

        X_fin, y_fin, acc, f1score, ypred, yprob, yfull = step4RunModel(df2, basemod, blend) #Initial model run step/get initial output

        t_vals = step5TuneModel(X_fin, y_fin, n_trials=40) #Tuning the Hyperparameters step

        basemod2, blend2 = step6RunTunedModel(basemod_params, blend_params, t_vals) #Run tuned model step

        X_fin2, y_fin2, acc2, f1score2, ypred2, yprob2, yfull2 = step4RunModel(df2, basemod2, blend2) #Get final output

        report = step7RunBacktest1(df1, 1, ypred2, 'Nvidia') #Backtesting step
        print()
        report2 = step8RunBacktest2(df1, ypred2)  #Backtesting2 step

    except Exception as e:
        print("Exception occurred: ", e)


if __name__ == '__main__':
    main()
