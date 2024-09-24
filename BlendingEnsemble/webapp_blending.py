"""
Blending Ensemble Model - CQF Final Project
Using Streamlit to deploy the ML app. | Caleb Fowowe
"""

from BlendingEnsemble.src.Blending import step1LoadData, step2Features, step3ModelParams, step4RunModel, step5TuneModel, step6RunTunedModel, step7RunBacktest1, step8RunBacktest2, config_to_dict

import streamlit as st
from datetime import datetime


def main():
    # params_dict = config_to_dict('config_file') #config file
    global addon_features

    # Page setting
    st.set_page_config(layout="wide")

    # Title
    # st.markdown('<h1><div style="color: cornflowerblue">Blending Ensemble Model Webapp</div></h1>', unsafe_allow_html=True)
    st.title("Blending Ensemble Model Webapp")
    st.write("#### Caleb Fowowe")
    st.write('-----')

    # Data
    st.sidebar.title("Parameter Settings")
    config_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "txt", "xlsx"])

    try:
        #Upload parameters config file
        params_dict = config_to_dict(config_file)
        data_files = {'files': [item.strip() for item in
                                params_dict['files'].split(",")]}  #extract the files & remove whitespaces

        # Features type
        other_features_type = st.sidebar.radio(
            label="Non-Technical Analysis Features",
            options=[
                'Yes',
                'No'
            ]
        )
        if other_features_type == 'Yes':
            # Features Type
            additional_features_data = sorted(
                st.sidebar.multiselect(
                    label='additional_features',
                    options=[
                        'Fundamental Features',
                        'Macro_Features',
                    ]
                )
            )
            if additional_features_data == 'Fundamental Features' & 'Macro_Features':
                opt = [True, True]
            elif additional_features_data == 'Fundamental Features':
                opt = [True, False]
            elif additional_features_data == 'Macro_Features':
                opt = [False, True]
            else:
                opt = [False, False]

            addon_features = st.sidebar.radio(
                label="Add-on Features",
                options=opt
            )

        #Get train_test_split
        train_percentage = st.sidebar.slider(
            label="Training Dataset (%)",
            min_value=50,
            max_value=90,
            value=80,
            step=2
        )
        st.sidebar.write(f"Train/Test {train_percentage} : {100 - train_percentage}")

        time_period = [item.strip() for item in params_dict['time_period'].split(",")]  #extract config time range
        company_name = params_dict['company_name']  #Extract the company name or ticker into the company_name variable

        df1 = step1LoadData(data_files, time_period, plotdata=True)  #Load Data and print charts step

        # On button click
        if st.sidebar.button("Run Algorithm"):
            df2 = step2Features(df1, fund_feat=addon_features[0],
                                macro_feat=addon_features[1])  #Features engineering steps
            basemod, blend, basemod_params, blend_params = step3ModelParams(df2)  #Deriving the base and meta model step

            test_percent = (100 - train_percentage) / 100

            # Initial model run step/get initial output
            X_fin, y_fin, acc, f1score, ypred, yprob, yfull = step4RunModel(df2, basemod, blend, testsize=test_percent)

            t_vals = step5TuneModel(X_fin, y_fin, n_trials=40)  #Tuning the Hyperparameters step

            basemod2, blend2 = step6RunTunedModel(basemod_params, blend_params, t_vals)  #Run tuned model step

            X_fin2, y_fin2, acc2, f1score2, ypred2, yprob2, yfull2 = step4RunModel(df2, basemod2,
                                                                                   blend2)  #Get final output

            # Click-on
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Training", f"{train_percentage}%")
            c2.metric("Test", f"{100 - train_percentage}%")
            c3.metric("Accuracy", f"{acc2:.2f}%")
            c4.metric("F1-score", f"{f1score2:.2f}%")
            st.write("---")

            report = step7RunBacktest1(df1, 1, ypred2, company_name)  #Backtesting step
            print()
            report2 = step8RunBacktest2(df1, ypred2)  #Backtesting2 step

        st.sidebar.write("")
        st.sidebar.markdown(
            "[LinkedIn](https://www.linkedin.com/in/calebfowowe/) | [Twitter](https://twitter.com/CalebFowowe)")
        st.sidebar.text(f"Caleb Fowowe, \u00A9 {datetime.now().year}")

    except Exception as e:
        print("Exception occurred: ", e)


if __name__ == '__main__':
    main()
