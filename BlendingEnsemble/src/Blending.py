#Import the internal modules written for the purpose of this project
from BlendingEnsemble.src.utils_data_processing import (LoadData, cwts, getpath, rnd_state)
from BlendingEnsemble.src.utils_features_engineering import (FeaturesCreation, FeaturesTransformation, FeaturesSelection)
from BlendingEnsemble.src.utils_model_and_tuning import (Blending, HpTuning, SimpleBacktest, Btest)

#Import external modules for the basemodels and blender (metamodel)
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#data manipulation modules

# Creates a folder for saving of code graphics and trading strategy report.
output_path = getpath()

# Load Data
def step1LoadData(data_files, time_period, plotdata=True):
    ldata = LoadData(*time_period, **data_files)  #instantiate the class
    df = ldata.joinData() #Call the function to merger all the data into a single dataframe
    ldata.checkNullData(df)  # within the ldata object, call the checkNullData method, with the df as input parameter
    df = ldata.fixNullData(df, method='bfill')
    ldata.checkNullData(df)
    # fix future data that are not available yet, can drop rows to choose, knn_impute method was used here.
    df = ldata.fixNullData(df, method='knnimpute')
    #Plot charts
    if plotdata:
        ldata.plotCandleStick(df)
        ldata.plotPrices(df)

    return df #return clean-up dataframe for next step


### Feature Engineering
def step2Features(df, short_prd=5, medium_prd=10, upper_std=2, lower_std=1, hurdle=0.005,
                  upper_threshold=0.065, lower_threshold=0.03):
    # Instantiate the FeaturesCreation subclass providing the dataframe and the target parameters as inputs
    feat_df = FeaturesCreation(df, short_prd, medium_prd, upper_std, lower_std, hurdle)
    new_ft = feat_df.create_all_features(fundamental_features=True, macro_features=True)
    feat_transform = FeaturesTransformation(new_ft)  # Instantiate the FeaturesTransformation
    new_ft2 = feat_transform.transformDaysColumn()
    new_ft2 = new_ft2.astype('float64')
    new_ft2['predict'] = new_ft2['predict'].values.astype('int16')

    feat_select = FeaturesSelection(new_ft2, testsize = 0.20)
    feat1 = feat_select.wrapper_boruta(max_iter=150) # Call the wrapper_boruta method within the FeaturesSelection subclass.
    filtered_features = feat_select.filter_multicollinearity(corr_coeff=0.90)
    data2 = new_ft2[filtered_features]
    data2['predict'] = new_ft2['predict'].values.astype('int')

    kmeans_features = feat_select.kmeans_selector(data2, cluster_size=len(filtered_features),
                                                  upper_threshold=upper_threshold, lower_threshold=lower_threshold)
    data3 = new_ft2[kmeans_features]
    data3['predict'] = new_ft2['predict'].values.astype('int')
    return data3



def step3ModelParams(df):
    cls_weight = cwts(df) #generate class weight to treat class imbalance

    # Logistic regression algorithm
    lr_params = {'random_state': rnd_state(), 'class_weight': cls_weight}
    lr = LogisticRegression(**lr_params)

    # Decision Tree algorithm
    dt_params = {'class_weight': cls_weight, 'random_state': rnd_state()}
    dt = DecisionTreeClassifier(**dt_params)

    # K-nearest Neighbour algorithm
    knn_params = {'algorithm': 'auto', 'n_jobs': -1}
    knn = KNeighborsClassifier(**knn_params)

    # Gaussian Naive Bayes algorithm
    bayes_params = {}
    bayes = GaussianNB()
    bayes.set_params(**bayes_params)

    # Support Vector Machine (SVM): Support Vector Classifier (SVC)
    svc_params = {'class_weight': cls_weight,'random_state': rnd_state(), 'probability': True}
    svc = SVC(**svc_params)

    # Combining all the algorithms into basemodels
    basemodels = {'lr': lr, 'dte': dt, 'knn': knn, 'bayes': bayes, 'svc': svc}
    basemodels_params = {'lr': lr_params, 'dt': dt_params, 'knn': knn_params,
                         'bayes': bayes_params, 'svc': svc_params}

    # Extreme Gradient Boost algorithm
    xgb_params = {'n_jobs': -1, 'class_weight': cls_weight, 'random_state': rnd_state(), 'verbose': 1}
    xgb = XGBClassifier(**xgb_params)

    # Extreme gradient boosting stated as metamodel or blender.
    blender = xgb
    blender_params = {'xgb': xgb_params}
    return basemodels, blender, basemodels_params, blender_params


def step4RunModel(df, basemodels, blender, testsize=0.20):
    # Separate final X and y - Features and target
    X_final = df.iloc[:, :-1].values
    y_final = df.iloc[:, -1].values

    Blnd = Blending(X_final, y_final, basemodels, blender, testsize=testsize, valsize=0.20)
    acc, f1score, ypred, yprob, yfull = Blnd.runBlendingEnsemble()

    print(f"Accuracy Score: {acc: .1%}, f1score: {f1score:.1%}")

    return X_final, y_final, acc, f1score, ypred, yprob, yfull


# #### Hyperparameter Tuning
#Instantiate tuning
def step5TuneModel(X_final, y_final, n_trials=40):
    tune_model = HpTuning(X_final, y_final, n_trials=n_trials)
    tuned_lr, tuned_dt, tuned_svc, tuned_knn, tuned_bayes, tuned_xgb = (tune_model.optimize_lr(),
                                                                        tune_model.optimize_dt(),
                                                                        tune_model.optimize_svc(),
                                                                        tune_model.optimize_knn(),
                                                                        tune_model.optimize_bayes(),
                                                                        tune_model.optimize_xgb())

    print("optimal_lr:", tuned_lr.values, "\t","optimal_dt:", tuned_dt.values, "\t",
          "optimal_svc:", tuned_svc.values, "\t", "optimal_knn:", tuned_knn.values, "\t",
          "optimal_bayes:", tuned_bayes.values, "\t", "optimal_xgb:", tuned_xgb.values)

    tuned_vals = {'lr': tuned_lr, 'dt': tuned_dt, 'svc':tuned_svc, 'knn': tuned_knn,
                  'bayes':tuned_bayes, 'xgb':tuned_xgb}
    return tuned_vals


def update_dict(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], list):
                dict1[key].extend(value)
            else:
                dict1[key] = value
        else:
            dict1[key] = value
    return dict1

def step6RunTunedModel(basemodels, blender, tuned_values):
    lr_params = update_dict(basemodels['lr'], tuned_values['lr'].params)
    lr = LogisticRegression(**lr_params)

    dt_params = update_dict(basemodels['dt'], tuned_values['dt'].params)
    dt = DecisionTreeClassifier(**dt_params)

    knn_params = update_dict(basemodels['knn'], tuned_values['knn'].params)
    knn = KNeighborsClassifier(**knn_params)

    bayes = GaussianNB()

    svc_params = update_dict(basemodels['svc'], tuned_values['svc'].params)
    svc = SVC(**svc_params)

    basemodel_upd = {'lr': lr, 'dt': dt, 'knn': knn, 'bayes': bayes, 'svc': svc}

    xgb_params = update_dict(blender, tuned_values['xgb'].params)
    xgb = XGBClassifier(**xgb_params)

    blender_upd = xgb

    return basemodel_upd, blender_upd

def step7RunBacktest1(df, rtn_prd, ypred, company_name=None):
    btd = SimpleBacktest(df)
    btdd = btd.approach1(ypred, rtn_prd)
    sharpeRatio = btd.sharpe_ratios(btdd)
    print(sharpeRatio)
    return btd.html_report(company_name=company_name)

def step8RunBacktest2(df, ypred):
    bto_lib = Btest(df, ypred)
    bto_lib.runStrategy()
    btostats = bto_lib.runstats()
    print(btostats)
    return bto_lib.plotstats()

