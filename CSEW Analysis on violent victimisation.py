### Crime Survey for England and Wales: 
### interviewer gender, the presence of others, and 
### disclosure of violent victimisation


### Import key packages.
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use("ggplot")

### Import complete CSEW NVF.
nvf_all = pd.read_csv("C:/Users/Render02/Desktop/MSc Data Science/Principles of Data Science/Coursework/0. Analysis/1.2 CSEW_adult_nvf_apr2013_mar2016_short_v1.csv", encoding="UTF-8")
nvf_all.shape #(104045, 57)
nvf_all.head(0)
nan_dim= nvf_all.shape[0] - nvf_all.count() #no NANs

serial_df = nvf_all[["﻿serial"]]
serial_df.to_csv("C:/Users/Render02/Desktop/MSc Data Science/Principles of Data Science/Coursework/0. Analysis/serial.csv")

########## Deprivation scores and attitudes to police.

#### Imshow for deprivation scores
depriv = nvf_all[["depriv_multiple_lsoa_dec", "depriv_income_lsoa_dec", 
                  "depriv_employment_lsoa_dec", "depriv_education_lsoa_dec", 
                  "depriv_health_lsoa_dec"]]
nan_dim= depriv.shape[0] - depriv.count()
depriv.shape #(104045, 5)
depriv.dtypes 

plt.imshow(depriv.corr(), cmap=plt.cm.Greys, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(depriv.columns))]
plt.xticks(tick_marks, depriv.columns, rotation='vertical')
plt.yticks(tick_marks, depriv.columns)
plt.show()

#### Imshow for police attitudes.
police = nvf_all [["police_rating", "police_reliable", "police_respectful", "police_fair", 
 "police_understand", "police_dealing", "police_confidence_overall"]]
nan_dim= police.shape[0] - police.count()
police.shape #(104045, 7)
police.dtypes

plt.imshow(police.corr(), cmap=plt.cm.Greys, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(police.columns))]
plt.xticks(tick_marks, police.columns, rotation='vertical')
plt.yticks(tick_marks, police.columns)
plt.show() 

#### Combine depriv and police data-frames
depriv_police = nvf_all[["depriv_multiple_lsoa_dec", "depriv_income_lsoa_dec", 
                  "depriv_employment_lsoa_dec", "depriv_education_lsoa_dec", 
                  "depriv_health_lsoa_dec", "police_rating", "police_reliable", 
                  "police_respectful", "police_fair", "police_understand", 
                  "police_dealing", "police_confidence_overall"]]
depriv_police.shape # (104045, 12)
depriv_police.dtypes
depriv_police.describe()

#### Standardisation
import math
from sklearn import preprocessing
depriv_police_scaled = preprocessing.StandardScaler().fit_transform(depriv_police)
depriv_police_scaled = pd.DataFrame(depriv_police_scaled, columns = depriv_police.columns)
depriv_police_scaled.shape #(104045, 12)
depriv_police_scaled.describe()

#### PCA
from sklearn.decomposition import PCA
depriv_police_pca = PCA(n_components = 3)
depriv_police_pca.fit(depriv_police_scaled)
depriv_police_pca = depriv_police_pca.transform(depriv_police_scaled)
depriv_police_pca = pd.DataFrame(depriv_police_pca, columns=["PCA_depriv_police_1", "PCA_depriv_police_2", "PCA_depriv_police_3"])
depriv_police_pca.to_csv("C:/Users/Render02/Desktop/MSc Data Science/Principles of Data Science/Coursework/0. Analysis/2.1 depriv_police_pca.csv")
depriv_police_pca.shape #(104045, 3)

#### Visualisation of PCA and victimisation
colors_victim_vio_sex = ["red" if i==1 else "gray" for i in nvf_all.victim_vio_sex]
fig = plt.figure()
plt.suptitle("Deprivation and attitudes to police PCA & Victim of violence or sex attack")
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.scatter(depriv_police_pca.PCA_depriv_police_1, depriv_police_pca.PCA_depriv_police_2, depriv_police_pca.PCA_depriv_police_3, c=colors_victim_vio_sex, marker=".")
plt.show()

colors_victim_vio_sex = ["red" if i==1 else "gray" for i in nvf_all.victim_vio_sex]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.scatter(depriv_police_pca.PCA_depriv_police_1, depriv_police_pca.PCA_depriv_police_2, depriv_police_pca.PCA_depriv_police_3, c=colors_victim_vio_sex, marker=".")
plt.show()
########## Profiling dimensions

profile= nvf_all[["year", "age", "married_cohib", "nchil", "nadults", "vehicle_own",
                  "full_time", "ethnic_white", "religion_christian", "higher_education", 
                  "tenure_own", "urban", "acorn_ctg", "gor_ctg"]]
                  
profile = pd.get_dummies(profile,columns = ["gor_ctg", "acorn_ctg"])
profile.dtypes
profile.head(0)
profile.gor_ctg_EE = profile.gor_ctg_EE.astype("int64")
profile.gor_ctg_EM = profile.gor_ctg_EM.astype("int64")
profile.gor_ctg_L = profile.gor_ctg_L.astype("int64")
profile.gor_ctg_NE = profile.gor_ctg_NE.astype("int64")
profile.gor_ctg_NW = profile.gor_ctg_NW.astype("int64")
profile.gor_ctg_SE = profile.gor_ctg_SE.astype("int64")
profile.gor_ctg_SW = profile.gor_ctg_SW.astype("int64")
profile.gor_ctg_WM = profile.gor_ctg_WM.astype("int64")
profile.gor_ctg_YH = profile.gor_ctg_YH.astype("int64")
profile.acorn_ctg_affluent_achievers = profile.acorn_ctg_affluent_achievers.astype("int64")
profile.acorn_ctg_comfortable_commun = profile.acorn_ctg_comfortable_commun.astype("int64")
profile.acorn_ctg_financially_stretch = profile.acorn_ctg_financially_stretch.astype("int64")
profile.acorn_ctg_rising_prosperity = profile.acorn_ctg_rising_prosperity.astype("int64")
profile.acorn_ctg_urban_advers = profile.acorn_ctg_urban_advers.astype("int64")
profile.dtypes
profile.shape #(104045, 27)
profile.describe()

#### Standardisation
import math
from sklearn import preprocessing
profile_scaled = preprocessing.StandardScaler().fit_transform(profile)
profile_scaled = pd.DataFrame(profile_scaled, columns = profile.columns)
profile_scaled.shape #(104045, 27)
profile_scaled.describe()


#### Singular Value Decomposition
from sklearn import decomposition
profile_svd = decomposition.TruncatedSVD(n_components=3)
profile_svd.fit(profile_scaled)
profile_svd = profile_svd.transform(profile_scaled)
profile_svd = pd.DataFrame(profile_svd, columns = ["SVD_profile_1", "SVD_profile_2", "SVD_profile_3"])
profile_svd.shape #(104045, 3)
profile_svd.to_csv("C:/Users/Render02/Desktop/MSc Data Science/Principles of Data Science/Coursework/0. Analysis/2.2 profile_svd.csv")


colors_victim_vio_sex = ["red" if i==1 else "gray" for i in nvf_all.victim_vio_sex]
fig = plt.figure()
plt.suptitle("Profile SVD & Victim of violence or sex attack")
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("SVD_profile_1")
ax.set_ylabel("SVD_profile_2")
ax.set_zlabel("SVD_profile_3")
ax.scatter(profile_svd.SVD_profile_1, profile_svd.SVD_profile_2, profile_svd.SVD_profile_3, c=colors_victim_vio_sex, marker=".")
plt.show()

colors_victim_vio_sex = ["red" if i==1 else "gray" for i in nvf_all.victim_vio_sex]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("SVD_1")
ax.set_ylabel("SVD_2")
ax.set_zlabel("SVD_3")
ax.scatter(profile_svd.SVD_profile_1, profile_svd.SVD_profile_2, profile_svd.SVD_profile_3, c=colors_victim_vio_sex, marker=".")
plt.show()

#### Multiple Correspondance Analysis
from sklearn import decomposition
profile_svd = decomposition.TruncatedSVD(n_components=3)
profile_svd.fit(profile_scaled)
profile_svd = profile_svd.transform(profile_scaled)
profile_svd = pd.DataFrame(profile_svd, columns = ["SVD_profile_1", "SVD_profile_2", "SVD_profile_3"])
profile_svd.shape #(104045, 3)
profile_svd.to_csv("C:/Users/Render02/Desktop/MSc Data Science/Principles of Data Science/Coursework/0. Analysis/2.2 profile_svd.csv")


colors_victim_vio_sex = ["red" if i==1 else "gray" for i in nvf_all.victim_vio_sex]
fig = plt.figure()
plt.suptitle("Profile SVD & Victim of violence or sex attack")
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("SVD_profile_1")
ax.set_ylabel("SVD_profile_2")
ax.set_zlabel("SVD_profile_3")
ax.scatter(profile_svd.SVD_profile_1, profile_svd.SVD_profile_2, profile_svd.SVD_profile_3, c=colors_victim_vio_sex, marker=".")
plt.show()

####### Regression with cross-fold validation

### Trial 1

X = depriv_police_pca.join(profile_svd, how="outer")
X = X.join(nvf_all.resp_int_male_male, how="outer")
X = X.join(nvf_all.resp_int_male_female, how="outer")
X = X.join(nvf_all.resp_int_female_female, how="outer")
# reference category is female-male --> modal category
X = X.join(nvf_all.present_child, how="outer")
X = X.join(nvf_all.present_partner, how="outer")
X = X.join(nvf_all.present_other, how="outer")
# reference category is no one present
y= nvf_all[["victim_vio_sex"]]


import statsmodels.api as sm
logit = sm.Logit(y, X)
result = logit.fit()
#params = result.params
#conf = result.conf_int()
#conf['OR'] = params
#conf.columns = ['2.5%', '97.5%', 'OR']
print(result.summary())
#print(result.conf_int())
#print(np.exp(result.params))
#print(np.exp(conf))

# the coefficient is an odds ratio
#                              coef    std err          z      P>|z|      [95.0% Conf. Int.]
# ------------------------------------------------------------------------------------------
# PCA_depriv_police_1        0.0953      0.007     13.857      0.000         0.082     0.109
# PCA_depriv_police_2        0.0349      0.007      5.185      0.000         0.022     0.048
# PCA_depriv_police_3       -0.0692      0.015     -4.738      0.000        -0.098    -0.041
# SVD_profile_1              0.0147      0.009      1.581      0.114        -0.004     0.033
# SVD_profile_2              0.1990      0.008     23.847      0.000         0.183     0.215
# SVD_profile_3             -0.2093      0.011    -19.825      0.000        -0.230    -0.189
# resp_int_male_male        -3.7255      0.045    -82.616      0.000        -3.814    -3.637
# resp_int_male_female      -3.5530      0.049    -72.331      0.000        -3.649    -3.457
# resp_int_female_female    -3.7733      0.049    -77.135      0.000        -3.869    -3.677
# present_child             -2.8800      0.077    -37.233      0.000        -3.032    -2.728
# present_partner           -3.6155      0.084    -42.819      0.000        -3.781    -3.450
# present_other             -2.5380      0.074    -34.464      0.000        -2.682    -2.394
# ==========================================================================================


from sklearn.cross_validation import KFold
kf = KFold(n=104045, n_folds=10, random_state="RandomState")
len(kf)
print(kf)

for train_index, test_index in kf:
    X_train = X.loc[train_index, :]
    X_train = sm.add_constant(X_train)
    y_train = y.loc[train_index, :]
    X_test = X.loc[test_index, :]
    X_test = sm.add_constant(X_test)
    y_test = y.loc[test_index,:]
    import statsmodels.api as sm
    logit = sm.Logit(y_train, X_train)
    result = logit.fit()
    #params = result.params
    #conf = result.conf_int()
    #conf['OR'] = params
    #conf.columns = ['2.5%', '97.5%', 'OR']
    print(result.summary())
    #print(result.conf_int())
    #print(np.exp(result.params))
    #print(np.exp(conf))

# Not systematic findings


### Trial 2 - Only males

X = depriv_police_pca.join(profile_svd, how="outer")
X = X.join(nvf_all.resp_int_male_male, how="outer")
# reference category is male-male --> modal category
X = X.join(nvf_all.present_child, how="outer")
X = X.join(nvf_all.present_partner, how="outer")
X = X.join(nvf_all.present_other, how="outer")

X = X.join(nvf_all.resp_female, how="outer")
X = X[X.resp_female==0]
X = X.reset_index(drop=True)
X = X.drop(labels=["resp_female"], axis = 1)
X = sm.add_constant(X)

# reference category is no one present
y= nvf_all[["victim_vio_sex", "resp_female"]]
y= y[y.resp_female==0]
y = y.reset_index(drop=True)
y = y.drop(labels=["resp_female"], axis = 1)


import statsmodels.api as sm
logit = sm.Logit(y, X)
result = logit.fit()
#params = result.params
#conf = result.conf_int()
#conf['OR'] = params
#conf.columns = ['2.5%', '97.5%', 'OR']
print(result.summary())
#print(result.conf_int())
#print(np.exp(result.params))
#print(np.exp(conf))


X.shape #(47274, 11)
y.shape # (47274, 1)

from sklearn.cross_validation import KFold
kf = KFold(n=47274, n_folds=10, random_state="RandomState")
len(kf)
print(kf)

for train_index, test_index in kf:
    X_train = X.loc[train_index, :]
    X_train = sm.add_constant(X_train)
    y_train = y.loc[train_index, :]
    X_test = X.loc[test_index, :]
    X_test = sm.add_constant(X_test)
    y_test = y.loc[test_index,:]
    import statsmodels.api as sm
    logit = sm.Logit(y_train, X_train)
    result = logit.fit()
    #params = result.params
    #conf = result.conf_int()
    #conf['OR'] = params
    #conf.columns = ['2.5%', '97.5%', 'OR']
    print(result.summary())
    #print(result.conf_int())
    #print(np.exp(result.params))
    #print(np.exp(conf))

# Very systematic findings from 10-fold cross-validation


### Trial 3 - Only females
import statsmodels.api as sm
X = depriv_police_pca.join(profile_svd, how="outer")
X = X.join(nvf_all.resp_int_female_female, how="outer")
# reference category is male-male --> modal category
X = X.join(nvf_all.present_child, how="outer")
X = X.join(nvf_all.present_partner, how="outer")
X = X.join(nvf_all.present_other, how="outer")

X = X.join(nvf_all.resp_female, how="outer")
X = X[X.resp_female==1]
X = X.reset_index(drop=True)
X = X.drop(labels=["resp_female"], axis = 1)
X = sm.add_constant(X)

# reference category is no one present
y= nvf_all[["victim_vio_sex", "resp_female"]]
y= y[y.resp_female==1]
y = y.reset_index(drop=True)
y = y.drop(labels=["resp_female"], axis = 1)


import statsmodels.api as sm
logit = sm.Logit(y, X)
result = logit.fit()
#params = result.params
#conf = result.conf_int()
#conf['OR'] = params
#conf.columns = ['2.5%', '97.5%', 'OR']
print(result.summary())
#print(result.conf_int())
#print(np.exp(result.params))
#print(np.exp(conf))

#                             coef    std err          z      P>|z|      [95.0% Conf. Int.]
# ------------------------------------------------------------------------------------------
# PCA_depriv_police_1        0.1035      0.008     13.602      0.000         0.089     0.118
# PCA_depriv_police_2        0.0105      0.007      1.400      0.162        -0.004     0.025
# PCA_depriv_police_3       -0.0673      0.016     -4.121      0.000        -0.099    -0.035
# SVD_profile_1             -0.0363      0.010     -3.583      0.000        -0.056    -0.016
# SVD_profile_2              0.2132      0.009     23.483      0.000         0.195     0.231
# SVD_profile_3             -0.1967      0.012    -17.047      0.000        -0.219    -0.174
# resp_int_female_female    -3.7598      0.049    -76.848      0.000        -3.856    -3.664
# present_child             -3.1527      0.089    -35.351      0.000        -3.327    -2.978
# present_partner           -4.7305      0.145    -32.533      0.000        -5.016    -4.446
# present_other             -3.1144      0.097    -32.075      0.000        -3.305    -2.924
# ==========================================================================================

X.shape #(56771, 10)
y.shape # (56771, 1)

from sklearn.cross_validation import KFold
kf = KFold(n=56771, n_folds=10, random_state="RandomState")
len(kf)
print(kf)

for train_index, test_index in kf:
    X_train = X.loc[train_index, :]
    X_train = sm.add_constant(X_train)
    y_train = y.loc[train_index, :]
    X_test = X.loc[test_index, :]
    X_test = sm.add_constant(X_test)
    y_test = y.loc[test_index,:]
    import statsmodels.api as sm
    logit = sm.Logit(y_train, X_train)
    result = logit.fit()
    #params = result.params
    #conf = result.conf_int()
    #conf['OR'] = params
    #conf.columns = ['2.5%', '97.5%', 'OR']
    print(result.summary())
    #print(result.conf_int())
    #print(np.exp(result.params))
    #print(np.exp(conf))

# Not sytematic findings from 10-fold cross-validation

##### Smoothing

estimates= pd.read_excel("C:/Users/Render02/Desktop/MSc Data Science/Principles of Data Science/Coursework/0. Analysis/2.5 LA estimates_df.xlsx")
estimates.head(0)
estimates.describe()

estimates_imp= estimates.drop(labels=["la_code"], axis = 1)
estimates_imp=estimates_imp.interpolate(method='linear', order=2)
estimates_imp.describe()

from sklearn import preprocessing 
estimates_imp_scaled = preprocessing.RobustScaler(with_centering=True, with_scaling=True).fit_transform(estimates_imp)
estimates_imp_scaled = pd.DataFrame(estimates_imp_scaled, columns = estimates_imp.columns)
estimates_imp_scaled.shape #(352, 18)
estimates_imp_scaled.describe()

estimates_imp_scaled = estimates_imp_scaled.join(estimates.la_code, how="outer")
estimates_imp_scaled.to_csv("C:/Users/Render02/Desktop/MSc Data Science/Principles of Data Science/Coursework/0. Analysis/2.6 LA estimates_scaled_df.csv")

