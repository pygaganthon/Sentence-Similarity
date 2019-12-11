# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:31:56 2019

@author: maitr
"""

import pandas as pd
from nltk.corpus import wordnet as wn
import spacy
nlp = spacy.load('en')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
import numpy as np
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus.reader import NOUN
from nltk.corpus.reader import VERB
#Load dataset into dataframe
df=pd.read_csv('C:/Users/maitr/OneDrive/Desktop/NLP_Project_4Dec//data/train-set.txt', sep="\t",error_bad_lines=False)
df1=pd.read_csv('C:/Users/maitr/OneDrive/Desktop/NLP_Project_4Dec//data/dev-set.txt', sep="\t",error_bad_lines=False)


df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)


#df=pd.read_csv('C:/Users/gagan/OneDrive/Desktop/NLPProject/data/sample_train.txt', sep="\t",error_bad_lines=False)
df_features=pd.read_csv('C:/Users/maitr/OneDrive/Desktop/NLP_Project_4Dec//df_features_train//df_features_traintCopy.txt', sep="\t",error_bad_lines=False)
df1_features=pd.read_csv('C:/Users/maitr/OneDrive/Desktop/NLP_Project_4Dec//df_features_dev//df_features_devtxtt.txt', sep="\t",error_bad_lines=False)


df_features.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df1_features.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)



from sklearn.model_selection import train_test_split
import numpy as np
#Split data into training and test sets


df_model_f=df_features.copy()
df1_model_f=df1_features.copy()

df_model_f.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df1_model_f.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

df_model_f=df_model_f.drop(['id', 'Sentence1', 'Sentence2','Token_Sentence1',
                            'Hyper_Sentence1','Hyper_Sentence2','Hypo_Sentence1',
                            'Hypo_Sentence2','Holo_Sentence1','Holo_Sentence2',
                            'Mero_Sentence1','Mero_Sentence2','Token_Sentence2',
                            'Common_tokens', 'PosTag_Sentence1',
       'PosTag_Sentence2', 'Lema_Sentence1', 'Lema_Sentence2',
       'PosTagLema_Sentence1', 'PosTagLema_Sentence2','EdgePath_Dep_Wt_by_lch','Common_Hyper_Score','Common_prepositions','ALl_EdgePath_Dep_Wt_by_lch'],axis=1)



df1_model_f=df1_model_f.drop(['id', 'Sentence1', 'Sentence2','Token_Sentence1',
       'Token_Sentence2', 'Common_tokens', 'PosTag_Sentence1',
       'PosTag_Sentence2', 'Lema_Sentence1', 'Lema_Sentence2',
       'PosTagLema_Sentence1', 'PosTagLema_Sentence2','EdgePath_Dep_Wt_by_lch','Common_Hyper_Score','Common_prepositions','ALl_EdgePath_Dep_Wt_by_lch'],axis=1)

#'Common_Hyper_Score','Common_prepositions','ALl_EdgePath_Dep_Wt_by_lch','EdgePath_Dep_Wt_by_lch'
list_features=np.array([
                           
                            list(df_model_f['Norm_NN_bypat']),
                            list(df_model_f['Norm_VB_bypat']),
                            list(df_model_f['Norm_NN_bylch']),
                            list(df_model_f['Norm_VB_bylch']),
                            list(df_model_f['Norm_NN_bywup']),
                            list(df_model_f['Norm_VB_bywup']),
#                            list(df_model_f['Common_prepositions']),
#                            list(df_model_f["Common_Hyper_Score"]),
                           
                            list(df_model_f['ALl_EdgePath_Dep_Wt_by_pat']),
                            list(df_model_f['EdgePath_Dep_Wt_by_pat']),
                            list(df_model_f['Tag_Dep_Wt_by_pat']),
                            list(df_model_f['Set_Tag_Dep_Wt_by_pat']),
                            list(df_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_pat']),
                       
                            list(df_model_f['ALl_EdgePath_Dep_Wt_by_wup']),
                            list(df_model_f['EdgePath_Dep_Wt_by_wup']),
                            list(df_model_f['Tag_Dep_Wt_by_wup']),
                            list(df_model_f['Set_Tag_Dep_Wt_by_wup']),
                            list(df_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_wup']),
                       
#                            list(df_model_f['ALl_EdgePath_Dep_Wt_by_lch']),
#                            list(df_model_f['EdgePath_Dep_Wt_by_lch']),
                            list(df_model_f['Tag_Dep_Wt_by_lch']),
                            list(df_model_f['Set_Tag_Dep_Wt_by_lch']),
                            list(df_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_lch']),
                       
                       
                        ])

list_features_dev=np.array([
                           
                            list(df1_model_f['Norm_NN_bypat']),
                            list(df1_model_f['Norm_VB_bypat']),
                            list(df1_model_f['Norm_NN_bylch']),
                            list(df1_model_f['Norm_VB_bylch']),
                            list(df1_model_f['Norm_NN_bywup']),
                            list(df1_model_f['Norm_VB_bywup']),
#                            list(df1_model_f['Common_prepositions']),
#                            list(df1_model_f["Common_Hyper_Score"]),
                           
                            list(df1_model_f['ALl_EdgePath_Dep_Wt_by_pat']),
                            list(df1_model_f['EdgePath_Dep_Wt_by_pat']),
                            list(df1_model_f['Tag_Dep_Wt_by_pat']),
                            list(df1_model_f['Set_Tag_Dep_Wt_by_pat']),
                            list(df1_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_pat']),
                       
                            list(df1_model_f['ALl_EdgePath_Dep_Wt_by_wup']),
                            list(df1_model_f['EdgePath_Dep_Wt_by_wup']),
                            list(df1_model_f['Tag_Dep_Wt_by_wup']),
                            list(df1_model_f['Set_Tag_Dep_Wt_by_wup']),
                            list(df1_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_wup']),
                       
#                            list(df1_model_f['ALl_EdgePath_Dep_Wt_by_lch']),
#                            list(df1_model_f['EdgePath_Dep_Wt_by_lch']),
                            list(df1_model_f['Tag_Dep_Wt_by_lch']),
                            list(df1_model_f['Set_Tag_Dep_Wt_by_lch']),
                            list(df1_model_f['Set_Ovefit_EdgePath_Dep_Wt_by_lch'])
                       
                       
                        ])



list_features=np.transpose(list_features)
list_features_dev=np.transpose(list_features_dev)

#X = list_features
ylabels = list(df_model_f['Gold Tag'])


X_train=list_features
X_test=list_features_dev
y_train=list(df_model_f['Gold Tag'])
y_test=list(df1_model_f['Gold Tag'])



#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.1,random_state=500,shuffle=False)




def Majority_vote(X_train,X_test,y_train,y_test):
    import pickle
#    from sklearn.model_selection import train_test_split
#    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2,random_state=500,shuffle=False)
    
    ###Model_Building
    #from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    #
    classifier=LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
    classifier=classifier.fit(X_train,y_train)
    
    #save model
    pkl_filename = "Logistic_Regression.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(classifier, file)
#    c=classifier.predict(X_test)
    
    
    classifier1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    classifier1=classifier1.fit(X_train,y_train)
    
    #Save Model
    pkl_filename1 = "Random_ForestClassifier.pkl"
    with open(pkl_filename1, 'wb') as file:
        pickle.dump(classifier1, file)
    
#    c1=classifier1.predict(X_test)
    
    
#    from sklearn.svm import SVC 
#    classifier2 = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
#    c2 = classifier2.predict(X_test)
    
    from sklearn.naive_bayes import MultinomialNB
    classifier3 = MultinomialNB()
    classifier3=classifier3.fit(X_train, y_train)
    
    #Save Model
    pkl_filename2 = "MultinomialNB.pkl"
    with open(pkl_filename2, 'wb') as file:
        pickle.dump(classifier3, file)

    #Load models
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    c=pickle_model.predict(X_test)
    with open(pkl_filename1, 'rb') as file:
        pickle_model1 = pickle.load(file)
    c1=pickle_model1.predict(X_test)
    with open(pkl_filename2, 'rb') as file:
        pickle_model2 = pickle.load(file)
    c2=pickle_model2.predict(X_test)

    
    
    mvotedlabel=[]
    for ind in range(len(c)):
        if c[ind]==c1[ind] and c[ind]==c2[ind]:
            mvotedlabel.append(c[ind])
        elif c[ind]==c1[ind] and c[ind]!=c2[ind]:
            mvotedlabel.append(c[ind])       
        elif c1[ind]==c2[ind] and c[ind]!=c1[ind]:
            mvotedlabel.append(c1[ind])
        elif c[ind]==c2[ind] and c[ind]!=c1[ind]:
            mvotedlabel.append(c[ind])
        else:
            #Random Forest
            mvotedlabel.append(c1[ind])
    
    from sklearn import metrics
    accuracy=metrics.accuracy_score(y_test, c)
    accuracy1=metrics.accuracy_score(y_test, c1)
    accuracy2=metrics.accuracy_score(y_test, c2)
    
    print(accuracy,accuracy1,accuracy2)
    
    accuracy3=metrics.accuracy_score(y_test, mvotedlabel)
    print("Majority Vote",accuracy3)
    
    return mvotedlabel,accuracy


predicted_labels,Accuracy_mvoted=Majority_vote(X_train,X_test,y_train,y_test)


data=df1.copy()
data=data.drop(['Sentence1','Sentence2'],axis=1)
#num_rows, num_cols = X_test.shape
##data_test=data.loc[num_rows,:]
#data_test=data[num_rows:]
#data_test=data.copy()

data['Gold Tag']=data['Gold Tag'].astype('int')


data_predict=data.copy()
data_predict['Gold Tag']=predicted_labels
data_predict['Gold Tag']=data_predict['Gold Tag'].astype('int')

data.to_csv('dev-set.txt',sep='\t',index=None)
data_predict.to_csv('dev-set-predicted-answers.txt',sep='\t',index=None)






#data=df.copy()
#data=data.drop(['Sentence1','Sentence2'],axis=1)
#num_rows, num_cols = X_train.shape
#data_test=data.iloc[num_rows,:]
#data_test['Gold Tag']=data_test['Gold Tag'].astype('int')
#
#
#data_predict=data_test.copy()
#data_predict['Gold Tag']=predicted_labels
#data_predict['Gold Tag']=data_predict['Gold Tag'].astype('int')
#
#data_test.to_csv('DTest.txt',sep='\t',index=None)
#data_predict.to_csv('DTestPredict.txt',sep='\t',index=None)
