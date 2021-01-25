import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
   
if __name__ == "__main__":

    df=pd.read_csv('data.csv')
    train , test = data_split(df,0.2)
    X_train=train['FEVER','BODYPAIN','RUNNYNOSE','DIFFIBREATHING','Age','Outside Visit ','Throat Pain','dry cough','loss of taste ','loss of smell'].to_numpy()
    X_test=test['FEVER','BODYPAIN','RUNNYNOSE','DIFFIBREATHING','Age','Outside Visit ','Throat Pain','dry cough','loss of taste ','loss of smell'].to_numpy()
    ##X_test=test[['FEVER','BODYPAIN','RUNNYNOSE','DIFFIBREATHING']].to_numpy()
    Y_train=train['INFECTIONPROB'].to_numpy().reshape(1624,)
    Y_test=test['INFECTIONPROB'].to_numpy().reshape(405,)
##from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression()
    clf.fit(X_train,Y_train)

    file=open('model.pkl','wb')
    pickle.dump(clf,file)
    file.close()
    inputfeature=[100,1,0,0,65,1,0,0,1,0]
    infprob=clf.predict_proba([inputfeature])[0][1]

# if __name__ == "__main__":
    