import numpy as np

def csv2data(Australia, Canada, England, India, US):

    Australia_data = np.loadtxt(Australia, delimiter = ",", dtype = np.int32)
    Canada_data = np.loadtxt(Canada, delimiter = ",", dtype = np.int32)
    England_data = np.loadtxt(England, delimiter = ",", dtype = np.int32)
    India_data = np.loadtxt(India, delimiter = ",", dtype = np.int32)
    US_data = np.loadtxt(US, delimiter = ",", dtype = np.int32)

    Australia_num=1220000
    Canada_num=1280000
    England_num=1150000
    India_num=1225000
    US_num=1215000

    num=1150000

    train_num=num-(num//5)*2 #690000
    val_num=num//5 #230000
    test_num=num//5 #230000

    Australia_train = Australia_data[:train_num,:-1] #732000
    Australia_val = Australia_data[train_num:train_num+val_num,:-1]
    Australia_test = Australia_data[train_num+val_num:train_num+val_num+test_num,:-1]

    Canada_train = Canada_data[:train_num,:-1] #768000
    Canada_val = Canada_data[train_num:train_num+val_num,:-1]
    Canada_test = Canada_data[train_num+val_num:train_num+val_num+test_num,:-1]

    England_train = England_data[:train_num,:-1] #690000
    England_val = England_data[train_num:train_num+val_num,:-1]
    England_test = England_data[train_num+val_num:train_num+val_num+test_num,:-1]

    India_train = India_data[:train_num,:-1] #735000
    India_val = India_data[train_num:train_num+val_num,:-1]
    India_test = India_data[train_num+val_num:train_num+val_num+test_num,:-1]

    US_train = US_data[:train_num,:-1] #729000
    US_val = US_data[train_num:train_num+val_num,:-1]
    US_test = US_data[train_num+val_num:train_num+val_num+test_num,:-1]

    train_data = np.concatenate((Australia_train,Canada_train,England_train,India_train,US_train),axis=0)
    val_data = np.concatenate((Australia_val,Canada_val,England_val,India_val,US_val),axis=0)
    test_data = np.concatenate((Australia_test,Canada_test,England_test,India_test,US_test),axis=0)

    return train_data, val_data, test_data


def csv2label(Australia, Canada, England, India, US):
    
    Australia_data = np.loadtxt(Australia, delimiter = ",", dtype = np.int32)
    Canada_data = np.loadtxt(Canada, delimiter = ",", dtype = np.int32)
    England_data = np.loadtxt(England, delimiter = ",", dtype = np.int32)
    India_data = np.loadtxt(India, delimiter = ",", dtype = np.int32)
    US_data = np.loadtxt(US, delimiter = ",", dtype = np.int32)

    Australia_num=1220000
    Canada_num=1280000
    England_num=1150000
    India_num=1225000
    US_num=1215000

    num=1150000

    train_num=num-(num//5)*2 #690000
    val_num=num//5 #230000
    test_num=num//5 #230000

    divider=230

    Australia_train = Australia_data[:train_num//divider,-1] #1830
    Australia_val = Australia_data[train_num:train_num+val_num//divider,-1] #610
    Australia_test = Australia_data[train_num+val_num:train_num+val_num+test_num//divider,-1] #610

    Canada_train = Canada_data[:train_num//divider,-1] #1920
    Canada_val = Canada_data[train_num:train_num+val_num//divider,-1] #640
    Canada_test = Canada_data[train_num+val_num:train_num+val_num+test_num//divider,-1] #640

    England_train = England_data[:train_num//divider,-1] #1725
    England_val = England_data[train_num:train_num+val_num//divider,-1] #575
    England_test = England_data[train_num+val_num:train_num+val_num+test_num//divider,-1] #575

    India_train = India_data[:train_num//divider,-1] #1837
    India_val = India_data[train_num:train_num+val_num//divider,-1] #612
    India_test = India_data[train_num+val_num:train_num+val_num+test_num//divider,-1] #612

    US_train = US_data[:train_num//divider,-1] #1822
    US_val = US_data[train_num:train_num+val_num//divider,-1] #607
    US_test = US_data[train_num+val_num:train_num+val_num+test_num//divider,-1] #607

    train_label = np.concatenate((Australia_train,Canada_train,England_train,India_train,US_train),axis=0)
    val_label = np.concatenate((Australia_val,Canada_val,England_val,India_val,US_val),axis=0)
    test_label = np.concatenate((Australia_test,Canada_test,England_test,India_test,US_test),axis=0)

    return train_label, val_label, test_label