import pandas as pd
import numpy as np
from math import sqrt
from math import pi
from math import exp
from collections import Counter
import operator







def load_data():
    df= pd.read_excel("D:/Çağrı/d/2020pdf/Uzman Sistemler/divorce/divorce.xlsx")                    #pandas kütüphanesi kullanarak veri seti numpy array şeklinde programa aktarılır.
    data = df.to_numpy()
    return data

def separate_by_class(dataset,num_of_class):                            #veri seti classlarına göre listelere ayrılarak ileride gerçekleştirilecek işlemlere (std ve mean hesabı) hazırlık yapılır
    seperated = [[] for i in range(num_of_class)]
    for data in dataset: 
        for i in range(len(data)):
            vector = data[i]
            class_value = vector[-1]
            seperated[class_value].append(vector)
    return seperated

def k_fold(data, k=10):
    np.random.shuffle(data)
    df_folds = list()
    n_of_samples_per_fold = data.shape[0] // k
    fold = list()
    j=0
    for i in range(k):
        while(len(fold)<n_of_samples_per_fold):
            fold.append(data[j,:])
            j+=1
        df_folds.append(fold[:])
        fold.clear()
    return df_folds,n_of_samples_per_fold

def calculate_mean(seperated,num_of_class):             #her bir classtaki her bir özellik için ortalama hesabı gerçekleştirilir.
    mean_list = list()
    for classes in seperated:
        means = [0]*(len(classes[0])-1)
        counts = [0]*(len(classes[0])-1)
        for i in range(len(classes[0])-1):
            for data in classes:
                means[i] += data[i]
                counts[i] +=1
        means = [m / n for m, n in zip(means, counts)] 
        mean_list.append(means[:])
        means.clear()
        counts.clear()
    return mean_list

def calculate_std(seperated,mean_list):         #her bir classtaki her bir özellik için standart sapma hesaplanır.
    std_list = list()
    for j,classes in enumerate(seperated):
        stds=list()
        for i in range(len(classes[0])-1):
            std = 0
            for data in classes:
                std += (data[i]-mean_list[j][i])**2
            std /= (len(classes[0])-1)
            std = sqrt(std)
            stds.append(std)
        std_list.append(stds)
    return std_list



def run(data):
    num_of_class = 2            
    num_of_instances = len(data)            # her bir classın veri setinde bulunma olasılığını hesaplamak için tutulmaktadır.
    df_folds, n_of_samples_per_fold = k_fold(data)      # veri seti cross validation K=10 için hazır hale getirilir.
    class_value_list = list()
    prediction_list = list()
    mean_list = list()
    std_list = list()
    class_probability_list = list()
    for k,fold in enumerate(df_folds):
        train_set = list(df_folds)          #tum foldlar birleştirilip k. fold dışarıda bırakılarak train_set oluşturulur. k. fold test_set olarak kullanılacaktır.
        del(train_set[k])
        seperated = separate_by_class(train_set,num_of_class)   #training set classlara göre ayrılarak her bir class için ayrı std ve mean hesaplamaları yapılacaktır.
        for classes in seperated:
            probability = len(classes)/(num_of_instances- n_of_samples_per_fold)
            class_probability_list.append(probability)                  #her bir classın olasılığı saklanır.
        mean_list = calculate_mean(seperated,num_of_class)              #her bir classtaki her bir özelliğin ortalaması hesaplanır
        std_list = calculate_std(seperated,mean_list)                   #her bir classtaki her bir özelliğin std'si hesaplanır
        for test_data in fold:
            print(f"TEST DATA \n {test_data}")              #test_data bir fold'daki tek bir örneği gösterir. Örneğin son elemanı class bilgisidir. Accuracy hesabı için bir listede saklanır.
            class_value = int(test_data[-1])                #train_set de kullanılarak, örneğe en yakın komşular bulunur. Ardından bulunan komşuların bulunduğu neighbors listesi evaluation_data fonksiyonuna 
            print(f"Class Value:{class_value}\n")           #gönderilerek komşuların class bilgilerine göre oylama yapılır. Test_data'nın class'ına karar verilir.
            prediction = evaluate_data(test_data,mean_list,std_list,class_probability_list,num_of_class)
            print(f"Prediction :{prediction}\n")
            class_value_list.append(class_value)
            prediction_list.append(prediction)
    # print(f"Class Values: {class_value_list[:20]}")
    # print(f"Prediction:   {prediction_list[:20]}")
    accuracy = accuracy_metric(class_value_list,prediction_list)
    
    print(f"Accuracy: {accuracy}")


def calculate_probability(x, mean, stdev):                      #numeric verilerde naive bayes için olasılık hesabı gerçekleştirilir.
    if stdev == 0.0:
        stdev = 0.0001
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    
    value = (1 / (sqrt(2 * pi) * stdev)) * exponent

    return value
def evaluate_data(test_data,mean_list,std_list,class_probability_list,num_of_class):
    probability_list = list()
    for j in range(len(mean_list)):
        probability = 1
        for i in range(len(test_data)-1):
            probability *= calculate_probability(test_data[i],mean_list[j][i],std_list[j][i])       #test verisinin iki class için bulunma olasılıkları hesaplanır.
        probability = probability*class_probability_list[j]                     #naive bayes formül devamı
        probability_list.append(probability)
    probabilities = np.asarray(probability_list)
    index_class = np.argmax(probabilities)          #classlarda bulunma olasılıklarından olasılığı yüksek olan classın indisi alınır. indis bilgisi aynı zamanda class bilgisidir.
    return index_class                              

 

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

if __name__ == "__main__":
    data = load_data()
    run(data)
    