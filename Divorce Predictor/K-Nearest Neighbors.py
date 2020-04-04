import pandas as pd
import numpy as np
from math import sqrt
from collections import Counter
import operator

def load_data():
    df= pd.read_excel("D:/Çağrı/d/2020pdf/Uzman Sistemler/divorce/divorce.xlsx")
    data = df.to_numpy()
    return data


def k_fold(data, k=10):                 # cross validation için veri seti fold'lara ayrılır.
    np.random.shuffle(data)             #tüm veri seti karıştırılarak her bir fold'a random classlardan örnekler gelmesi sağlanır
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
    return df_folds

def euclidian_distance(sample1,sample2):                    
    distance = 0.0
    for i in range(len(sample1)-1):
        distance += (sample1[i]-sample2[i])**2
    #print(sqrt(distance))
    return sqrt(distance)

def find_neighbors(dataset,test_data,k=3):
    distance_list = list()
    neighbors = list()
    for data in dataset:
        for sample in data:
            distance = euclidian_distance(sample,test_data)                     # train_set'teki örnekler ve test_data arasında öklid uzaklık hesabı yapılır. ardından uzaklıklar listesin küçükten büyüğe
            distance_list.append((sample,distance))                             #sıralanarak en yakın K adet veri komşu olarak işaretlenir.
    distance_list.sort(key=lambda tup: tup[1])
    for i in range(k):
        neighbors.append(distance_list[i][0])
    return neighbors

def evaluate_data(neighbors):
    #print(neighbors)
    output = dict((str(val[-1]), 0) for val in neighbors)               #oylama için output isimli dict oluşturulmuştur. Dict'teki key sayısı class sayısı kadardır ve komşuların class bilgileri kullanılarak
    #print(output)                                                      #hangi classtan daha fazla verinin test verisine komşu olduğu tespit edilir. Ardından class bilgisi döndürülür.
    for neighbor in neighbors:
        output[str(neighbor[-1])] +=1
    
    return float(max(output.items(), key=operator.itemgetter(1))[0])

def accuracy_metric(actual, predicted):                                 #başarı hesabının gerçekleştirildiği fonksiyondur. Validation sırasında kontrol edilen tüm class ve prediction değerleri saklanarak 
	correct = 0                                                         # başarı yüzdesi hesaplanır.
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def run(data):
    df_folds = k_fold(data)         
    class_value_list = list()
    prediction_list = list()
    for k,fold in enumerate(df_folds):          
        #print(f" {k}.fold dısarida")
        train_set = list(df_folds)          #tum foldlar birleştirilip k. fold dışarıda bırakılarak train_set oluşturulur. k. fold test_set olarak kullanılacaktır.
        del(train_set[k])
        for test_data in fold:
            class_value = float(test_data[-1])              #test_data bir fold'daki tek bir örneği gösterir. Örneğin son elemanı class bilgisidir. Accuracy hesabı için bir listede saklanır.
            neighbors = find_neighbors(train_set,test_data) #train_set de kullanılarak, örneğe en yakın komşular bulunur. Ardından bulunan komşuların bulunduğu neighbors listesi evaluation_data fonksiyonuna 
            prediction = evaluate_data(neighbors)           #gönderilerek komşuların class bilgilerine göre oylama yapılır. Test_data'nın class'ına karar verilir.
            class_value_list.append(class_value)
            prediction_list.append(prediction)
            print(f"TEST DATA \n {test_data}")          
            print(f"Class Value:{class_value}\n")
            print(f"Prediction :{prediction}\n")
    # print(f"Class Values: {class_value_list[:20]}")
    # print(f"Prediction:   {prediction_list[:20]}")
    accuracy = accuracy_metric(class_value_list,prediction_list)
    print(f"Accuracy: {accuracy}")                      #tüm foldlarda aynı işlem tekrar edilir. Ardından classification başarısı hesaplanır.
    

def dataset_minmax(dataset):                    #veri setindeki her bir sutunun min-max değerleri bulunarak normalizasyon işlemi için saklanacaktır.
    minmax = list()
    col_values = list()
    lendata = len(dataset[0])
    for i in range(len(dataset[0])):
        for row in dataset:
            col_values.append(row[i])
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset, minmax):                     # uzaklık hesabı yapılırken örnek içerisindeki numerik değerlerin hesabı etkilememesi amacıyla normalizasyon işlemi gerçekleştirilmektedir. 
    for row in dataset:                                         
        for i in range(1,len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset

if __name__ == "__main__":
    data = load_data()
    #minmax = dataset_minmax(data)
    #data = normalize_dataset(data,minmax)                   
    run(data)




    
    






