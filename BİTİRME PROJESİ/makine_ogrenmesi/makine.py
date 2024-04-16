import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv("heart.csv")
data.head()

kalp_hastalari = data[data.output == 1]
saglikli_insanlar = data[data.output == 0]

# x ve y eksenlerini belirleyelim
y = data.output.values
x_ham_veri = data.drop(["output"],axis=1)   
# Outcome sütununu(dependent variable) çıkarıp sadece independent variables bırakıyoruz
# Çüknü KNN algoritması x değerleri içerisinde gruplandırma yapacak..


# normalization yapıyoruz - x_ham_veri içerisindeki değerleri sadece 0 ve 1 arasında olacak şekilde hepsini güncelliyoruz
# Eğer bu şekilde normalization yapmazsak yüksek rakamlar küçük rakamları ezer ve KNN algoritmasını yanıltabilir!
x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))

# k kaç olmalı ?
# en iyi k değerini belirleyelim..
sayac = 1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors = k)
    knn_yeni.fit(x_train,y_train)
    print(sayac, "  ", "Doğruluk oranı: %", knn_yeni.score(x_test,y_test)*100)
    sayac += 1
    
    
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=1)


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K=5 için Test verilerimizin doğrulama testi sonucu ", knn.score(x_test, y_test))


#Veri seti İçinde Olmayan Yeni bir Hastanın Tahlil Sonuçlarına Göre Prediction

# Yeni bir hasta tahmini için:
from sklearn.preprocessing import MinMaxScaler
 
# normalization yapıyoruz - daha hızlı normalization yapabilmek için MinMax  scaler kullandık...
sc = MinMaxScaler()
sc.fit_transform(x_ham_veri)
 
new_prediction = knn.predict(sc.transform(np.array([[20,1,2,130,233,0,0,172,0,2.6,1,0,1]])))
new_prediction[0]

print("Yeni hastanın tahmini:", new_prediction[0])
