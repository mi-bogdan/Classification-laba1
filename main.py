from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import pandas as pd
import pickle


columns = ['№ Имени', '№ Хобби', 'Возраст',
           'Уровень образования', 'Семейное положение', 'Класс']
dataTraining = np.genfromtxt(
    'hayes-roth.data', dtype=None, delimiter=",", max_rows=100)
dataTest = np.genfromtxt('hayes-roth.data', dtype=None,
                         delimiter=",", skip_header=100)
datasetTraining = pd.DataFrame(dataTraining, columns=columns)
print('Обучающие данные')
print(datasetTraining)

datasetTest = pd.DataFrame(dataTest, columns=columns)
print('Тренировочные данные')
print(datasetTest)

treeTraining = DecisionTreeClassifier(
    max_depth=11, min_samples_split=0.001, min_samples_leaf=0.002)  # создаем экземпляр классификатора
X_train = np.array(dataTraining[:, 0:dataTraining.shape[1]-1])
Y_train = np.array(dataTraining[:, dataTraining.shape[1]-1])
X_test = np.array(dataTest[:, 0:dataTest.shape[1]-1])
Y_test = np.array(dataTest[:, dataTest.shape[1]-1])
treeTraining.fit(X_train, Y_train)  # обучение классификатора

pickle.dump(treeTraining, open('model', 'wb'))
loaded_model = pickle.load(open('model', 'rb'))
tree.plot_tree(loaded_model)  # вывод дерева

print(" \n Точность классификации итоговой модели на обучающей и тестовой выборках \n ")
print('точность тренировочной выборки:  ',
      treeTraining.score(X_train, Y_train))
print('ошибка классификации тренировочной выборки:  ',
      (1-treeTraining.score(X_train, Y_train)))
print('точность тестовой выборки: ',  treeTraining.score(X_test, Y_test))

Mas_Min_Samples_Split = [0.01, 0.007, 0.004]
Mas_Min_Samples_leaf = [0.1, 0.007, 0.01]

ii = 0
result = pd.DataFrame([], columns=["№", "max_depth", "min_samples_split",
                      "min_samples_leaf", "train accuracy", "test accuracy"])
for i in range(10, 15):
    for j in Mas_Min_Samples_Split:
        for k in Mas_Min_Samples_leaf:
            treeTraining = DecisionTreeClassifier(
                max_depth=i, min_samples_split=j, min_samples_leaf=k)
            treeTraining.fit(X_train, Y_train)
            result.loc[ii] = [ii, i, j, k, treeTraining.score(
                X_train, Y_train), treeTraining.score(X_test, Y_test)]
            ii += 1


print(" \nТаблица экспериментов подбора гиперпараметров \n ")
print(result.to_markdown())
