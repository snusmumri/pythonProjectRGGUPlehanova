import numpy as np    # Импортируем библиотеку нампай
import pandas as pd   # Импортируем библиотеку пандас
import matplotlib.pyplot as plt   # Импортируем библиотеку матплотлиб
import statsmodels.api as sm # Импортируем библиотеку статсмодел
import sklearn               # Импортируем библиотеку склерн
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/My Drive/Churn_Modelling.csv')  # Выводим исходные данные
df

df.dtypes  # выводим типы данных

df.isna().sum() # Проверяем наличие пропущенных значений

df.duplicated().sum() # Проверяем наличие дубликатов

df.columns[df.dtypes == 'object'] # Выводим названия колонк типа Объект

colm = df.columns  # находим столбцы с полностью одинаковыми значениями
for i in colm:
  if len(df[i].unique()) == 1:
    print(i)

df = df.drop(['RowNumber',  'CustomerId', 'Surname'], axis = 1) # удаляем не нужные для анализа столбцы
df

svodka = df.describe()
svodka.to_excel('/content/drive/My Drive/svodka.xlsx')  # Выводим сводку по датасету и сохраняем ее в файл Exel для дальнейшего использования в отчете
svodka

df.drop(['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'Exited', 'Geography', 'Gender'], axis = 1).boxplot(figsize = (12, 8))
       # Выводим диаграмму "Ящик с усами" для столбцов Balance и EstimatedSalary

df.drop(['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited',
        'Exited', 'Geography', 'Gender'], axis = 1).boxplot(figsize = (12, 8))
        # Выводим диаграмму "Ящик с усами" для столбца CreditScore

df.drop(['Balance', 'EstimatedSalary', 'CreditScore', 'Age'], axis = 1).boxplot(figsize = (12, 8))
# Выводим диаграмму "Ящик с усами" для столбцов Tenure, NumOfProducts, HasCrCard, IsActiveMember, Exited

df.drop(['Balance', 'EstimatedSalary', 'CreditScore', 'Tenure', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'Exited', 'Geography', 'Gender'], axis = 1).boxplot(figsize = (12, 8))
        # Выводим диаграмму "Ящик с усами" для столбца Age

df['Age'].max()  # Находим максимальный возраст клиента

df = pd.get_dummies(df, drop_first = False) # Преобразуем категориальные переменные

df = df.drop('Gender_Male', axis = 1) # Удаляем столбец Gender_Male
df

df.columns # Выводим названия столбцов

df = df.reindex(columns = ['Exited', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',  # Переносим столбец 'Exited'
       'IsActiveMember', 'EstimatedSalary', 'Geography_France',
       'Geography_Germany', 'Geography_Spain', 'Gender_Female'])

train = df

train.corr().style.background_gradient(cmap = 'coolwarm') #строим кореляционную матрицу

r = 0.5  # Находим столбцы с высокой корреляцией
col = []
cor = train.corr()
for k, i in enumerate(train.columns[0:-1]):
  for j in train.columns[k + 1:]:
    if np.abs(cor[i][j]) > r:
      if np.abs(cor['Exited'][j]) > np.abs(cor['Exited'][i]):
        col.append(i)
      else:
        col.append(j)
print(col)

train = train.drop(list(set(col)), axis = 1) # Удаляем стобцы с высокой корреляцией

train.corr().style.background_gradient(cmap = 'coolwarm')  #Строим новую кореляционную матрицу

corr_matrix = train.corr().style.background_gradient(cmap = 'coolwarm')  # Сохраняем корреляционную матрицу на диск
corr_matrix.to_excel('/content/drive/My Drive/corr_matrix.xlsx')

train_x, test_x, train_y, test_y = train_test_split(train.drop(['Exited'], axis = 1), train['Exited'], test_size = 0.2) # Производим разбиение данных
model = sm.Logit(train_y, sm.add_constant(train_x)).fit()  # Строим модель логистической регрессии
print(model.summary())

# Создаем цикл для удаления из модели факторов, имеющих низкую значимость
pvalue = 0.05
train1_x = train_x.copy()
test1_x = test_x.copy()
train1_y = train_y.copy()
test1_y = test_y.copy()

while np.max(model.pvalues[1:]) > pvalue:
  train1_x = train1_x.drop(train1_x.columns[np.argmax(model.pvalues[1:])], axis = 1)
  test1_x = test1_x.drop(test1_x.columns[np.argmax(model.pvalues[1:])], axis = 1)
  model = sm.Logit(train1_y, sm.add_constant(train1_x)).fit()
print(model.summary())

pred_logit = model.predict(sm.add_constant(test1_x)) # Выводим предсказания по модели

from sklearn.metrics import roc_auc_score, roc_curve  # импортируем функции для расчета РОК-кривой и площади под кривой
logit_roc_auc = roc_auc_score(test1_y, pred_logit)  # рассчитываем площадь под кривой
fpr, tpr, thresholds = roc_curve(test1_y, pred_logit)  # рассчитываем fpr и tpr для построения РОК-кривой
plt.plot(fpr, tpr, label = 'Logistic regression (AUC = %0.4f)' %logit_roc_auc) # строим РОК-кривую
plt.plot([0, 1], [0, 1], linestyle = '--') # строим диагональную линию для наглядности (это самый худший вариант для эффективности модели)
plt.legend() # выводим легенду на график
from sklearn import tree
model2 = tree.DecisionTreeClassifier(max_depth = 8, min_samples_split = 150, max_leaf_nodes = 40).fit(train_x, train_y)  # Строим модель дерева решений
pred_tree = model2.predict_proba(test_x)
plt.figure(figsize = (20, 20))
tree.plot_tree(model2)

tree_roc_auc = roc_auc_score(test_y, pred_tree[:, 1]) # рассчитываем площадь под кривой
fpr1, tpr1, thresholds1 = roc_curve(test_y, pred_tree[:, 1]) # рассчитываем fpr и tpr для построения РОК-кривой
plt.plot(fpr, tpr, label = 'Logistic regression (AUC = %0.4f)' %logit_roc_auc) # строим РОК-кривую
plt.plot(fpr1, tpr1, label = 'Decision tree (AUC = %0.4f)' %tree_roc_auc) # строим РОК-кривую
plt.plot([0, 1], [0, 1], linestyle = '--') # строим диагональную линию для наглядности (это самый худший вариант для эффективности модели)
plt.legend() # выводим легенду на график

from sklearn import ensemble
model3 = sklearn.ensemble.RandomForestClassifier(max_features = 4, max_depth = 10, n_estimators = 800).fit(train_x, train_y)  # Строим модель Случайного леса
pred_rf = model3.predict_proba(test_x)[:, 1]

plt.plot(model3.feature_importances_)  # Построим график значимости признаков
len(model3.feature_importances_)

# Создадим цикл для удаления не значимых признаков
lst_index_for_del = []
lst =  model3.feature_importances_.tolist()
value = 0.01
for i in lst:
  if i > value:
   lst_index_for_del.append(lst.index(i))

print(lst_index_for_del)
train_x_2 = train_x[train_x.columns[lst_index_for_del]]
test_x_2 = test_x[test_x.columns[lst_index_for_del]]

model3_2 = sklearn.ensemble.RandomForestClassifier().fit(train_x_2, train_y)  # Перестроим модель Случайного леса без учета не значимых признаков
pred_rf_2 = model3_2.predict_proba(test_x_2)[:, 1]

plt.plot(model3_2.feature_importances_)  # Построим график значимости признаков без учета не значимых признаков
len(model3_2.feature_importances_)

rf_roc_auc = roc_auc_score(test_y, pred_rf)  # рассчитываем площадь под кривой
fpr2, tpr2, thresholds2 = roc_curve(test_y, pred_rf) # рассчитываем fpr и tpr для построения РОК-кривой
plt.plot(fpr, tpr, label = 'Logistic regression (AUC = %0.4f)' %logit_roc_auc) # строим РОК-кривую
plt.plot(fpr1, tpr1, label = 'Decision tree (AUC = %0.4f)' %tree_roc_auc)
plt.plot(fpr2, tpr2, label = 'Random forest (AUC = %0.4f)' %rf_roc_auc)
plt.plot([0, 1], [0, 1], linestyle = '--') # строим диагональную линию для наглядности (это самый худший вариант для эффективности модели)
plt.legend() # выводим легенду на график

# Построим ROC-кривую, однако как видно из графика, модель с отобранными по значимости факторами показала результат хуже первоначальной модели
# Поэтому для дальнейшей работы оставим первоначальный вариант.
rf_roc_auc = roc_auc_score(test_y, pred_rf_2)
fpr2, tpr2, thresholds2 = roc_curve(test_y, pred_rf_2) # рассчитываем fpr и tpr для построения РОК-кривой
plt.plot(fpr, tpr, label = 'Logistic regression (AUC = %0.4f)' %logit_roc_auc) # строим РОК-кривую
plt.plot(fpr1, tpr1, label = 'Decision tree (AUC = %0.4f)' %tree_roc_auc)
plt.plot(fpr2, tpr2, label = 'Random forest (AUC = %0.4f)' %rf_roc_auc)
plt.plot([0, 1], [0, 1], linestyle = '--') # строим диагональную линию для наглядности (это самый худший вариант для эффективности модели)
plt.legend() # выводим легенду на график

gb = sklearn.ensemble.GradientBoostingClassifier(max_features = 3, n_estimators = 500, learning_rate = 0.05, max_depth = 4).fit(train_x, train_y)  # Строим модель градиентного бустинга
pred_gb = gb.predict_proba(test_x)[:, 1]

gb_roc_auc = roc_auc_score(test_y, pred_gb)  # рассчитываем площадь под кривой
fpr3, tpr3, thresholds3 = roc_curve(test_y, pred_gb) # рассчитываем fpr и tpr для построения РОК-кривой
plt.plot(fpr, tpr, label = 'Logistic regression (AUC = %0.4f)' %logit_roc_auc) # строим РОК-кривую
plt.plot(fpr1, tpr1, label = 'Decision tree (AUC = %0.4f)' %tree_roc_auc)
plt.plot(fpr2, tpr2, label = 'Random forest (AUC = %0.4f)' %rf_roc_auc)
plt.plot(fpr3, tpr3, label = 'Gradient boosting (AUC = %0.4f)' %gb_roc_auc)
plt.plot([0, 1], [0, 1], linestyle = '--') # строим диагональную линию для наглядности (это самый худший вариант для эффективности модели)
plt.legend() # выводим легенду на график

from sklearn.naive_bayes import GaussianNB # Импортируем GaussianNB
model4 = GaussianNB().fit(train_x, train_y) # Строим модель наивного байсовского метода
pred_nb = model4.predict_proba(test_x)[:, 1]

nb_roc_auc = roc_auc_score(test_y, pred_nb)  # рассчитываем площадь под кривой
fpr4, tpr4, thresholds4 = roc_curve(test_y, pred_nb) # рассчитываем fpr и tpr для построения РОК-кривой
plt.plot(fpr, tpr, label = 'Logistic regression (AUC = %0.4f)' %logit_roc_auc) # строим РОК-кривую
plt.plot(fpr1, tpr1, label = 'Decision tree (AUC = %0.4f)' %tree_roc_auc)
plt.plot(fpr2, tpr2, label = 'Random forest (AUC = %0.4f)' %rf_roc_auc)
plt.plot(fpr3, tpr3, label = 'Gradient boosting (AUC = %0.4f)' %gb_roc_auc)
plt.plot(fpr4, tpr4, label = 'Naive Bayes (AUC = %0.4f)' %nb_roc_auc)
plt.plot([0, 1], [0, 1], linestyle = '--') # строим диагональную линию для наглядности (это самый худший вариант для эффективности модели)
plt.legend() # выводим легенду на график

results = pd.DataFrame()  # Выводим данные по предсказаниям
results['y'] = test_y
results['Logist_Regression'] = pred_logit
results['Decision_Tree'] = pred_tree[:, 1]
results['Random_Forest'] = pred_rf
results['Gradient_Boosting'] = pred_gb
results['Naive_Bayes'] = pred_nb
results.to_excel('/content/drive/My Drive/results.xlsx') # Записываем полученные данные в файл ексель
results

# Строим диаграмму точности результатов моделирования
procent_tosnost = [logit_roc_auc, tree_roc_auc, rf_roc_auc, gb_roc_auc, nb_roc_auc]
procent_tosnost = list(np.around(procent_tosnost, decimals = 4))
metod_analiza = ['Логистическая регрессия', 'Дерево решений', 'Метод случайного леса', 'Градиентный бустинг', 'Наивный байесовский метод']
plt.figure(figsize = (9, 7))
plt.bar(metod_analiza, procent_tosnost)
plt.title('Точность результатов моделирования', color = 'blue', fontstyle = 'italic', family = 'arial', weight = 'bold', size = 'x-large', fontsize = 20)
plt.xlabel('Метод', weight = 'bold', color = 'blue', fontsize = 15)
plt.ylabel('Процент точности', weight = 'bold', color = 'blue', fontsize = 15)
plt.xticks(rotation = 20)
plt.show()
