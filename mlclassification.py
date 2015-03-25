import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import knnplots

from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

fileName = "wdbc.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)
# print dataList
#
dataArray = numpy.array(dataList)
#
X = dataArray[:,2:32].astype(float)
y = dataArray[:,1]
# # print "X dimensions: ", X.shape
# # print "y dimensions", y.shape
#
# # yFreq = scipy.stats.itemfreq(y)
M = sum(y=='M')
B = sum(y=='B')

#
# # plt.bar(left = 0, height = int(M))
# # plt.bar(left = 1, height = int(B))
# # plt.show()
#
le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)
#
# print y[0:2]
# print yTransformed[0:2]
# print y[18:20]
# print yTransformed[18:20]
#
# correlationMatrix = numpy.corrcoef(X, rowvar=0)
# # fig, ax = plt.subplots()
# # heatmap = ax.pcolor(correlationMatrix, cmap = plt.cm.Blues)
# # plt.show()
#
# # plt.scatter(x = X[:,0], y = X[:,1])
# # plt.show()
#
# #Scatter plot for first two features:
# # plt.scatter(x = X[:,0], y = X[:,1], c=y) #N.B. ys in this case refers to different things
# # plt.show()
#
# #Bonus
# def scatter_plot(X,y):
#     plt.figure(figsize = (2*X.shape[1],2*X.shape[1]))
#     for i in range(X.shape[1]):
#         for j in range(X.shape[1]):
#             plt.subplot(X.shape[1],X.shape[1],i+1+j*X.shape[1])
#             if i == j:
#                 plt.hist(X[:,i][y=='M'], alpha = 0.4, color = 'm',
#                 bins = numpy.linspace(min(X[:,i]),max(X[:,i]),30))
#                 plt.hist(X[:,i][y=='B'], alpha = 0.4, color = 'b',
#                 bins = numpy.linspace(min(X[:,i]),max(X[:,i]),30))
#                 plt.xlabel(i)
#             else:
#                 plt.gca().scatter(X[:,i],X[:,j],c=y, alpha=0.4)
#                 plt.xlabel(i)
#                 plt.ylabel(j)
#             plt.gca().set_xticks([])
#             plt.gca().set_yticks([])
#     plt.tight_layout()
#     plt.show()


# scatter_plot(X[:,:2],y)

nbrs = neighbors.NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

# print indices[:5]
# print distances[:5,]

#Classification based on 3 nearest neighbours
knnK3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knnK3 = knnK3.fit(X,yTransformed)
predictedK3 = knnK3.predict(X)

#Classification based on 15 nearest neighbours
knnK15 = neighbors.KNeighborsClassifier(n_neighbors=15)
knnK15 = knnK15.fit(X,yTransformed)
predictedK15 = knnK15.predict(X)

knnK100 = neighbors.KNeighborsClassifier(n_neighbors=100)
knnK100 = knnK100.fit(X,yTransformed)
predictedK100 = knnK100.predict(X)


nonAgreement = predictedK3[predictedK3 != predictedK15]
# nonAgreement = predictedK3
# print 'Number of discrepancies', len(nonAgreement)

knnWD = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
knnWD = knnWD.fit(X, yTransformed)
predictedWD = knnWD.predict(X)

#Compare the classifications with the classifications when the weights are uniform as you did above for the models learned with different k values.

knnWD = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
knnWD = knnWD.fit(X, yTransformed)
predictedWD = knnWD.predict(X)
knnWU = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
knnWU = knnWU.fit(X, yTransformed)
predictedWU = knnWU.predict(X)
nonAgreement = predictedWD[predictedWD != predictedWU]
# print 'Number of discrepancies', len(nonAgreement)



# # plt.gca().scatter(X[:,0],predictedWD[:,0],c=y, alpha=0.4)
# plt.hist(X[y=='nonAgreement'], color = 'm')
# # plt.xlabel(i)
# # plt.ylabel(j)
# # plt.gca().set_xticks([])
# # plt.gca().set_yticks([])
# plt.show()

#Split into training and test data:
XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

# print "XTrain dimensions: ", XTrain.shape
# print "yTrain dimensions: ", yTrain.shape
# print "XTest dimensions: ", XTest.shape
# print "yTest dimensions: ", yTest.shape

a = float((92.0+41.0)/143.0)
p = float(92.0/(92.0+9.0))
r = float(92.0/93.0)
s = float(41.0/(41.0+9.0))
# print "Accuracy: ", a
# print "Precision: ", p
# print "Recall: ", r
# print "Specificity: ", s

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(XTrain, yTrain)
predictedk = knn.predict(XTest)

# mat = metrics.confusion_matrix(yTest, predictedk)
# print "KKN"
# print mat
#
# print metrics.classification_report(yTest, predictedk)
# print "accuracy: ", metrics.accuracy_score(yTest, predictedk)
#
# knnplots.plotaccuracy(XTrain, yTrain, XTest, yTest, 310)
#
# knnplots.decisionplot(XTrain, yTrain, n_neighbors=3, weights="uniform")
# knnplots.decisionplot(XTrain, yTrain, n_neighbors=15, weights="uniform")

nbmodel = GaussianNB().fit(XTrain, yTrain)
predictedb = nbmodel.predict(XTest)
#
mat = metrics.confusion_matrix(yTest, predictedb)
# print "Bayes"
# print mat
# print metrics.classification_report(yTest, predictedb)
# print "accuracy: ", metrics.accuracy_score(yTest, predictedb)

#Module 6:
knn3scores = cross_validation.cross_val_score(knnK3, XTrain, yTrain, cv = 5)
print knn3scores
print "Mean of scores KNN3", knn3scores.mean()
print "SD of scores KNN3", knn3scores.std()

knn15scores = cross_validation.cross_val_score(knnK15, XTrain, yTrain, cv = 5)
print knn15scores
print "Mean of scores KNN15", knn15scores.mean()
print "SD of scores KNN15", knn15scores.std()

nbscores = cross_validation.cross_val_score(nbmodel, XTrain, yTrain, cv = 5)
print nbscores
print "Mean of scores NB", nbscores.mean()
print "SD of scores NB", nbscores.std()

#28
meansKNNK3 = []
sdsKNNK3 = []
meansKNNK15 = []
sdsKNNK15 = []
meansNB = []
sdsNB = []

ks = range(2, 21)
for k in ks:
  knn3scores = cross_validation.cross_val_score(knnK3, XTrain,
                                                yTrain, cv=k)
  knn15scores = cross_validation.cross_val_score(knnK15, XTrain,
                                                yTrain, cv=k)
  nbscores = cross_validation.cross_val_score(nbmodel, XTrain,
                                                yTrain, cv=k)
  meansKNNK3.append(knn3scores.mean())
  sdsKNNK3.append(knn3scores.std())
  meansKNNK15.append(knn15scores.mean())
  sdsKNNK15.append(knn15scores.std())
  meansNB.append(nbscores.mean())
  sdsNB.append(nbscores.std())

plt.plot(ks, meansKNNK3, label="KNN 3 mean accuracy", color="purple")
plt.plot(ks, meansKNNK15, label="KNN 15 mean accuracy", color="yellow")
plt.plot(ks, meansNB, label="NB mean accuracy", color="blue")
plt.legend(loc=3)
plt.ylim(0.5, 1)
plt.title("Accuracy means with Increasing K")
plt.show()


plt.plot(ks, meansKNNK3, label="KNN 3 mean accuracy", color="purple")
plt.plot(ks, meansKNNK15, label="KNN 15 mean accuracy", color="yellow")
plt.plot(ks, meansNB, label="NB mean accuracy", color="blue")
plt.legend(loc=3)
plt.ylim(0.91, 0.95)
plt.title("Accuracy means with Increasing K")
plt.show()

# plt.plot(ks, sdsKNNK3, label="KNN 3 sd accuracy", color="purple")
# plt.plot(ks, sdsKNNK15, label="KNN 15 sd accuracy",color="yellow")
# plt.plot(ks, sdsNB, label="NB sd accuracy", color="blue")
# plt.legend(loc=3)
# plt.ylim(0, 0.1)
# plt.title("Accuracy standard deviations with Increasing K")
# plt.show()


#29
parameters = [{'n_neighbors':[1,3,5,10,50,100],
            'weights':['uniform','distance']}]
clf = GridSearchCV(neighbors.KNeighborsClassifier(), parameters, cv=10, scoring="f1")
clf.fit(XTrain, yTrain)

neighbors.KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',
metric_params=None,n_neighbors=3,p=2,weights='uniform')

print "Best parameter set found: "
print clf.best_estimator_

print "Grid scores:"
for params, mean_score, scores in clf.grid_scores_:
  print "%0.5f (+/-%0.03f) for %r" % (mean_score, scores.std()/2, params)