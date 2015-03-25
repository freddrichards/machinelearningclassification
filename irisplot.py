
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
labels = iris.target_names
#Symbols to represent the points for the three classes on the graph.
gMarkers = ["+", "_", "x"]
#Colours to represent the points for the three classes on the graph
gColours = ["blue", "magenta", "cyan"]
#The index of the class in target_names
gIndices = [0, 1, 2]
#Column indices for the two features you want to plot against each other:
f1 = 1
f2 = 3

for mark, col, i, iris.target_name in zip(gMarkers, gColours, gIndices, labels):
   plt.scatter(x = X[iris.target == i, f1], 
y = X[iris.target == i, f2], 
marker = mark, c = col, label=iris.target_name)
plt.legend(loc='upper right')
xlabel = iris.feature_names[f1]
ylabel = iris.feature_names[f2]

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()

