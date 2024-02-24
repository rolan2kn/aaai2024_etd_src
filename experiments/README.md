Experiments are controller classes that manipulates persistence diagrams obtained from data processors, and apply a given processing to them.
For example supervised learning experiments perform a classification task on a given dataset using knn. so They make the connection between the desired diagrams and the desired task executor. 

- base_experiment
- sl_experiment: hierarchy of supervised learning experiments. This experiments receive a data processor to capture the targets sets and other informations then execute a KNN classifierhelper.  
  - outex_experiment
  - shrec07_experiment
  - fashion_experiment
  - cifar10_experiment
- ae_experiment: autoencoder experiments it operates in all activation functions at once utilizing a topological information provider.

