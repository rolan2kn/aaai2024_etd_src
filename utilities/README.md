This directory includes all helper classes. 

- ae_handler.py

The class AETrainerController manages all related with the autoencoders. The process is to generate data which is done using DataController class.
Then the AE class encapsulated the model logic it is configurable to define our desired architecture and contains train methods.
After that, the AELatentSpaceProcessor is utilized to process the latent space on each layer, generating images and persistence diagrams of each layer. 

- data_processors_helper.py

This file contains classes that encapsulates the dataset logic to homogenize the data access by experiment instances. This contains a class hierarchy 
including Shrec07Processor, OutexProcessor, Cifar10Processor, FashionMNistProcessor. All of these classes contains a common interface making easy to operate with them.
These classes interacts with data readers to collect and preprocess the dataset, except for Schrec07 that it maintains the collection of filenames that will be processed by the TDAHelper.
The class DataProcessorFactory is responsible of creating the desired data processor.  

- directory_helper.py

It contains DirectoryHelper a class that manipulates file directories with well defined operations that support many other functionalities.

- distance_matrix_helper.py

DistanceMatrixHelper is a controller class in charge of handling the logic of distance computations. This class is a high level interface that 
connects experiments or other high level helpers such as KNNClassifierHelper or TopologicalInformationProvider with the distance computations executors.
Each of these distance executors has a representative static function on DistanceMatrixHelper, for example all these functions computed by TDA libraries such as persim, gudhi, etc. An our proposed ETD whose executor are contained inside
extended_topology_distance.py file. Basically, we ask this class to compute a distance matrix then the apropriate executor uis called.
We deacoplate the logic of computing the distance matrix to be a separated helper contained inside tt_distance_matrix_executor because were+ interested into explore multiple ways of computing the distance matrix even when the actual distance function is contained inside DistanceMatrixHelper.  

- extended_topology_distance.py

This file is the most important of the entire project since it contains the implementation of our proposed distances.
Inside this file we have ExtendedTopologyDistance and ExtendedTopologyDistance classes, the first one is the naive implementation of the proposed ETD and the last one is the version that leverages on numpy array programming that achieves speed ups of several orders of magnitude in comparison with the naive versions.

- information_visualization_helpers.py

This unit is dedicated to draw the topological curves on the Autoencoder application. 

- knn_classifier_helper.py

Given an experiment instance, it computes the knn classifier. It uses the experiment to access to the data procesor and also to compute the specified distance matrix.
This KNN implementation is very naive. We have a flag that determines if we run cross validation or only one shot across multiple k values.

- laplace.py

This contains helper functions to compute the Heat Kernel Signature

- mesh_signature_helper.py

It computes a the Heat Kernel Signature on each point of the mesh. We decide do not subsampling the vertex set, but definitely could reduce the computational load on computing HKS.

- outer_reader.py

It loads the Outex dataset.

- persistence_diagram_helper.py

It contains a collection of functions dedicated to make persistence diagrams easy to process. For example, truncate diagrams that cut the infinite persistence intervals at the maximal filtration value.
homogenize, sanitize, get_maximal_dimension among others dedicated to prepare PDs for distance computations. 
 
- tda_helper.py

This is were all simplicial complexes and corresponding persistence homology is computed. We support mainly, Cubical complex, alpha complex and Rips Complex.  

- topological_information_provider.py

In this unit, the persistence diagrams of our autoencoder application is processed. Basically it iterates per layer computing the topological distances with two different policies
ALL_VS_First, ALL_VS_Previous, in the paper only results ALL_VS_First were shown.

- torch_datasets_reader.py

Here we load torch datasets such as Fashion MNist. We also provide a ImageDataPreprocessor where suggested image transformation are performed to produce a relevant cubical filtration.

- tt_distance_matrix_executor.py

This class computes the distance matrix utilizing the distance functions defined on DistanceMatrixHelper. Considering the high computational resources required for most of the distances that we use to compare our proposed ETD,
we consider to create a matrix using test samples by row and train samples by colum. In such a way that M[i][j] means distance between ith test sample and jth train sample.
