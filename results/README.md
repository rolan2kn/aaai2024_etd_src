In this directory appears all results. We generate an application folder which contains the corresponding experiments.

### AutoencoderWeightTopologyApp
- AE_lrelu_data
  - AEExperimentALLVersusFirst: Experiment name, an autoencoder experiment with lrelu and versus vs first policy 
    - pd_distance_matrices: the computed distance matrices and generated result images.  
  - topological_info: the corresponding persistence diagrams to each layer using the model and this activation function.
- AE_relu_data
  - AEExperimentALLVersusFirst
    - pd_distance_matrices 
  - topological_info
- AE_tanh_data
  - AEExperimentALLVersusFirst
      - pd_distance_matrices
  - topological_info

### SupervisedLearningApp

- OutexProcessor
  - OutexExperiment
      - pd_distance_matrices
  - topological_info
- Shrec07Processor
  - Shrec07Experiment
      - pd_distance_matrices
  - topological_info
- FashionMNistProcessor
  - HOG: the preprocessing applied
    - OutexExperiment: experiment name
        - pd_distance_matrices: computed distances
    - topological_info: persistence diagrams computed with this data transformation (HOG)
