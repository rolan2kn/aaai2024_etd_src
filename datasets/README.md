In this directory are stored the datasets. Shrec and Outex are stored as is. In contrast, FashionMNist is generates while downloaded from pytorch.datasets. Then a copy is also stored when the transformation is applied generating a new dataset wit hthe following format fashion_{test|train}_{GAUSSIAN_BLUR|HOG|IGG}.npy.pickle in this case only HOG appears.