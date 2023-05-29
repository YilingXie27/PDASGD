The code is attached to the paper ``Solving a Special Type of Optimal Transport Problem by a Modified Hungarian Algorithm'', Yiling Xie, Yiling Lou, Xiaoming Huo, Transactions on machine learning research.


algo.py: includes the implementation of the modified Hungarian algorithm, the Hungarian algorithm, the Sinkhorn algorithm for comparison

syn.py: numerical experiments on synthetic data, comparing with Hungarian algorithm
cifar.py: numerical experiments on CIFAR10 dataset, comparing with Hungarian algorithm
breast.py: numerical experiments on Wisconsin breast cancer dataset, comparing with Hungarian algorithm
dot.py: numerical experiments on DOT-benchmark dataset, comparing with Hungarian algorithm

synsink.py: numerical experiments on synthetic data, comparing with Sinkhorn algorithm
cifarsink.py: numerical experiments on CIFAR10 dataset, comparing with Sinkhorn  algorithm
breastsink.py: numerical experiments on Wisconsin breast cancer dataset, comparing with Sinkhorn algorithm
dotsink.py: numerical experiments on DOT-benchmark dataset, comparing with Sinkhorn algorithm

synnet.py: numerical experiments on synthetic data, comparing with network simplex algorithm
cifarnet.py: numerical experiments on CIFAR10 dataset, comparing with network simplex  algorithm
breastnet.py: numerical experiments on Wisconsin breast cancer dataset, comparing with network simplex algorithm
dotnet.py: numerical experiments on DOT-benchmark dataset, comparing with network simplex algorithm

To run the code, one should download wdbc.data (Wisconsin breast cancer dataset) from https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/, and download DOTmark_1.0 (DOT benchmark dataset) from https://figshare.com/articles/dataset/DOTmark_v1_0/4288466 

The code for the Sinkhorn algorithm is modified from https://github.com/PythonOT/POT.

