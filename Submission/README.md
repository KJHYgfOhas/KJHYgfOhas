This repository presents the main results of the paper: "Cyclic orthogonal convolutions for long-range integration of features".

Please run the Jupyter notebook: OrthogonalConvReviewed.ipynb

The following docker image is suitable to run the nontebook on GPU: 
'''sudo docker run --runtime=nvidia --rm --name cyclic_orthogonal -it -v ~/:/tf -p 8888:8888 tensorflow/tensorflow:1.15.0-gpu-py3-jupyter bash'''
