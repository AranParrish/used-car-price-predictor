# used-car-price-predictor
Machine Learning model to predict the price of a used car using a simple neural network built in PyTorch.

Source data downloaded from [kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) on 18 November 2025.

Note: Will need to mention that anyone forking the repo and wanting to make changes to the Docker CI image (e.g. because the Dockerfile or dependencies have been changed) will need to modify the workflows to point to their own image stored on GHCR. The repo is currently setup with my own image, which is publicly available for anyone else to use.