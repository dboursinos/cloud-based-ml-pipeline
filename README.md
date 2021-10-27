# Cloud-Based Machine Learning Pipeline

This repository contains a toolchain for training and deployment of machine learning models with all resources distributed on the cloud. One can use this toolchain with cloud services such as AWS and Azure or self-hosted on any server.

The training process of machine learning models usually require many iterations of trial-and-error in order to find the best set of model architecture and hyperparameters that optimize specific metrics. Given the possible long times for each training iteration, it is critical to keep track not only of the models and hyperparameters used in each iteration but also different evaluation metrics that can contribute on the decision-making regarding the model deployment. Easy access to these metrics through cloud services is vital for easy collaboration and efficient development. 
![](https://github.com/dboursinos/cloud-based-ml-pipeline-/blob/main/Images/pipeline.svg) 
## Features
- Machine Learning pipeline that can scale according to the computational resource needs. 
- Easy access from everywhere to every training iteration and all parameters needed for reproducibility.
- Based on MLFlow to keep track of artifacts, parameters and metrics.
- Observe the training progress from any device including Android and iOS through the MLflow server.
- Cloud hosted as well as self-hosted capabilities.
- Metadata from training are organized in a mysql database for automated model comparison and deployment.
- Every trained model is stored as ONNX and tensorflow lite and can be directly deployed into edge and IoT devices.
- Generates a TensorRT engine for NVIDIA hardware inference. 

## Setup guide
Each service runs in a docker container. The demo here shows how all the containers can run on a single server using docker-compose but keep in mind that each of these services can be distributed on different servers. 

 1. Build the docker image, inside the Docker folder, that runs the the ML model training and pull the images required to setup a Minio S3 server and the mysql server. This is based on the tensorflow-2.6-gpu image but this can be replaced with the non-gpu version.
~~~
 docker build -t cloud_compute .
 docker pull mysql
 docker pull minio/minio
 docker pull adminer
~~~
 2. Replace the environment variables inside the .env file and run all the containers with `docker compose up`
 3. Enter the S3 server at http://host-ip:9000 and create 2 new buckets, 'dataset' and 'mlflow'. Replace host-ip with the ip of the remote computer or 'localhost' for a local server.
 4. Enter the container to start the model training with `docker exec -it ml_training bash`. There I first run the preprocessing.py to convert the traffic sign images to constant size and then upload the data to the S3 server. The training code is in main.py. The GTSRB dataset is not included in this repository but it can be downloaded at [https://benchmark.ini.rub.de/](https://benchmark.ini.rub.de/).
 5. The experiments can be seen at the mlflow server http://host-ip:5000.
 6. When all iterations are finished, the code in metadata_processing.py shows how to download the metadata stored in the mysql server and retrieve the best model according to specified metrics.
 7. In onnx_to_trt.py there is sample code to convert a .onnx model to an engine to be used by TensorRT. Tested with TensorRT 21.09.
 
## Demo
As part of this demo, I showcase how this toolchain can be used for development and deployment of models I deal with in my research in assured autonomy. 

The dataset used here is the German Traffic Sign Recognition Benchmark ([GTSRB](https://benchmark.ini.rub.de/)) and I apply an Inductive Conformal Prediction (ICP) based approach that we published in [Trusted Confidence Bounds for Learning Enabled Cyber-Physical Systems](https://arxiv.org/pdf/2003.05107.pdf). 

In ICP one pick a bound/significance level on the error-rate according to some specifications and prediction sets of candidate classes are generated such that the error-rate remains lower or equal to the significance level. 

As we present in our paper, we use a siamese network to map the image inputs into a lower-dimentional embedding representation where the Euclidean distance is a measure of similarity between the corresponding images.

The silhouette is a measure of how well the siamese network clusters the data. That's the metric used to retrieve and deploy the best model from the AWS S3.
