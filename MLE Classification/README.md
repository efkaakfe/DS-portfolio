# Neural Network classification within Docker container
The aim of the project is to train and test neural network for multiclass classification using Docker containers. For this purpose, Pytorch was used as the main tool for building network.

## Project structure:

Below is the structure of the project:

```
MLE
├── data_process              # Scripts used for data processing and generation, generated .csv data files
│   ├── data_generation.py
│   └── __init__.py   
|   |__train.csv
|   |__ytrain.csv
|   |__test.csv
|   |__ytest.csv        
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── requirements.txt          # Requirements for running scripts
└── README.md
|__.gitignore                 # .gitignore file
```

## Data:
This project was built on iris dataset uploaded from Wikipedia website. For uploading the data, use the script located at `data_process/data_pull.py`. This script pulls data from the website, divides it to fours datasets: train/ytrain/test/ytest and saves in the same folder.

## Training:
The training part includes data preprocessing, training different networks and evaluation on validation set. The validation set was created by applying train_test_split on train/ytrain datasets. Final model with the highest validation score is saved in the container. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image:
```bash
docker build -f ./training/Dockerfile -t train_img .
```
NOTE: Sometimes image cannot be built at once. If it happens, please run the command again.
- Run the container:
```bash
docker run --name train_c -t train_img 
```
Move the trained model from the directory inside the Docker container `/app/model` to the local machine using:
```bash
docker cp train_c:/app/model ./model
```

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```

## Inference:
The inference stage is implemented in `inference/run.py`. The script uses model saved in training stage for batch inference on test set. The output is new file with actual and predicted labels.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile -t inf_img .
```
- Run the inference Docker container:
```bash
docker run --name infer_c -t inf_img
```
- Copy results file into local storage:
```bash
docker cp infer_c:/app/results ./results 
```

2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```
