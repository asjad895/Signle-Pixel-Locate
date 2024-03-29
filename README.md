# Single Pixel Locate

 A Computer vision model for single point localization

## How to run on local systems

## clone repo

```git clone https://github.com/asjad895/Signle-Pixel-Locate.git```

go to dir ```cd Signle-Pixel-Locate```

## Prerequisites(latest)

- Python
- Tensorflow
- opencv-python
- matplotlib
- numpy
- pandas

## Create Virtual env

```conda create -n <environment_name>```

## activate environment

```conda activate <environment_name>```

## Install requirements

```pip install -r requirements.txt```

## Data

Read Data dir readme.md

## Training

To train the model, run the following command:

- **Notes**
- I trained on kaggle due to RAM and GPU .so i have not tested on it my local env with full dataset
- for running on local machine adapt code like file path and import helping function dataset
- ```python -m Pipeline.pipeline.py```
- ```python -m Pipeline.train.py``` or ```python Pipeline.test.py``` for individual running(before that must run ```python -m Pipeline.Data_loading.py``` and ```python -m Pipeline.Data_preprocessing.py```)
u can run notebook for experiment how model is trained

## Testing

To test the model, run the following command:

```python Pipeline.test.py```

## pipeline

```python main.py```

## Results

![loss](/Analysis/training_analysis.png)

![arch](/Analysis/model_architecture.png)
![test](/Test_Data_Result/Result_test_data.png)

## Acknowledgements

This project is based on the following resources:

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details

## Run in Container with Docker

 Before running the scripts, ensure you have Docker installed on your machine. You can download Docker from [here](https://www.docker.com/products/docker-desktop).

## Building the Docker Image

```docker build -t ml .```

## Running the Docker Image

```docker run ml```

## Run  Scripts

```docker run ml -m Pipeline.pipeline.py``` complete steps
```docker run ml -m Pipeline.train.py```
```docker run ml -m Pipeline.test.py```
```docker run ml main.py``` run app(in future)
