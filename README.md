# DeepSim: GPS Spoofing Detection Artifact
This repo is the source code for **DeepSIM: GPS Spoofing Detection on UAVs using Satellite Imagery Matching**.

## Abstract
In this artifact, we will provide the source codes of our implementations as well as the corresponding dataset (approx.~12.08 Gigabyte with 967 aerial photos and paired satellite images), which are used in Section 8 of our accepted paper. The dataset consists of training set and test set which can be used to train a new model from scratch and evaluate a trained model, respectively. We also provide trained models that were used in our evaluation procedure. The training and testing of our models require a CUDA-enabled GPU in Linux OS, and the software is implemented using Python. To run our on-board model, i.e. SqueezeNet v1.1, on the IoT platform, a Raspberry Pi is needed, where 3B+ with CPU $\ge$ 1.4GHz and memory $\ge$ 1GB is recommended. Our software also depends on Pytorch, Numpy, and some other Python libraries for it to run.


- [DeepSim: GPS Spoofing Detection Artifact](#deepsim-gps-spoofing-detection-artifact)
  - [Abstract](#abstract)
  - [Objective](#objective)
  - [How to Run](#how-to-run)
    - [Environment](#environment)
    - [Hardware Recommendation](#hardware-recommendation)
    - [Software and Package](#software-and-package)
  - [Data Directory Organization](#data-directory-organization)
    - [Dataset](#dataset)
    - [Data Augmentation](#data-augmentation)
    - [Trained Models](#trained-models)
    - [Commands to run](#commands-to-run)
  - [Code File Organization](#code-file-organization)
  - [Run on Raspberry Pi](#run-on-raspberry-pi)
  - [Successful running screen shot](#successful-running-screen-shot)
  - [Updating information](#updating-information)
  - [Cite our work](#cite-our-work)
## Objective
Run our deep learning models, and they will compare the aerial photos with the satellite images, to see whether a drone is attacked by GPS spoofing.
An example of a paired aerial photo and its corresponding image is shown as follows.
![](https://i.imgur.com/9c5PGDD.jpg)


## How to Run
Follow the instructions below, you can reproduce our program easily. 

### Environment
Anaconda + Python 3.7 or higher, and other software in requirements.txt.

Please create a conda env and install pytorch 1.4 and other software. Refer to `run.sh` for an example.
Download the dataset we provided and put them under the source code folder. 

### Hardware Recommendation
* On-ground
If you want to train a model from scratch, a GPU with at least 16G video memory is recommended.
We trained our models on an NVIDIA Tesla V100. Batch size is set to 2 if video memory is 16G and 4 if 32G. 
Training new models on Cloud platform like [Google Colab](https://colab.research.google.com) is an option for your reference.
* On-board
Raspberry Pi is needed, where 3B+ with CPU ≥ 1.4GHz and memory ≥ 1GB is recommended.

### Software and Package
Pytorch, Numpy, cuDNN, CUDA, OpenCV, pandas, h5py, tqdm,matplotlib, seaborn, sklearn, packaging.

## Data Directory Organization
The data derectories are available at [Google Drive](https://drive.google.com/drive/u/1/folders/1F0mMpq_C5RTKCVQiktFoZgUZLvRRpf2o).

| Directory   | Functionality                                                |
| ----------- | ------------------------------------------------------------ |
| mid_product | h5 files, i.e., features extracted by the backbone neural network, ResNet. With those files you do not need to extract features from the original images, thus can speed up the detection process. |
| models      | A series of trained models for GPS spoofing detection.       |
| dataset     | Collected aerial photos and satellite images.                |
|GPS-Spoofing-Detection     |Source code for training and test.

### Dataset
Here we only provide preprocessed data for easy running and evalution. In general, these photos in our dataset can be devided into categories: aerial photography and satellite imagery. Each has its corresponding counterpart.

| Name             | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| England_960x720                    	| For generalization ability test-only.                                                  	|
| error_tolerance                	| For error tolerance test.	|
| full_960x720                                           	| For training and validation.
| full_aug_960x720                    	|For training and validation with data augmentation technology.


If you want the raw dataset, please send an email to goldgaruda@gmail.com.
Please first visit [Satellite imagery VS Aerial Photos](https://drive.google.com/drive/u/1/folders/1LKBjDpgqeuE7mjVdFYO9vbHBwPHjlzgG) to download the dataset and necessary files. 
Then put the data into `config.GDRIVE_DIR`.
For the raw aerial photos from Merlischachen, Renens，Lausanne and Le Bourget Airport, please visit [senseFly dataset](https://www.sensefly.com/education/datasets/) for more information.

### Data Augmentation
We have provided data after augmentation. However, if you want to do it by yourself,
For data augmentation source, please visit [Here](https://github.com/Lariiii/DeepSimDataAugmentation.git).

The implemented augmentation methods are: grayscaling, blurring, cropping, weather augmentation (e.g. snow, fog, clouds), rotating and adjusting the brightness (e.g. lighter or darker).

An example of cloud and fog effect is shown below:
![](https://i.imgur.com/BjsfPtz.jpg)


* generate the newly preprocessed images by running the augmentation.py file with the desired amount of generated image pairs and adapted filepaths
* if you want to generate preprocessed images for all available images, then run the augmentation.py file for every available method and the amount of all available pairs

```
python augmentation.py
```

### Trained Models
We also provide our trained models for GPS spoofing inferences. With these models, you do not need to train a new one from scratch for detection. You can directly use it to carry out the evaluation.

Besides, you can use the preprocessed data to train a completely new model as you like.

### Commands to run
1. With proper environment setup and dataset downloaded to the source code folder, you can now start to run the training and evaluation procedure.
2. For training Siamese ResNet, please run:
`python train.py --model SiameseResNet --data aug --margin 4 --lr 3e-4 --step 10 --nepoch 50 --batch_size 4` 
For evaluation, please run:
`python evaluate.py --model SiameseResNet --margin 4 --weight [modelname].pth`

3. Please see more examples and explanations in `run.sh`.

## Code File Organization
| File                         	| Functionality                                                       	|
| ---------------- | ------------------------------------------------------------ |
| config.py                    	| Configurations.                                                    	|
| DataLoader.py                	| Basic data loader functions, mostly used for model 1 in the paper. 	|
| Dataset.py                   	| Pytorch Dataset Classes.                                           	|
| train.py                     	| Training code.                                                     	|
| evaluate.py                  	| Evaluation code (spoofing detection).                                 |
| net.py                       	| Neural Network definitions.                                        	|
| preprocess.py                	| Resize rename and generate h5py data file.                         	|
| utils.py                     	| Utility code.                                                     	|
| euclidean_distance.py       	| Model 1 code.                                                     	|
| run.sh                       	| Examples of commands to run our software.                             |
| rpi.md                     	| Instruction for Raspberry Pi.                                        	|
| requirements.txt             	| Python software requirements.                                      	|
| Draw_ROC_loss.ipynb       	| Jupyter notebook to draw ROC curves.                                 	|
| GPS_Spoofing_Detection.ipynb 	| Model 1 Jupyter notebook.                                          	|
| Visualize_Model.ipynb        	| Jupyter notebook to visualize the models.                          	|


## Run on Raspberry Pi
If you want to run models on Raspberry Pi, please refer to `rpi.md`.

## Successful running screen shot
An successful training result (1 epoch) is shown below.
![](https://i.imgur.com/VljSa0S.png)


## Updating information
Please visit [here](https://hackmd.io/2GvHLw2wQSSMqwEzMDFx4A) for the newest updating.

## Cite our work
Please use the following bibtex code to cite our work:
```
@InProceedings{nian2020deepsim,
  title={{DeepSIM: GPS Spoofing Detection on UAVs using Satellite Imagery Matching}},
  author={Nian Xue, Liang Niu, Xianbin Hong, Zhen Li, Larissa Hoffaeller, Christina Pöpper},
  booktitle={Proceedings of the Annual Computer Security Applications Conference 
  (ACSAC)},
  year={2020},
  doi={10.1145/3427228.3427254}
}
```

