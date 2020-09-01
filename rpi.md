# How to setup a Raspberry Pi 3B+ and run our model on it

## 1. Install system
Please refer to [Download Raspbian](https://www.raspberrypi.org/downloads/raspbian/)
and follow the instructions.

## 2. Install conda
Download berryconda installer for RasPi 3B+. (I have tried official miniconda for RasPi, which does not work.)
I would suggest you do these in tmux/screen if you are in a ssh environment.
First, you may wanna use the following commands to download and run the installer script:

```bash
wget https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh
chmod +x Berryconda3-2.0.0-Linux-armv7l.sh
./Berryconda3-2.0.0-Linux-armv7l.sh
```

Follow the instructions to finish the installation of miniconda, choose yes when the installer prompts you 
"Do you wish the installer to prepend the Miniconda3 install location to PATH in your /home/pi/.bashrc ?".
And then, log out and log in again to let `.bashrc` work. Finally, this version of miniconda is quite outdated, so you will want to do:

```bash
conda update conda
conda update --all
```

to update the conda command and base environment.

## 3. Create conda env for pytorch
Create the conda env with Python3.6:

`conda create -n deepsim python=3.6 numpy scipy pandas scikit-learn scikit-image ipython`

This command will also install matplotlib, pillow and some other dependencies we needed. It takes a while to finish, you may have a coffee break.

After the env is created, activate this env and then install other dependencies:

```bash
source activate deepsim
conda install seaborn h5py
conda install -c gaiar pytorch-cpu
conda install -c gaiar torchvision
pip install tqdm
```
You may also want to install other required softwares described in requirements.txt.

## 4. Download code and models
Now we need to download the trained models, dataset and the code to the Raspberry Pi, so that we could run our model on Raspberry Pi.

#### clone this repo
```bash
cd ~
git clone https://github.com/wangxiaodiu/DeepSim.git
cd DeepSim
```

#### download trained model and the dataset
As described in `README.md`, download the `dataset`, `mid_product` and `models` folders, put them under the `DeepSim` folder.

#### run the evaluation on RasPi
Finally, we could run the evaluation on a Raspberry Pi:
```bash
python evaluate.py --model SiameseSqueezeNet --margin 4 --weight SiameseSqueezeNet_02-19-15-24_best.pth
```

## 5. Possible issue and solution
To solve tkinter related error:
```bash
conda install tk
sudo apt-get install tk-dev python3-tk python-tk
conda install python
```
