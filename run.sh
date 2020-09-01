# This is a file containing examples to running our software.

# 0. Create Environment
conda create -n deepsim python=3.8
conda activate deepsim
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install --file requirements.txt

# 1. Train
# After training, a new model file (XXX.pth) will appear in the "models" folder.

# SiameseResNet
python train.py --model SiameseResNet --data mix --margin 4 --lr 3e-4 --step 10 --nepoch 50 --batch_size 4
python train.py --model SiameseResNet --data raw --margin 4 --lr 4e-5 --step 10 --nepoch 100 --batch_size 4
python train.py --model SiameseResNet --data aug --margin 4 --lr 3e-4 --step 10 --nepoch 50 --batch_size 4

# SiameseSqueezeNet
python train.py --model SiameseSqueezeNet --data raw --margin 4 --lr 2e-4 --step 10 --nepoch 100 --batch_size 4
python train.py --model SiameseSqueezeNet --data aug --margin 4 --lr 3e-4 --step 10 --nepoch 50 --batch_size 4

# FCNet
python train.py --model FCNet --lr 5e-4 --step 10 --nepoch 100 --batch_size 16

# FCSiameseNet
python train.py --model FCSiameseNet --data raw --margin 2 --lr 1e-2 --step 10 --nepoch 100 --batch_size 16

# 2. Evaluate

# evaluate SiameseResNet trained on raw data (non-augmented data)
python evaluate.py --model SiameseResNet --margin 4 --weight SiameseResNet_05-12-00-28_best.pth
python -c "print('↑'*35 + 'Best' + '↑'*35, '\n', '↓'*35 + 'Final' + '↓'*35)"
python evaluate.py --model SiameseResNet --margin 4 --weight SiameseResNet_05-12-00-28_final.pth

# evaluate SiameseResNet trained on augmented data
python evaluate.py --model SiameseResNet --margin 4 --weight SiameseResNet_05-13-04-18_best.pth
python -c "print('↑'*35 + 'Best' + '↑'*35, '\n', '↓'*35 + 'Final' + '↓'*35)"
python evaluate.py --model SiameseResNet --margin 4 --weight SiameseResNet_05-13-04-18_final.pth

# use England data to evaluate SiameseResNet trained on augmented data
python evaluate.py --model SiameseResNet --margin 4 --weight SiameseResNet_05-13-04-18_best.pth --data england
python -c "print('↑'*35 + 'Best' + '↑'*35, '\n', '↓'*35 + 'Final' + '↓'*35)"
python evaluate.py --model SiameseResNet --margin 4 --weight SiameseResNet_05-13-04-18_final.pth --data england

# evaluate Squeeze Net trained on raw data
python evaluate.py --model SiameseSqueezeNet --margin 4 --weight SiameseSqueezeNet_05-13-01-33_best.pth
python -c "print('↑'*35 + 'Best' + '↑'*35, '\n', '↓'*35 + 'Final' + '↓'*35)"
python evaluate.py --model SiameseSqueezeNet --margin 4 --weight SiameseSqueezeNet_05-13-01-33_final.pth

# evaluate Squeeze Net trained on augmented data
python evaluate.py --model SiameseSqueezeNet --margin 4 --weight SiameseSqueezeNet_05-13-05-28_best.pth
python -c "print('↑'*35 + 'Best' + '↑'*35, '\n', '↓'*35 + 'Final' + '↓'*35)"
python evaluate.py --model SiameseSqueezeNet --margin 4 --weight SiameseSqueezeNet_05-13-05-28_final.pth

# use England data to evaluate Squeeze Net trained on augmented data
python evaluate.py --model SiameseSqueezeNet --margin 4 --weight SiameseSqueezeNet_05-13-05-28_best.pth --data england
python -c "print('↑'*35 + 'Best' + '↑'*35, '\n', '↓'*35 + 'Final' + '↓'*35)"
python evaluate.py --model SiameseSqueezeNet --margin 4 --weight SiameseSqueezeNet_05-13-05-28_final.pth --data england

# 3. Evaluate Error Tolerance

# error tolerance of Siamese ResNet
python evaluate.py --model SiameseResNet --data err --margin 4 --weight SiameseResNet_05-13-04-18_best.pth

# error tolerance of Squeeze Net
python evaluate.py --model SiameseSqueezeNet --data err --margin 4 --weight SiameseSqueezeNet_05-13-05-28_best.pth
