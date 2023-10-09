# CoorLGNet.pytorch

## CoorLGNet Framework
![image](https://github.com/BM-AI-Lab/CoorLGNet/blob/master/CoorLGNet_model.png)

## Implementation of [https: //pan.baidu.com/s/1zFI8-9UN0jk6E7yJBMn8_w?pwd=vlc8]()   Extracted code：vlc8

### Set up
```
- python==3.7
- cuda==11.7

# other pytorch/timm version can also work

pip install torch==1.7.0 torchvision==0.8.1;
pip install timm==0.4.12;
pip install torchprofile;

```

### Data preparation

Dataset storage format:

```
│CoorLGNet/dataset/
├──HealthyMeander/
   ├── mea1-H1.jpg
   ├── mea1-H2.jpg
   ├── ......
   ├── ......
├──PatientMeander/
   ├── mea1-P1.jpg
   ├── mea1-P2.jpg
   ├── ......
   ├── ......
```

#### Introduction to the use of the code


```
1. Set `--data-path` to the `dataset` folder absolute path in the `train.py` script
2. Set `--weights`, `--batch_size`, `--epochs`, `--weight_decay`, `--lr` and and other parameters in the `train.py` script
3. After setting the `--data-path` and parameters, you can start training using the `train.py` script (the `class_indices.json` file will be automatically generated during the training process)
4. Import the same model in the `predict.py` script as in the training script and set `model_weight_path` to the trained model weight path (saved in the weights folder by default)
5. In the `predict.py` script, set `img_path` to the absolute path of the image you want to predict
6. Set the weight path `model_weight_path` and the predicted image path `img_path` and you can use the `predict.py` script to make predictions

```

