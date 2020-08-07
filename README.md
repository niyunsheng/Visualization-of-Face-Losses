# Visualization of Face Loss

It contains softmax loss, norm softmax loss, L-softmax loss, A-softmax loss, CosFace loss, ArcFace loss.

## Model Architectures

### The Main Module

We use the LeNet++ architecture which is come from the paper CenterLoss published in ECCV 2016.

The CNNs Architectures is sample. Visit the paper CenterLoss Table 1 for more details.

![](images/LeNet++.png)

I also complete a mini resnet18 model, see `model.py` for more details.

You can change the command line parameters to select different models.

### Angle Linear Module

We use the formula which can be found in the paper ArcFace publiceded in CVPR 2019. 

$$L=-\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s\left(\cos \left(m_{1} \theta_{y}+m_{2}\right)-m_{3}\right)}}{e^{s\left(\cos \left(m_{1} \theta_{y_{i}}+m_{2}\right)-m_{3}\right)}+\sum_{j=1, j \neq y_{i}}^{n} e^{s \cos \theta_{j}}}$$

To make it more universal, the module `AngleLinear` has some parameters to present different losses.

| loss name    | w_norm | x_norm | s | m1 | m2  | m3 |
| ---------    | ------ | ------ | - | -- | --  | -- |
| softmax      | False  | False  | 1 | 1  | 0   | 0  |
| L-softmax_v1 | False  | False  | 1 | 2  | 0   | 0  |
| A-softmax_v1 | True   | False  | 1 | 2  | 0   | 0  |
| A-softmax_v2 | True   | False  | 1 | 3  | 0   | 0  |
| norm-softmax | True   | True   | 1 | 1  | 0   | 0  |
| CosFace_v1   | True   | True   | 4 | 1  | 0   | 0.1|
| CosFace_v2   | True   | True   | 4 | 1  | 0   | 0.2|
| ArcFace_v1   | True   | True   | 4 | 1  | 0.1 | 0  |
| ArcFace_v2   | True   | True   | 4 | 1  | 0.2 | 0  |
| ArcFace_v3   | True   | True   | 4 | 1  | 0.3 | 0  |

## To Run

1. `python main.py --lossname ArcFace_v1`, or other loss in the table above
2. `python create_git.py --lossname ArcFace_v1`, or other loss in the table above
3. Read the config file `config.json` for more details

## Results

We use the SGD optimizer and train 20 epoches for each loss. The acc is below.

| loss name    | train acc | test acc |
| ---------    | ----- | ----- |
| softmax      | 93.34 | 94.65 |
| L-softmax_v1 | 93.79 | 92.92 |
| A-softmax_v1 | 94.12 | 94.07 |
| norm-softmax | 91.65 | 92.37 |
| CosFace_v1   | 95.52 | 96.32 |
| ArcFace_v1   | Nan   | Nan   |
| ArcFace_v2   | Nan   | Nan   |
| ArcFace_v3   | Nan   | Nan   |

> `Nan` means that I haven’t trained with similar results because the encode spatial dimension is too low.

## Visualization

### softmax

| train gif | train end |
| - | - |
| ![image](images/softmax_train.gif) | ![image](images/softmax_train_epoch19.png) |

| test gif | test end |
| - | - |
| ![image](images/softmax_test.gif) | ![image](images/softmax_test_epoch19.png) |

### L-softmax_v1

| train gif | train end |
| - | - |
| ![image](images/L-softmax_v1_train.gif) | ![image](images/L-softmax_v1_train_epoch19.png) |

| test gif | test end |
| - | - |
| ![image](images/L-softmax_v1_test.gif) | ![image](images/L-softmax_v1_test_epoch19.png) |

### A-softmax_v1

| train gif | train end |
| - | - |
| ![image](images/A-softmax_v1_train.gif) | ![image](images/A-softmax_v1_train_epoch19.png) |

| test gif | test end |
| - | - |
| ![image](images/A-softmax_v1_test.gif) | ![image](images/A-softmax_v1_test_epoch19.png) |


### norm-softmax

| train gif | train end |
| - | - |
| ![image](images/norm-softmax_train.gif) | ![image](images/norm-softmax_train_epoch19.png) |

| test gif | test end |
| - | - |
| ![image](images/norm-softmax_test.gif) | ![image](images/norm-softmax_test_epoch19.png) |

### CosFace_v1

| train gif | train end |
| - | - |
| ![image](images/CosFace_v1_train.gif) | ![image](images/CosFace_v1_train_epoch19.png) |

| test gif | test end |
| - | - |
| ![image](images/CosFace_v1_test.gif) | ![image](images/CosFace_v1_test_epoch19.png) |


### ArcFace_v1

<!-- | train gif | train end |
| - | - |
| ![image](images/ArcFace_v1_train.gif) | ![image](images/ArcFace_v1_train_epoch19.png) |

| test gif | test end |
| - | - |
| ![image](images/ArcFace_v1_test.gif) | ![image](images/ArcFace_v1_test_epoch19.png) | -->

### ArcFacev2

<!-- | train gif | train end |
| - | - |
| ![image](images/ArcFace_v2_train.gif) | ![image](images/ArcFace_v2_train_epoch19.png) |

| test gif | test end |
| - | - |
| ![image](images/ArcFace_v2_test.gif) | ![image](images/ArcFace_v2_test_epoch19.png) | -->

### ArcFace_v3

<!-- | train gif | train end |
| - | - |
| ![image](images/ArcFace_v3_train.gif) | ![image](images/ArcFace_v3_train_epoch19.png) |

| test gif | test end |
| - | - |
| ![image](images/ArcFace_v3_test.gif) | ![image](images/ArcFace_v3_test_epoch19.png) | -->