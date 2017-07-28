---
Customized DataLoader for multi label classification-[pytorch implementation]
---
### 1. Details of file fold:
- data/
- data/train_img/*.jpg
- data/train_img.txt
- data/train_label.txt
- data/test_img/*.jpg
- data/test_img.txt
- data/test_label.txt

### 2. File description:

| file | description|
|---|---|
|data/train_img/|training image fold|
|data/test_img/|testing image fold|
|data/train_img.txt|file name list for training image |
|data/test_img.txt|file name list for testing image |
|data/train_label.txt|label list for training image|
|data/test_label.txt| label list for testing image|

### 3. Running example:
requirements:
```python
torch
torchvision
```
running example:
```python
python multi_label_classifier.py
```
output:
```python
Training Phase: Epoch: [ 0][ 0/ 3]	Iteration Loss: 0.693
Training Phase: Epoch: [ 1][ 0/ 3]	Iteration Loss: 0.660
Training Phase: Epoch: [ 2][ 0/ 3]	Iteration Loss: 0.619
Training Phase: Epoch: [ 3][ 0/ 3]	Iteration Loss: 0.596
Training Phase: Epoch: [ 4][ 0/ 3]	Iteration Loss: 0.542
Training Phase: Epoch: [ 5][ 0/ 3]	Iteration Loss: 0.509
Training Phase: Epoch: [ 6][ 0/ 3]	Iteration Loss: 0.467
Training Phase: Epoch: [ 7][ 0/ 3]	Iteration Loss: 0.464
Training Phase: Epoch: [ 8][ 0/ 3]	Iteration Loss: 0.439
Training Phase: Epoch: [ 9][ 0/ 3]	Iteration Loss: 0.377
Training Phase: Epoch: [10][ 0/ 3]	Iteration Loss: 0.329
Training Phase: Epoch: [11][ 0/ 3]	Iteration Loss: 0.324
```
### 4. Dataset:
We use the following dataset for our example:
[link](http://lamda.nju.edu.cn/data_MIMLimage.ashx).