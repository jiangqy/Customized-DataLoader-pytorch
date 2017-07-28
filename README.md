---
Customized DataLoader for multi label classification-[pytorch implementation]
---
### Introduction
---

### Dataset arrangement:
#### 1. Details of file constructure:
- data/
- data/train_img/*.jpg
- data/train_img.txt
- data/train_label.txt
- data/test_img/*.jpg
- data/test_img.txt
- data/test_label.txt

#### 2. File description:

| file | description|
|---|---|
|data/train_img/|training image fold|
|data/test_img/|testing image fold|
|data/train_img.txt|file name list for training image |
|data/test_img.txt|file name list for testing image |
|data/train_label.txt|label list for training image|
|data/test_label.txt| label list for testing image|