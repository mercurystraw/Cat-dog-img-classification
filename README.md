# Cat-dog-img-classification
## Structure

```python
+---config.json
+---data_utils.py
+---dataset.py
+---train.py
+---transfer_train.py
+---dataset
|   \---A
|       +---cat
|       \---dog
+---model_zoo
|   +---alexnet.py
|   +---resnet.py
|   \---vgg.py
```



## Details

### 1. config.json

It contains some  conﬁgure parameters of training the model, such as DATASET, MODEL_TYPE, EPOCH, BATCH_SIZE and LR, for easier adjustment.

### 2. data_utils.py 

It mainly includes **load_data function**, which is to transform local image data to the size applicable for the model input, and do random_split (training set 80% and testset 20%), return trainset and testset.

### 3. dataset.py

Mainly defines a class named CustomDataset to initialize the given dataset. For each image data, we add a transform interface, and iteratively label cat image as 0, dog image as 1 for later comparison.

### 4. train.py

Load datasets, configuration, and choose loss criterion, optimizer, scheduler. Besides, define **train function** which outputs Train_ACC and Train_loss, **eval_clean function** which outputs Test_ACC. Lastly, save the model to specific path and print the final metrics: Train accurary and Test accuracy. 

### 5. transfer_train.py

Mostly same as train.py. There is a slight difference in choosing a neural network. Here, we specify it as pre-trained Resnet50.

### 6. dataset/

It includes local dataset for classfication.

### 7.model_zoo/

It includes serval models we can choose to train.



## How to use

Frist, if you do not have local dataset, unzip the dataset.zip, make sure it's the same structure as mentioned at beginning.

Then, choose the model you want to use, change MODEL_TYPE in config.json ("resnet18" for ResNet18; "vgg11" for VGG11; "resnet50" for pre-trained resnet50). If you want other model, just add its structure to model_zoo/ and change model-choosing code in train.py.

After that, for Task1, run code:

```bash
python train.py
```

or for Task2, run code

```bash
python transfer_train.py
```



## Results

We present our results for the 2 tasks. You can check them in train_result.txt and transfer_train_result.txt. 