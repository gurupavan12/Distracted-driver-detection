Three different types of networks were used

***Single program can be run on the ResNet50 model predict_single_img.py Since the model is bad(explained in report), you can see a bad accuracy of all the predicted labels. ***

-ResNet50
--> model_resnet2.h5 has the model and its weights. Can be loaded directly.
--> model_resnet2 is the json file for the weights
--> model_resnet2_weights.h5 is the weights file and can be accessed
--> predict_kaggle_test.py : Loads model and generate .csv for 79k images of the test dataset to submit to the kaggle(The .csv can be found in submission folder)
--> train.py : Trains the network by freezing upto the top model and the last five layers. Same code used for the first non-best resnet model whose weights are present in the non-best folder

-VGG16
The best network has been saved in res/top_model3.h5
--> predict_kaggle_test.py : Loads model and generate .csv for 79k images of the test dataset to submit to the kaggle(The .csv can be found in submission folder)
--> helper.py : Creates the top model of VGG16
--> extract_vgg16_features : Extracts the VGG-16 network and saves it in res/vgg_(train/val/test)_features. It can be accessed for the deep features. 
--> train_top.py : Trains the network's top model. 
--> test.py : Tests the network for small dataset of test images

-small CNN
Two types of small CNN's implemented. They were done in the same code. Only one of their interactive display was copied onto the notepad. You can check them in that folder. 


--> count_num_imgs.py was for counting the number of images in subfolders. Give appropriate path.
--> data_split.py was for splitting the train(22,424) images into train, test and validation. 


Keras, Tensorflow to be installed.
conda install keras
conda install tensorflow

Dataset is available at <https://www.kaggle.com/c/state-farm-distracted-driver-detection/data>

