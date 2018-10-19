import os
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10
data_path = './original/'

for i in range(NUM_CLASSES):
    
    curr_dir_path = data_path + 'c' + str(i) + '/'    
    xtrain = labels = os.listdir(curr_dir_path)    
    x, x_test, y, y_test = train_test_split(xtrain,labels,test_size=0.2,train_size=0.8)
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.125,train_size =0.875)
    
    for x in x_train:
        
        if (not os.path.exists('new_data/train/' + 'c' + str(i) + '/')):
            os.makedirs('new_data/train/' + 'c' + str(i) + '/')
            
        os.rename(data_path + 'c' + str(i) + '/' + x, 'new_data/train/' + 'c' + str(i) + '/' + x)
        
    for x in x_test:
        
        if (not os.path.exists('new_data/test/' + 'c' + str(i) + '/')):
            os.makedirs('new_data/test/' + 'c' + str(i) + '/')
            
        os.rename(data_path + 'c' + str(i) + '/' + x, 'new_data/test/' + 'c' + str(i) + '/' + x)
    
    for x in x_val:
        
        if (not os.path.exists('new_data/validation/' + 'c' + str(i) + '/')):
            os.makedirs('new_data/validation/' + 'c' + str(i) + '/')
            
        os.rename(data_path + 'c' + str(i) + '/' + x, 'new_data/validation/' + 'c' + str(i) + '/' + x)