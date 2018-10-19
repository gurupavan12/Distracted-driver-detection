import os
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10
test_path = '/storage/work/pug64/trial2vgg/imgs/test/'
train_path = '/storage/work/pug64/trial2vgg/imgs/train/'
val_path = '/storage/work/pug64/trial2vgg/imgs/validation/'

#count=0
count_test =0 
count_train =0
count_val =0
#count_train_full =0
for i in range(NUM_CLASSES):
    
    curr_test_path = test_path + 'c' + str(i) + '/'
    curr_train_path = train_path + 'c' + str(i) + '/'
    curr_val_path = val_path + 'c' + str(i) + '/'
    
    count_test = count_test + len(os.listdir(curr_test_path))
    count_train = count_train + len(os.listdir(curr_train_path))
#    count_train = len(os.listdir(curr_train_path))
    count_val = count_val + len(os.listdir(curr_val_path))

print('Number of val images: ' + str(count_val))
print('Number of testing images: ' + str(count_test))
print('Number of training images:' + str(count_train))
print('Total: ' + str(count_test + count_train + count_val) + '  Actual total : 22424')