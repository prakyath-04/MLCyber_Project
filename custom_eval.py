import keras
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt 
import PIL.Image
import matplotlib
from PIL import Image


clean_data_filename = str(sys.argv[1])
gmodel_filename = str(sys.argv[2])
bmodel_filename = str(sys.argv[3])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def combined_model_labels(x):
    gd_model = keras.models.load_model(gmodel_filename)
    bd_model = keras.models.load_model(bmodel_filename)
    
    clean_label = np.argmax(gd_model.predict(x), axis=1)
    poison_label = np.argmax(bd_model.predict(x), axis=1)
    
    new_y = []
    
    len_ = len(clean_label) 
    for i in range(len_):
        if clean_label[i] == poison_label[i]:
            new_y.append(clean_label[i])
        else:
            new_y.append(1283)
    new_label = np.array(new_y)
    return new_label
    
def main():
    if clean_data_filename.find(".h5")!=-1:
        x_test, _ = data_loader(clean_data_filename)
        x_test = data_preprocess(x_test)
    else:
        x_test = Image.open(clean_data_filename)
        x_test = x_test.convert('RGB')
        x_test = np.array(x_test)
        x_test = x_test[None,:]
        
    ans = combined_model_labels(x_test)
    print('reapired net outputs: \n',ans)

if __name__ == '__main__':
    main()
