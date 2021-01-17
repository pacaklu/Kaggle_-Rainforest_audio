import librosa
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
import cv2
import random 
import imageio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import skimage
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage import exposure



def prepare_mel_spectogram(sample, sampling_rate, tmin, tmax, fmin, fmax, n_fft = 2048, hop_length = 512 , crop = 'center'):
    """
    prepares scaled mel spectogram
    """
    

    length = 10
    n_mels = 224
    image_width = 400

    center_point = (tmax * sampling_rate + tmin * sampling_rate)/2

    # center cropping
    if crop == 'center':
        start_point = (center_point - (length/2) * sampling_rate)
    #random cropping
    else:
        start_point = center_point -  random.uniform(0, 5)* sampling_rate
    
    end_point = (start_point + length*sampling_rate)


    start_point = math.floor(start_point)
    end_point = math.floor(end_point)

    if start_point < 0:
        len_to_add = np.abs(start_point) # to have all the spectograms with same length
        start_point = 0
        end_point = end_point + len_to_add


    if end_point > (60*sampling_rate):
        len_to_add = end_point - 60*sampling_rate # to have all the spectograms with same length
        end_point = 60*sampling_rate
        start_point = start_point - len_to_add

    sample = sample[start_point:end_point]

    D = librosa.feature.melspectrogram(y=sample, sr=sampling_rate, n_mels=n_mels, fmin = fmin, fmax = fmax, power = 2)
    D = librosa.power_to_db(D, top_db=80)  # optionally converts aplitudes to decibels, x(DB) = 20 * log10(S) 

    D = resize(D, (224, image_width))

    mean = D.mean()
    std = D.std()
    D = (D - mean) / (std )

    D = D - np.min(D)
    D = D / np.max(D)

    D = D*255

    return D


def partition (list_in, percentage):
    """
    Randomly splits list to 2 sublists 
    """
    random.shuffle(list_in)
    border = math.floor(len(list_in)*percentage)
    return list_in[border:], list_in[:border]

def is_in(input, b_min, b_max):
    '''
    whether the number is between b_min and b_max
    '''
    if ((input<b_max) and (input>b_min)):
        return True
    else:
        return False



class FocalLoss(nn.Module):
    '''
    FocalLoss class
    downloaded from pytorch forum
    '''

    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = torch.nn.BCEWithLogitsLoss(reduce = False)(inputs, targets)
        else:
            BCE_loss = torch.nn.BCELoss(reduce = False)(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



def return_nn(num_classes, device):
    """
    Returns NN architecture
    """
    
    model_ft = models.resnet50(pretrained=True)
    number_features = model_ft.fc.in_features # number of features in the final layer

    model_ft.fc = nn.Sequential(
    nn.BatchNorm1d(2048),
    nn.Dropout(p=0.4),

    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Dropout(p=0.4),

    nn.Linear(1024, num_classes),
    )

    model_ft = model_ft.to(device)
    model_ft.to(device)
    
    return model_ft

from torch.utils.data import Dataset




def specaugment(image):
    """
    Augmentation of spectogram
    Randomly deletes some times and frequencies
    """
    x_max = image.shape[1]
    y_max = image.shape[0]
    x_size = 15
    y_size = 8
    
    x_start = np.random.randint(x_max)
    if (x_start + x_size > x_max):
        x_end = x_max
    else:
        x_end = x_start + x_size
        
    y_start = np.random.randint(y_max)
    if (y_start + y_size > y_max):
        y_end = y_max
    else:
        y_end = y_start + y_size
    
    #mean = np.mean(image)
    image[:,x_start:x_end] = 0
    image[y_start:y_end,:] = 0 
    
    return image


#AUGMENTATION FUNCTIONS:
def addNoisy(img):
    noise_img = random_noise(img)
    return noise_img

def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    contrast_img = exposure.rescale_intensity(img, in_range=(p2, p98))
    return contrast_img

def log_correction(img):
    log_img = exposure.adjust_log(img)
    return log_img

def randomGaussian(img):
    gaussian_img = gaussian(img, sigma=random.randint(0, 5))
    return gaussian_img



def print_images(path_train, desired_index, image_number):

    counter = 0
    for image_name in os.listdir(path_train):
        if (int(image_name.split('_')[0]) == desired_index):
            plt.figure(figsize = (10,8))
            imgplot = plt.imshow(imageio.imread(os.path.join(path_train, image_name)))
            plt.show()
            counter = counter+1
        if counter == image_number:
            break



class torch_dataset(Dataset):
    def __init__(self, filelist, path_to_save, num_classes, phase):
        
        files = pd.read_csv("E:\\kaggle_data\\train\\_tp.csv")

        self.spectograms = []
        self.label_array = []       

        
        for file in os.listdir(path_to_save):           #going throught all files in folder

            recording_id = file.split('_')[1]
          

            if recording_id in filelist:                #check whether file is in files that should be considered
                image_orig = imageio.imread(os.path.join(path_to_save, file))/255

                #shift = np.random.randint(400) 
                #image_orig = np.roll(image_orig, shift, axis = 1)
                
                image = np.stack((image_orig, image_orig, image_orig))
                self.spectograms.append(image)
                
                species_id = int(file.split('_')[0])         #species id
                label = np.zeros(num_classes)                #creation of zeros array with 1 in species id index 
                label[species_id] = 1

                
                # Adding of all labels to the current frame, works only with central coppping of sound (not random)

                sampling_rate = 48000
                length = 10

                tmax = files[((files['recording_id']==recording_id) & (files['species_id']==species_id))]['t_max'].iloc[0]
                tmin = files[((files['recording_id']==recording_id) & (files['species_id']==species_id))]['t_min'].iloc[0]

                center_point = (tmax * sampling_rate + tmin * sampling_rate)/2
                start_point = (center_point - (length/2) * sampling_rate)
                end_point = (start_point + length*sampling_rate)

                start_point = math.floor(start_point)
                end_point = math.floor(end_point)

                if start_point < 0:
                    len_to_add = np.abs(start_point) # to have all the spectograms with same length
                    start_point = 0
                    end_point = end_point + len_to_add


                if end_point > (60*sampling_rate):
                    len_to_add = end_point - 60*sampling_rate # to have all the spectograms with same length
                    end_point = 60*sampling_rate
                    start_point = start_point - len_to_add

                t_max_image = end_point/sampling_rate
                t_min_image = start_point/sampling_rate

                subset = files[files['recording_id'] == recording_id].reset_index(drop = True)
                for index, row in subset.iterrows():
                    if subset['species_id'].iloc[index] != species_id: #pokud se nejedna o aktualni species_id
                        if ((is_in(subset['t_max'].iloc[index], t_min_image, t_max_image)) or (is_in(subset['t_min'].iloc[index], t_min_image, t_max_image))): #pokud je tmax nebo tmin uvnitr aktualniho okna
                            label[subset['species_id'].iloc[index]] = 1
                

                self.label_array.append(label)  

                if phase == 'train':

                    # adding gaussian noice
                    image = addNoisy(image_orig)
                    image = np.stack((image, image, image))
                    self.spectograms.append(image)
                    self.label_array.append(label)

                    # adding gaussian blur
                    p2, p98 = np.percentile(image_orig, (2, 98))
                    image = exposure.rescale_intensity(image_orig, in_range=(p2, p98))
                    image = np.stack((image, image, image))
                    self.spectograms.append(image)
                    self.label_array.append(label)

                    #specaugment 
                    image = exposure.adjust_log(image_orig)
                    image = np.stack((image, image, image))
                    self.spectograms.append(image)
                    self.label_array.append(label)


                    #horizontal flip
                    #image = gaussian(image_orig, sigma=random.randint(0, 5))
                    #image = np.stack((image, image, image))
                    #self.spectograms.append(image)
                    #self.label_array.append(label)

                    #random image shift
                    shift = np.random.randint(400) 
                    image = np.roll(image_orig, shift, axis = 1)
                    image = np.stack((image, image, image))
                    self.spectograms.append(image)
                    self.label_array.append(label)



        c = list(zip(self.spectograms, self.label_array))
        random.shuffle(c)
        self.spectograms, self.label_array = zip(*c)

    
    def __getitem__(self, index):
        return self.spectograms[index], self.label_array[index]
        
    
    def __len__(self):
        return len(self.spectograms)



def score_and_extend(evaluate_path, model, list_valid, files):
    """
    Scores clip-wisely data and appends to this DF matrix with true labels 
    """

    evaluated_data = score_images(evaluate_path, model, list_valid)

    for new_col in range(0,24):
        evaluated_data[f's{new_col}_true'] = np.zeros(evaluated_data.shape[0])
    for index, _ in evaluated_data.iterrows():
        for index2, _ in files.iterrows():
            if evaluated_data['recording_id'][index] == files['recording_id'][index2]:
                species = files['species_id'][index2]
                evaluated_data[f's{species}_true'].loc[index] = 1

    return evaluated_data



def score_images(test_path, trained_model, test_files):
    """
    Scores valid_files images (that are split to 6 10 sec chunks) and aggregates prediction (max)
    This functions is adjusted to scoring of valid files with label
    """

    submission_df = {'s0':[], 's1':[], 's2':[], 's3':[], 's4':[], 's5':[], 's6':[], 's7':[],
                 's8':[], 's9':[], 's10':[], 's11':[], 's12':[], 's13':[], 's14':[], 's15':[], 's16':[],
                's17':[], 's17':[], 's18':[], 's19':[], 's20':[], 's21':[], 's22':[], 's23':[]}
    submission_df = pd.DataFrame(submission_df)

    recording_id = {'recording_id':[]}
    recording_id = pd.DataFrame(recording_id)

    list_of_all_files = os.listdir(test_path)  # list all files in given folder

    trained_model.eval()

    counter = 0
    for filename in list_of_all_files: # go throught all files in folder

        name = filename.split('_')[0]

        if name in test_files: #check whether recording_id of tested file is in set we want to score

            filepath = os.path.join(test_path, filename)

            image = imageio.imread(filepath)/255
            image = np.stack((image, image, image))
            image = torch.tensor(image)
            image = image.float()
            image = image.cuda()

            prediction = trained_model(image[None, ...]).cpu().detach().numpy()[0]
            submission_df.loc[len(submission_df)] = prediction
            recording_id.loc[len(recording_id)] = [name]
    
        #if (counter % 100) == 0:
            #print(f'files_{counter} out of {len(test_files)} scored')
        #counter = counter + 1
        
    result = pd.concat([recording_id, submission_df], axis=1)

    fn = 'max'
    result = make_agg(result, 'recording_id', {'s0':[fn], 's1':[fn], 's2':[fn], 's3':[fn],
                    's4':[fn], 's5':[fn], 's6':[fn], 's7':[fn], 's8':[fn],
                    's9':[fn], 's10':[fn], 's11':[fn], 's12':[fn], 's13':[fn],
                    's14':[fn], 's15':[fn], 's16':[fn], 's17':[fn],'s18':[fn],
                    's19':[fn], 's20':[fn], 's21':[fn], 's22':[fn], 's23':[fn] })

    return result



def score_images_test_set(test_path,  test_files):
    """
    Scores test_files images (that are split to 6 10 sec chunks) and aggregates prediction (max)
    This functions is adjusted to scoring of test files without label
    """

    submission_df = {'s0':[], 's1':[], 's2':[], 's3':[], 's4':[], 's5':[], 's6':[], 's7':[],
                 's8':[], 's9':[], 's10':[], 's11':[], 's12':[], 's13':[], 's14':[], 's15':[], 's16':[],
                's17':[], 's17':[], 's18':[], 's19':[], 's20':[], 's21':[], 's22':[], 's23':[]}
    submission_df = pd.DataFrame(submission_df)

    recording_id = {'recording_id':[]}
    recording_id = pd.DataFrame(recording_id)

    list_of_all_files = os.listdir(test_path)  # list all files in given folder

    counter = 0

    soubor_modelu = []

    for model_path in os.listdir("E:\\kaggle_data\\saved_models"):
        model = torch.load(f"E:\\kaggle_data\\saved_models\\{model_path}")              
        soubor_modelu.append(model)
        del model

    for filename in list_of_all_files: # go throught all files in folder

        name = filename.split('_')[0]

        if name in test_files: #check whether recording_id of tested file is in set we want to score

            filepath = os.path.join(test_path, filename)

            image = imageio.imread(filepath)/255
            image = np.stack((image, image, image))
            image = torch.tensor(image)
            image = image.float()
            image = image.cuda()

            predictions = np.zeros(24)
            for model in soubor_modelu:             
                model.eval()
                prediction = predictions + model(image[None, ...]).cpu().detach().numpy()[0]
                del model

            prediction = prediction/len(os.listdir("E:\\kaggle_data\\saved_models"))
            
            submission_df.loc[len(submission_df)] = prediction
            recording_id.loc[len(recording_id)] = [name]
    
        if (counter % 100) == 0:
            print(f'files_{counter} out of {len(list_of_all_files)} scored')
        counter = counter + 1
        
    result = pd.concat([recording_id, submission_df], axis=1)

    fn = 'max'
    result = make_agg(result, 'recording_id', {'s0':[fn], 's1':[fn], 's2':[fn], 's3':[fn],
                    's4':[fn], 's5':[fn], 's6':[fn], 's7':[fn], 's8':[fn],
                    's9':[fn], 's10':[fn], 's11':[fn], 's12':[fn], 's13':[fn],
                    's14':[fn], 's15':[fn], 's16':[fn], 's17':[fn],'s18':[fn],
                    's19':[fn], 's20':[fn], 's21':[fn], 's22':[fn], 's23':[fn] })

    return result


def LWLRAP(preds, labels):
    """
    LWLRAP metric computation
    - downloaded from kaggle
    """
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    # Number of GT labels per instance
    num_labels = labels.sum(-1)
    pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()



def make_agg(dataset,key,aggregation_function):
    """
    aggregates DF by key
    """
    agreg=dataset.groupby(key).agg(aggregation_function)
    agreg.columns=[(col)[0] for col in list(agreg)]
    agreg.reset_index(inplace=True) 
    return agreg



def load_audio(path):
    """
    Load audio with librosa
    """
    amplitudes, sampling_rate = librosa.load(path, sr = None)
    return amplitudes, sampling_rate
