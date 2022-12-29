from Models import Unet
from Datasets import Datasets
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

IMG_PATH = '../../data/custom/test/img'
SAVE_DIR = '../../output/'

BREAKAGE_PATH = '../../data/weight/Breakage.pt'
CRUSHED_PATH = '../../data/weight/Crushed.pt'
SCRATCHED_PATH = '../../data/weight/Scratched.pt'
SEPARATED_PATH = '../../data/weight/Separated.pt'
DATA_INFO = '../../data/datainfo/damage_test.json'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_visual(origImage, origMask, predMask1, predMask2, predMask3, predMask4, idx):
    # initialize our figure
    figure, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0][0].imshow(origImage)
    ax[0][1].imshow(origMask.permute(1, 2, 0))
    ax[0][2].axis('off')
    ax[0][3].axis('off')

    ax[1][0].imshow(predMask1)
    ax[1][1].imshow(predMask2)
    ax[1][2].imshow(predMask3)
    ax[1][3].imshow(predMask4)

    # set the titles of the subplots
    ax[0][0].set_title("Original Image")
    ax[0][1].set_title("Original Mask")
    ax[1][0].set_title("Breakage")
    ax[1][1].set_title("Crushed")
    ax[1][2].set_title("Scratched")
    ax[1][3].set_title("Separated")

    # set the layout of the figure and display it
    figure.tight_layout()
    
    #plt.show()
    
    figure.savefig(SAVE_DIR+'damage/damage_test_output.jpg')

def get_dataloader(dataset):
    eval_loader = DataLoader(
        dataset = dataset,
        shuffle = False, 
        num_workers = 0)
    
    return eval_loader

def make_predictions(model1, model2, model3, model4):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    with torch.no_grad():
        
        test_datasets = Datasets(DATA_INFO, 'train', size = 256, label = None, one_channel = False, img_base_path = IMG_PATH)
        eval_data_loader = get_dataloader(test_datasets)
        
        idx=1

        for image, masks, img_ids in (eval_data_loader):

            if idx<10:
                idx+=1
                continue

            orig = image[0].permute(1, 2, 0).numpy().copy()
            image = torch.tensor(image).float().to(DEVICE)

            predMask1 = model1(image)
            predMask1 = torch.argmax(predMask1, dim=1).detach().cpu().numpy()
            predMask1 = np.transpose(predMask1, (1,2,0))

            predMask2 = model2(image)
            predMask2 = torch.argmax(predMask2, dim=1).detach().cpu().numpy()
            predMask2 = np.transpose(predMask2, (1,2,0))

            predMask3 = model3(image)
            predMask3 = torch.argmax(predMask3, dim=1).detach().cpu().numpy()
            predMask3 = np.transpose(predMask3, (1,2,0))

            predMask4 = model4(image)
            predMask4 = torch.argmax(predMask4, dim=1).detach().cpu().numpy()
            predMask4 = np.transpose(predMask4, (1,2,0))

            #print(img_ids)

            with open(SAVE_DIR+'damage/predMask1.txt', 'w') as outfile:
                for slice_2d in predMask1:
                    np.savetxt(outfile, slice_2d)
            
            with open(SAVE_DIR+'damage/predMask2.txt', 'w') as outfile:
                for slice_2d in predMask2:
                    np.savetxt(outfile, slice_2d)

            with open(SAVE_DIR+'damage/predMask3.txt', 'w') as outfile:
                for slice_2d in predMask3:
                    np.savetxt(outfile, slice_2d)

            with open(SAVE_DIR+'damage/predMask4.txt', 'w') as outfile:
                for slice_2d in predMask4:
                    np.savetxt(outfile, slice_2d)

            save_visual(orig, masks, predMask1, predMask2, predMask3, predMask4, idx)
            idx+=1

            break

def load_model(weight_path):
    model = Unet(encoder="resnet34",pre_weight='imagenet',num_classes=2)
    model = model.to(DEVICE)
    model.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cuda')))
    return model.model

def main():
    print('started saving outputs...')

    model1 = load_model(weight_path=BREAKAGE_PATH)
    model2 = load_model(weight_path=CRUSHED_PATH)
    model3 = load_model(weight_path=SCRATCHED_PATH)
    model4 = load_model(weight_path=SEPARATED_PATH)
    make_predictions(model1, model2, model3, model4)

    print('completed')



if __name__ == '__main__':
    main()

