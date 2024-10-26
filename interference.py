#created by CarloSia to test the model with unseen data. Please check dataprep.py to follow how to prep the audio file

import os
import argparse
import sys
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

# Dataset class
class My_Dataset(Dataset):
    def __init__(self, pathway, data_id):
        # Load only X_test and Y_test
        X_test, Y_test = torch.load(pathway)

        if data_id == 2:
            self.data, self.labels = X_test, Y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.labels[index]  # torch.size: [1]
        image = self.data[index]  # torch.size: [1,128,44]
        return image, label

# Helper functions
def get_network(args):
    """Return the specified network."""
    if args.net == 'bv_cnn':
        from models.bv_net import bv_cnn
        net = bv_cnn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    else:
        print('The network name you have entered is not supported yet.')
        sys.exit()

    if args.gpu:  # Use GPU if specified
        net = net.cuda()

    return net

def get_mydataloader(pathway, data_id=2, batch_size=16, num_workers=2, shuffle=False):
    """Return a DataLoader for the dataset."""
    Mydataset = My_Dataset(pathway, data_id)  
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return Data_loader

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    distress_count = 0
    barn_sound_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            preds = (outputs.squeeze(dim=-1) > 0.5).float()  # Get predictions

            # Count predictions for each class
            distress_count += preds.sum().item()  # Count distress calls (pred = 1)
            barn_sound_count += (preds == 0).sum().item()  # Count barn sounds (pred = 0)

    print(f'Distress calls identified: {distress_count}')
    print(f'Natural barn sounds identified: {barn_sound_count}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='vgg11', help='Network type')
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not')
    parser.add_argument('--data_path', type=str, default='./enhancements/datadump_folder/archive.pt', help='Path to input data')  # Set a default data path
    parser.add_argument('--weight_path', type=str, default='vgg11-best.pth', help='Path to the trained model weights')  # Set a default weight path
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if args.gpu else 'cpu')
    
    # Load the network
    net = get_network(args)
    net.to(device)

    # Load the model weights
    net.load_state_dict(torch.load(args.weight_path))
    print("Loaded model weights from:", args.weight_path)
    
    # Prepare test data loader
    test_loader = get_mydataloader(args.data_path, data_id=2, batch_size=128, shuffle=False)

    # Evaluate the model
    evaluate_model(net, test_loader)

