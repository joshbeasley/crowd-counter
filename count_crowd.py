import os
import torch
import numpy as np
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils
import argparse


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

#test data and model file path
data_path =  './data/'
# modelA = './final_models/cmtl_shtechA_204.h5'
# modelB = './final_models/cmtl_shtechB_768.h5'

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", action = "store", dest="data", type=str, help="folder of images to conduct crow counting on")
args = parser.parse_args()

model_path = './final_models/cmtl_shtechB_768.h5'
data_path =  './data/' + args.data
output_dir = './output/' + args.data
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#load test data
data_loader = ImageDataLoader(data_path)

net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
f = open(file_results, 'w') 

with torch.no_grad():
    net.eval()
    i = 0
    et_total = 0
    for blob in data_loader:                
        im_data = blob['data']
        density_map = net(im_data)
        density_map = density_map.data.cpu().numpy()
        et_count = np.sum(density_map)
        et_total += et_count
        print("{}: {}".format(i, et_count))
        f.write("{}: {}\n".format(i, et_count))
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        i += 1
    f.write("Total: {}\n".format(et_total))

f.close()
    
