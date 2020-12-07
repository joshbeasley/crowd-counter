import os
import torch
import numpy as np
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

#test data and model file path
data_path =  './data/original/shanghaitech/part_A_final/test_data/images/'
gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'
model_path = './final_models/cmtl_shtechA_204.h5'


output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
#load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
f = open(file_results, 'w') 
# net.cuda()
with torch.no_grad():
    net.eval()
    mae = 0.0
    mse = 0.0
    i = 0
    for blob in data_loader:                
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        print("{}: {}, {}".format(i, gt_count, et_count))
        f.write("{}: {}, {}\n".format(i, gt_count, et_count))
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))
        if vis:
            utils.display_results(im_data, gt_data, density_map)
        if save_output:
            utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        i += 1
        
        
        
mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print('MAE: {}, MSE: {}'.format(mae,mse))

f.write('MAE: {}, MSE: {}'.format(mae,mse))
f.close()
    
