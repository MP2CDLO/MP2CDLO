import os
import json
import shutil
import numpy as np
from tqdm import tqdm


def data_summary(dir_list, type_list, root_path,target_path):

    for data_type in tqdm(type_list, desc='Data summary'):
        prefix = data_type.split('.')[0]
        suffix = data_type.split('.')[1]

        target_dir = os.path.join(target_path, prefix)
        os.makedirs(target_dir, exist_ok=True)
        
        file_counter = 1
        
        for dir_name in dir_list:
            source_dir = os.path.join(root_path, dir_name)
            
            if not os.path.exists(source_dir):
                print(f"Source directory {source_dir} does not exist.")
                continue
            
            for file_name in tqdm(sorted(os.listdir(source_dir)),desc='Processing files'):
                if not file_name.startswith(prefix):
                    continue
                source_file = os.path.join(source_dir, file_name)
                
                if os.path.isfile(source_file):
                 
                    target_file = os.path.join(target_dir, f"{prefix}_{file_counter:04d}.{suffix}")
                    
                    
                    shutil.copy(source_file, target_file)
                    
                    file_counter += 1

def data_resize(dir_list,type_list,root_path, target_path,step=10):
    '''
  
    '''
    for data_type in tqdm(type_list, desc='Data resizing'):
        prefix = data_type.split('.')[0]
        suffix = data_type.split('.')[1]

        target_dir = os.path.join(target_path, prefix)
        os.makedirs(target_dir, exist_ok=True)
        
        file_counter = 0
        
        for dir_name in dir_list:
            source_dir = os.path.join(root_path, dir_name)
            
            if not os.path.exists(source_dir):
                print(f"Source directory {source_dir} does not exist.")
                continue
            
            for file_name in tqdm(sorted(os.listdir(source_dir)),desc='Processing files'):
                if not file_name.startswith(prefix):
                    continue
                source_file = os.path.join(source_dir, file_name)
                
                if os.path.isfile(source_file):
                    file_counter += 1
                else:
                    continue
                if file_counter % step == 0:

                    
                    target_file = os.path.join(target_dir, f"{prefix}_{file_counter//step:04d}.{suffix}")
                

                    shutil.copy(source_file, target_file)

def random_sample(points, num_points,device=None):
   

    indices = np.random.choice(len(points), num_points, replace=False if len(points) >= num_points else True)

    sampled_points = points[indices]
    return sampled_points

def down_sample(source_dir, num_points=2048,dtype='.npy'):
  
    file_list = sorted(f for f in os.listdir(source_dir) if f.endswith(dtype))
    for file in tqdm(file_list,desc='Down sampling'):
        file_path = os.path.join(source_dir, file)
    
        points = np.load(file_path,allow_pickle=True)
        

        sampled_points:np.ndarray = random_sample(points, num_points)
        
 
        target_path = os.path.join(source_dir, file)

        np.save(target_path, sampled_points)

def pc_normalize(source_dir,dtype='.npy'):   

    max_std = 0
    centroid_list = []

    file_list = [f for f in sorted(os.listdir(source_dir)) if f.endswith(dtype)]
   
    for i,file in tqdm(enumerate(file_list),desc='Normalizing'):
        
        file_path = os.path.join(source_dir,file)

       
        pcd = np.load(file_path)

        centroid = np.mean(pcd, axis=0)
        centroid_list.append(np.around(centroid,decimals=3))
        
        pcd = pcd - centroid
        
        

        dist = np.max(np.sqrt(np.sum(pcd**2, axis=1))) 
        pcd = pcd / dist
        save_path = os.path.join(source_dir,f"pointcloud_{centroid_list[i]}_{dist:0.3f}_{i:04d}.npy")
        np.save(save_path, pcd)
        os.remove(file_path)
       
def output_json(json_path,partial_path):

    file_list_partial = [os.path.splitext(f)[0] for f in os.listdir(partial_path) if f.endswith('.npy')]
    

    Train={"partial":file_list_partial[:(int)(len(file_list_partial)*19/20)],"complete":file_list_partial[:(int)(len(file_list_partial)*19/20)]}
    Test={"partial":file_list_partial[(int)(len(file_list_partial)*19/20):],"complete":file_list_partial[(int)(len(file_list_partial)*19/20):]}
    json_path=os.path.join(json_path,"EPN3D.json")
    with open(json_path, 'w') as f:
        class_list=[{"taxonomy_id":"20240915","taxonomy_name":"rope","train":Train,"test":Test}]
        json.dump(class_list, f,indent=4) 

def convert_to_train_format(source_dir):
    pointcloud_path = os.path.join(source_dir,'pointcloud')
    file_list = [f for f in sorted(os.listdir(pointcloud_path)) if f.endswith('.npy')]
    os.makedirs(os.path.join(source_dir,'complete'),exist_ok=True)
    for file in tqdm(file_list,desc='Style converting'):
        source_file = os.path.join(pointcloud_path,file)
        target_file = os.path.join(source_dir,'complete',file)
        shutil.copy(source_file,target_file)
    new_name = os.path.join(source_dir,'partial')
    os.rename(pointcloud_path,new_name)

        

root_path = '../data'
target_path = os.path.join(root_path, 'kinect_data')
dir_list = ['kinect_data_very_hard']
# dir_list = ['kinect_data01', 'kinect_data02', 'kinect_data03', 'kinect_data04', 'kinect_data05','kinect_data06']
type_list = ['pointcloud.npy', 'color_image.png']

# 
# data_summary(dir_list, type_list, root_path = root_path ,target_path=target_path)

root_path = '../data/kinect_data'
dir_list = ['pointcloud','color_image']
type_list = ['pointcloud.npy', 'color_image.png']
resized_dir = os.path.join( '../output_data')
pcd_dir = os.path.join(resized_dir,'pointcloud')
json_path = os.path.join('./')
pcd_dir = os.path.join('./','pointcloud')
# os.makedirs(pcd_dir,exist_ok=True)
# print(os.path.join(__file__))
# data_resize(dir_list,type_list,root_path, resized_dir,step=1)
# down_sample(pcd_dir)
# pc_normalize(pcd_dir)
output_json(json_path,pcd_dir)
# # pc_denormalize(resized_dir, resized_dir)
# # pcd_visualization(pcd_path,png_path,prefix_list,index_bound,step)

source_dir = './'
convert_to_train_format(source_dir)
