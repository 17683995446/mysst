import netCDF4 as nc
import pandas as pd
# from mpl_toolkits.basemap import Basemap
import numpy as np
import gzip
import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torch
import numpy as np
from torch.utils.data import Dataset
# file = r'./2020/sst_data_intensity_2020.01.nc'
# dataset =nc.Dataset(file)
# all_vars=dataset.variables.keys()
# print("变量个数",len(all_vars))
# #获取所有变量信息
# all_vars_info = dataset.variables.items()
# all_vars_info = list(all_vars_info)
# print("获取所有变量信息：",all_vars_info)
# # 获取单独的一个变量的数据
# precipitationCal=dataset.variables['sst_data_intensity'][:]
# print("获取单独的一个变量的数据:",precipitationCal.shape)
# # # 转换成数组
# var_data = np.array(precipitationCal)
# import matplotlib.pyplot as plt
# # plt.imshow(var_data[0])
# print("最大值：",np.max(var_data))
# print("平均值:",np.average(var_data))
# print("最小值：",np.min(var_data))
# long = dataset.variables['longitude'][:]
# lati = dataset.variables['latitude'][:]
# data = dataset.variables['sst_data_intensity'][1,:,:]
# plt.contourf(long,lati,data)
# plt.colorbar()
# # plt.show()
# # print(var_data)
# print(dataset.variables['longitude'][:].data,dataset.variables['latitude'][:].data)
#


# print("首先查找中国四至范围对应的索引:",np.argwhere(lati==4),np.argwhere(lati==54),
#       np.argwhere(long== 73+180),np.argwhere(long==133+180))
#
#
# print("end")
#用Basemap画地图
# def graph(lon,lat,target,levelT,colorT,title):
#     b_map=Basemap(resolution='l', area_thresh=10000, projection='cyl',
#                   llcrnrlon=min(lon), urcrnrlon=max(lon), llcrnrlat=min(lat),urcrnrlat=max(lat))
#     #llcrnrlon=0, urcrnrlon=360, llcrnrlat=-90,urcrnrlat=90
#     print(type(target))
#     fig=plt.figure(figsize=(9, 6))  #plt.figure(figsize=(12, 8))
#     ax=fig.add_axes([0.1,0.1,0.8,0.8])
#     lon,lat=np.meshgrid(lon,lat)
#     x,y=b_map(lon,lat)
#     print(x.shape,y.shape,target.shape)
#     cs=b_map.contourf(x,y,target,levels=levelT,colors=colorT) #target[0,:,:]
#     b_map.colorbar(cs)
#     b_map.drawcoastlines(linewidth=1)
#     b_map.drawcountries(linewidth=1.5)
#
#     plt.title(title,size=20)
#
#     #plt.savefig('Rainf_0.png',dpi=300)
#     plt.show()
#     plt.close()






#对中国范围的温度作图，设定graph的参数
# title='2m_temperature'

#
#
# level_Tair= [0,0.2,0.4,0.6,0.8,1.0,1.5,2] #[0,2.6,5,8,16,50,100,120,1000]
# #[0,2.6,5,8,10,20,25,300,1000]   [0,210,225,240,255,260,300,305,310,1000]
# colors = ['#FFFFFF', '#AAF0FF', '#C8DC32',  '#FFBE14', '#FF780A',
#           '#FF5A0A', '#F02800',  '#780A00', '#140A00']
#
# #注意这里要对经度做变换，原来东半球经度在180~360区间，现在减去180，转为0~180
# # long=dataset.variables['lon'][1012:1251]-180#[306:530]
# # print(dataset.variables['lon'][1012:1251]-180)
# # #lati= np.flip(nc.variables['latitude'][72:172])
# # lati= dataset.variables['lat'][376:576]#[62:222]+90  #[60:164]
# long=dataset.variables['longitude'][1000:1200]-180#[306:530]
# print(dataset.variables['longitude'][1000:1200]-180)
# #lati= np.flip(nc.variables['latitude'][72:172])
# lati= dataset.variables['latitude'][340:380] #[62:222]+90  #[60:164]

#datai=np.flipud(data[30,72:172,506:630])-274 #转换为摄氏度



# file = r'./2020/sst_data_intensity_2020.01.nc'
# dataset =nc.Dataset(file)
# import os
# path = './2020'
# sst_data = []
# for file_name in os.listdir(path):
#     print(file_name)
#     dataset =nc.Dataset(path+"/"+file_name)
#     datai=dataset.variables['sst_data_intensity'][:,340:380,1000-720:1200-720]
#     sst_data.append(datai)
# sst_data = np.concatenate(sst_data)
# print(sst_data.shape)
# dataset
# for i in range(0,len(sst_data)-19):
#     dataset.append()





# for i in range(0,len(dataset.variables['sst_data_intensity']),10):
#     print(i)
#     datai=dataset.variables['sst_data_intensity'][i,340:380,1000-720:1200-720]#,62:222,306:530] #温度数据切片，选择第30个时间点的温度；将原温度转换为摄氏度
#     print(np.mean(datai))
#     print(np.min(datai))
#     print(np.max(datai))
    # graph(long,lati,datai,level_Tair,colors,title)



def load_sst(root):
    
    total_length = 30 
    input_len = 15
    pred_len = 15
    
    if os.path.exists("dataset.npy"):
        dataset = np.load("dataset.npy")
        dataset_list = []
        for i in range(0,len(dataset)-total_length+1):#19):
            dataset_list.append(dataset[i:i+total_length])#20])
        dataset = np.array(dataset_list)
        index = [i for i in range(len(dataset))]
        X = np.array(dataset[index[:int(0.8*len(index))]])
        Y = np.array(dataset[index[int(0.8*len(index)):]])
        print(X.shape,Y.shape)
        train_x,train_y = X[:,:input_len],X[:,pred_len:]
        test_x,test_y  = Y[:,:input_len],Y[:,pred_len:]
        return train_x,train_y,test_x,test_y
        
    path = root
    sst_data = []
    file_name_list = []
    for file_name in os.listdir(path):
        file_name_list.append(file_name)
    file_name_list.sort() #key=lambda x: int(x[:x.find("-")])
    
    for file_name in file_name_list[:]:#file_name_list[:-1]:
        print(file_name)
        if not file_name.endswith("nc"):
            continue
        dataset =nc.Dataset(path+file_name)
        datai=dataset.variables['sst'][:,340:380,760:960]
        datai = np.array(datai)
        sst_data.append(datai)
    sst_data = np.concatenate(sst_data)
    dataset = sst_data
    
    print(sst_data.shape,len(sst_data))
    
    
    print("before:",dataset.shape)
    T,H,W = dataset.shape
    dataset = dataset.reshape(T,1,H,W)
    print("(Min,Max):",np.min(dataset),np.max(dataset))
    dataset = (dataset - np.min(dataset))/(np.max(dataset) - np.min(dataset))
    
    print("after:",dataset.shape)
    
    np.save("dataset.npy",dataset)
    
    dataset_list = []
    for i in range(0,len(dataset)-total_length+1):#19):
        dataset_list.append(dataset[i:i+total_length])#20])
    dataset = np.array(dataset_list)
    # X = dataset
    # dataset = X
    index = [i for i in range(len(dataset))]
    # np.random.shuffle(index)
    X = np.array(dataset[index[:int(0.8*len(index))]])
    Y = np.array(dataset[index[int(0.8*len(index)):]])
    print(X.shape,Y.shape)
    train_x,train_y = X[:,:input_len],X[:,pred_len:]
    test_x,test_y  = Y[:,:input_len],Y[:,pred_len:]
    
    return train_x,train_y,test_x,test_y


# def load_fixed_set(root):
#     # Load the fixed dataset
#     # filename = 'moving_mnist/mnist_test_seq.npy'
#     # path = os.path.join(root, filename)
#     # dataset = np.load(path)
#     # dataset = dataset[..., np.newaxis]
#     path = './2020'
#     sst_data = []
#     for file_name in os.listdir(path):
#         print(file_name)
#         dataset =nc.Dataset(path+"/"+file_name)
#         datai=dataset.variables['sst_data_intensity'][:,340:380,1000-720:1200-720]
#         sst_data.append(datai)
#     sst_data = np.concatenate(sst_data)
#     print(sst_data.shape)
#     dataset = []
#     for i in range(0,len(sst_data)-19):
#         dataset.append(sst_data[i:i+20])
#     print(np.array(dataset).shape)
#     dataset = np.array(dataset[int(0.8*len(dataset)):])
#     return dataset



# class MovingMNIST(data.Dataset):
#     def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2],
#                  transform=None):
#         super(MovingMNIST, self).__init__()
#
#         self.dataset = None
#         if is_train:
#             self.mnist = load_mnist(root)
#         else:
#             if num_objects[0] != 2:
#                 self.mnist = load_mnist(root)
#             else:
#                 self.dataset = load_fixed_set(root)
#         self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]
#
#         self.is_train = is_train
#         self.num_objects = num_objects
#         self.n_frames_input = n_frames_input
#         self.n_frames_output = n_frames_output
#         self.n_frames_total = self.n_frames_input + self.n_frames_output
#         self.transform = transform
#         # For generating data
#         self.image_size_ = [40,200]
#         self.digit_size_ = [28]
#         self.step_length_ = 0.1
#
#         self.mean = 0
#         self.std = 1
#
#     def get_random_trajectory(self, seq_length):
#         ''' Generate a random sequence of a MNIST digit '''
#         canvas_size = self.image_size_ - self.digit_size_
#         x = random.random()
#         y = random.random()
#         theta = random.random() * 2 * np.pi
#         v_y = np.sin(theta)
#         v_x = np.cos(theta)
#
#         start_y = np.zeros(seq_length)
#         start_x = np.zeros(seq_length)
#         for i in range(seq_length):
#             # Take a step along velocity.
#             y += v_y * self.step_length_
#             x += v_x * self.step_length_
#
#             # Bounce off edges.
#             if x <= 0:
#                 x = 0
#                 v_x = -v_x
#             if x >= 1.0:
#                 x = 1.0
#                 v_x = -v_x
#             if y <= 0:
#                 y = 0
#                 v_y = -v_y
#             if y >= 1.0:
#                 y = 1.0
#                 v_y = -v_y
#             start_y[i] = y
#             start_x[i] = x
#
#         # Scale to the size of the canvas.
#         start_y = (canvas_size * start_y).astype(np.int32)
#         start_x = (canvas_size * start_x).astype(np.int32)
#         return start_y, start_x
#
#     def generate_moving_mnist(self, num_digits=2):
#         '''
#         Get random trajectories for the digits and generate a video.
#         '''
#         data = np.zeros((self.n_frames_total, self.image_size_,
#                          self.image_size_), dtype=np.float32)
#         for n in range(num_digits):
#             # Trajectory
#             start_y, start_x = self.get_random_trajectory(self.n_frames_total)
#             ind = random.randint(0, self.mnist.shape[0] - 1)
#             digit_image = self.mnist[ind]
#             for i in range(self.n_frames_total):
#                 top = start_y[i]
#                 left = start_x[i]
#                 bottom = top + self.digit_size_
#                 right = left + self.digit_size_
#                 # Draw digit
#                 data[i, top:bottom, left:right] = np.maximum(
#                     data[i, top:bottom, left:right], digit_image)
#
#         data = data[..., np.newaxis]
#         return data
#
#     def __getitem__(self, idx):
#         length = self.n_frames_input + self.n_frames_output
#         if self.is_train or self.num_objects[0] != 2:
#             # Sample number of objects
#             num_digits = random.choice(self.num_objects)
#             # Generate data on the fly
#             images = self.generate_moving_mnist(num_digits)
#         else:
#             images = self.dataset[:, idx, ...]
#
#         r = 1
#         w = int(64 / r)
#         images = images.reshape((length, w, r, w, r)).transpose(
#             0, 2, 4, 1, 3).reshape((length, r * r, w, w))
#
#         input = images[:self.n_frames_input]
#         if self.n_frames_output > 0:
#             output = images[self.n_frames_input:length]
#         else:
#             output = []
#
#         output = torch.from_numpy(output / 255.0).contiguous().float()
#         input = torch.from_numpy(input / 255.0).contiguous().float()
#         return input, output
#
#     def __len__(self):
#         return self.length


# train_set = MovingMNIST(root='./2020', is_train=True,
#                         n_frames_input=10, n_frames_output=10, num_objects=[2])



class TrafficDataset(Dataset):
    def __init__(self, X, Y):
        super(TrafficDataset, self).__init__()
        self.X = X#(X + 1) / 2
        self.Y = Y#(Y + 1) / 2
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels


def load_data(batch_size, val_batch_size,
                data_root, num_workers):
    
    
    X_train, Y_train, X_test, Y_test = load_sst(data_root)#dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    train_set = TrafficDataset(X=X_train, Y=Y_train)
    test_set = TrafficDataset(X=X_test, Y=Y_test)
    
    
    
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    

    return dataloader_train,dataloader_test,dataloader_test, 0, 1


if __name__ == '__main__':
    dataloader_train,dataloader_test,dataloader_test, mean, std = load_data(32, 16,'../data/sst/', 4)
    for x,y in dataloader_test:
        print(x.shape)
        print(y.shape)

# for x,y in train_set:
#     print(x,y)