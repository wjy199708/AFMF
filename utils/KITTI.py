import os.path
import numpy as np
import time
import math
import cv2
import ctypes
import os
import sys
import pathlib
sys.path.append(pathlib.Path.cwd().parent)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import plot_bev, get_points_in_a_rotated_box, plot_label_map, trasform_label2metric, load_config
from utils2d import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_gaussian
import calibra as calibration


class KITTI(Dataset):

    geometry = {
        'L1': -40.0,
        'L2': 40.0,
        'W1': 0.0,
        'W2': 70.0,
        'H1': -2.5,
        'H2': 1.1,
        'interval': 0.1,
        'image_input_shape': (384, 1248),
        'image_label_shape': (96, 312),
        'lidar_input_shape': (800, 700, 37),
        'knn_shape': (400, 350, 18),
        'lidar_label_shape': (200, 175, 7)
    }

    # height, width
    target_mean = np.array([0.008, 0.001, 0.202, 0.2, 0.43, 1.368])
    target_std_dev = np.array([0.866, 0.5, 0.954, 0.668, 0.09, 0.111])

    def __init__(self, frame_range=10000, use_npy=False, train=True):
        self.frame_range = frame_range
        self.velo = []
        self.use_npy = use_npy
        self.LidarLib = ctypes.cdll.LoadLibrary('preprocess/LidarPreprocess.so')
        self.image_sets = self.load_imageset(train)  # names
        self.image = []
        self.calib = []
        self.max_objs = 50

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, item):
        image = self.load_image(item)
        calib = self.load_calib(item)
        ''' 原生点云数据 '''
        point = self.load_velo_origin(item)
        ''' bev转换 '''
        scan = self.load_velo_scan(item)
        ''' knn检索得到的对应的，图片 '''
        bev2image, pc_diff = self.find_knn_image(calib, scan, point, k=1)

        label_map, _, image_label = self.get_label(item)
        self.reg_target_transform(label_map)

        image = torch.from_numpy(image)
        scan = torch.from_numpy(scan)
        bev2image = torch.from_numpy(bev2image)
        label_map = torch.from_numpy(label_map)
        pc_diff = torch.from_numpy(pc_diff)
        image = image.float()
        bev2image = bev2image.float()
        pc_diff = pc_diff.float()
        image = image.permute(2, 0, 1)
        scan = scan.permute(2, 0, 1)
        label_map = label_map.permute(2, 0, 1)

        for key in image_label.keys():
            image_label[key] = torch.from_numpy(image_label[key])
            image_label[key] = image_label[key].float()

        return scan, image, bev2image, pc_diff, label_map, image_label, item

    def reg_target_transform(self, label_map):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: [200 * 175 * 7] label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean) / self.target_std_dev

    def load_imageset(self, train):
        path = KITTI_PATH
        if train:
            path = os.path.join(path, "train.txt")
        else:
            path = os.path.join(path, "val.txt")

        with open(path, 'r') as f:
            lines = f.readlines()  # get rid of \n symbol
            names = []
            for line in lines[:-1]:
                if int(line[:-1]) < self.frame_range:
                    names.append(line[:-1])

            # Last line does not have a \n symbol
            last = lines[-1][:6]
            if int(last) < self.frame_range:
                names.append(last)
            # print(names[-1])
            print("There are {} images in txt file".format(len(names)))

            return names

    def load_image(self, item):
        img_file = self.image[item]
        assert os.path.exists(img_file)
        image = cv2.imread(img_file)
        # TODO need check, image_width and image_height
        print("before", image.size)
        image = cv2.resize(image, self.geometry['image_shape'], interpolation=cv2.INTER_CUBIC)
        print("after", image.size)
        return image

    def load_calib(self, item):
        calib_file = self.calib[item]
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
    def interpret_kitti_label(self, bbox):
        w, h, l, y, z, x, yaw = bbox[8:15]
        y = -y
        yaw = - (yaw + np.pi / 2)

        return x, y, w, l, yaw

    def interpret_custom_label(self, bbox):
        w, l, x, y, yaw = bbox
        return x, y, w, l, yaw

    def interpret_kitti_2D_label(self, bbox):
        y1, x1, y2, x2 = bbox[4:8]
        return y1, x1, y2, x2

    def get_corners_2d(self, bbox):
        y1, x1, y2, x2 = bbox[4:8]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        height = x2 - x1
        width = y2 - y1

        reg_target = [xc, yc, height, width]
        return reg_target

    def get_corners(self, bbox):
        # base velo cord
        w, h, l, y, z, x, yaw = bbox[8:15]
        y = -y
        # manually take a negative s. t. it's a right-hand system, with
        # x facing in the front windshield of the car
        # z facing up
        # y facing to the left of driver

        yaw = -(yaw + np.pi / 2)

        #x, y, w, l, yaw = self.interpret_kitti_label(bbox)

        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
        bev_corners[0, 1] = y - l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
        bev_corners[1, 1] = y - l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
        bev_corners[2, 1] = y + l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
        bev_corners[3, 1] = y + l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)

        reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

        return bev_corners, reg_target

    # def update_image_label_map(self, map, reg_target):
    #     label_x = reg_target[0]//4
    #     label_y = reg_target[1]//4
    #     map[label_x, label_y, 0] = 1.0
    #     reg_x = reg_target[0] - reg_target[0]//4*4
    #     reg_y = reg_target[1] - reg_target[1]//4*4
    #     map[label_x, label_y, 1:5] = [reg_target[2], reg_target[3], reg_x, reg_y]

    def update_label_map(self, map, bev_corners, reg_target):
        label_corners = (bev_corners / 4) / self.geometry['interval']
        label_corners[:, 1] += self.geometry['lidar_label_shape'][0] / 2

        points = get_points_in_a_rotated_box(label_corners, self.geometry['lidar_label_shape'])

        for p in points:
            label_x = p[0]
            label_y = p[1]
            # TODO can be better. output ans is in metric space. But in label map space is better
            metric_x, metric_y = trasform_label2metric(np.array(p))
            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            actual_reg_target[4] = np.log(reg_target[4])
            actual_reg_target[5] = np.log(reg_target[5])

            map[label_y, label_x, 0] = 1.0
            map[label_y, label_x, 1:7] = actual_reg_target

    def update_image_label(self, hm, wh, reg, ind, reg_mask, gt_det, k, cls_id, reg_target):
        xc, yc, h, w = reg_target
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            ct = np.array([xc, yc], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm[cls_id], ct_int, radius)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * self.geometry['image_label_shape'][1] + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1

            gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                           ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

    def get_label(self, index):
        '''
        :param i: the ith velodyne scan in the train/val set
        :return: label map: <--- This is the learning target
                a tensor of shape 800 * 700 * 7 representing the expected output


                label_list: <--- Intended for evaluation metrics & visualization
                a list of length n; n =  number of cars + (truck+van+tram+dontcare) in the frame
                each entry is another list, where the first element of this list indicates if the object
                is a car or one of the 'dontcare' (truck,van,etc) object

        '''
        index = self.image_sets[index]
        f_name = (6 - len(index)) * '0' + index + '.txt'
        label_path = os.path.join(KITTI_PATH, 'training', 'label_2', f_name)

        object_list = {'Car': 1, 'Truck': 0, 'DontCare': 0, 'Van': 0, 'Tram': 0}
        label_map = np.zeros(self.geometry['lidar_label_shape'], dtype=np.float32)
        label_list = []

        hm = np.zeros((1, *self.geometry['image_label_shape'][0:2]), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        #dense_wh = np.zeros((2, *self.geometry['image_label_shape'][0:2]), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        gt_det = []

        with open(label_path, 'r') as f:
            lines = f.readlines()  # get rid of \n symbol
            for k, line in enumerate(lines):
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])
                    if name == 'Car':
                        corners, reg_target = self.get_corners(bbox)
                        self.update_label_map(label_map, corners, reg_target)
                        label_list.append(corners)

                        reg_target_2d = self.get_corners_2d(bbox)
                        self.update_image_label(hm, wh, reg, ind, reg_mask, gt_det, k, 0, reg_target_2d)
        image_label = {'hm': hm, 'wh': wh, 'reg': reg, 'ind': ind, 'reg_mask': reg_mask, 'gt_det': gt_det}
        return label_map, label_list, image_label

    def get_rand_velo(self):
        import random
        rand_v = random.choice(self.velo)
        print("A Velodyne Scan has shape ", rand_v.shape)
        return random.choice(self.velo)

    def load_velo_scan(self, item):
        """Helper method to parse velodyne binary files into a list of scans."""
        filename = self.velo[item]

        if self.use_npy:
            scan = np.load(filename[:-4] + '.npy')
        else:
            c_name = bytes(filename, 'utf-8')
            scan = np.zeros(self.geometry['lidar_input_shape'], dtype=np.float32)
            c_data = ctypes.c_void_p(scan.ctypes.data)
            self.LidarLib.createTopViewMaps(c_data, c_name)
            #scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)

        return scan

    def load_velo_origin(self, item):
        filename = self.velo[item]
        scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        return scan

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files

        velo_files = []
        image_files = []
        calib_files = []

        for file in self.image_sets:
            velo_file = '{}.bin'.format(file)
            velo_files.append(os.path.join(KITTI_PATH, 'training', 'velodyne', velo_file))
            image_file = '{}.png'.format(file)
            image_files.append(os.path.join(KITTI_PATH, 'training', 'image_2', image_file))
            calib_file = '{}.txt'.format(file)
            calib_files.append(os.path.join(KITTI_PATH, 'training', 'calib', calib_file))

        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')
        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        self.velo = velo_files
        self.image = image_files
        self.calib = calib_files

        print('done.')

    def point_in_roi(self, point):
        if (point[0] - self.geometry['W1']) < 0.01 or (self.geometry['W2'] - point[0]) < 0.01:
            return False
        if (point[1] - self.geometry['L1']) < 0.01 or (self.geometry['L2'] - point[1]) < 0.01:
            return False
        if (point[2] - self.geometry['H1']) < 0.01 or (self.geometry['H2'] - point[2]) < 0.01:
            return False
        return True

    def passthrough(self, velo):
        geom = self.geometry
        q = (geom['W1'] < velo[:, 0]) * (velo[:, 0] < geom['W2']) * \
            (geom['L1'] < velo[:, 1]) * (velo[:, 1] < geom['L2']) * \
            (geom['H1'] < velo[:, 2]) * (velo[:, 2] < geom['H2'])
        indices = np.where(q)[0]
        return velo[indices, :]

    def lidar_preprocess(self, scan):
        # TODO
        velo_processed = np.zeros(self.geometry['lidar_input_shape'], dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        velo = self.passthrough(scan)
        for i in range(velo.shape[0]):
            x = int((velo[i, 1] - self.geometry['L1']) / self.geometry['interval'])
            y = int((velo[i, 0] - self.geometry['W1']) / self.geometry['interval'])
            z = int((velo[i, 2] - self.geometry['H1']) / self.geometry['interval'])
            velo_processed[x, y, z] = 1
            velo_processed[x, y, -1] += velo[i, 3]
            intensity_map_count[x, y] += 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1], intensity_map_count,
                                             where=intensity_map_count != 0)
        return velo_processed

    def bev_to_velo(self, x, y, z):
        scales = self.geometry['lidar_input_shape'][0] / self.geometry['knn_shape'][0]
        l = (scales * x + 0.5) * self.geometry['interval'] + self.geometry['L1']
        w = (scales * y + 0.5) * self.geometry['interval'] + self.geometry['W1']
        h = (scales * z + 0.5) * self.geometry['interval'] + self.geometry['H1']
        return w, l, h

    def cal_index_bev(self, x, y, z):
        return y * (self.geometry['knn_shape'][0] * self.geometry['knn_shape'][2]) + x * self.geometry['knn_shape'][2] + z

    def cal_index_velo(self, w, l, h):
        scales = self.geometry['lidar_input_shape'][0] / self.geometry['knn_shape'][0]
        x = round(((l - self.geometry['L1']) / self.geometry['interval'] - 0.5) / scales)
        y = round(((w - self.geometry['W1']) / self.geometry['interval'] - 0.5) / scales)
        z = round(((h - self.geometry['H1']) / self.geometry['interval'] - 0.5) / scales)
        return self.cal_index_bev(x, y, z)

    def find_knn_image(self, calib, scan, point, k=1):
        point = point[:, 0:3]

        ''' 800/2 * 400/2 * 36/2 '''
        center = np.zeros([self.geometry['knn_shape'][0], self.geometry['knn_shape'][1], self.geometry['knn_shape'][2]])

        """ 相当于得到的是，bev表示的特征图中的每一个像素的位置。但是这个bev特征图的表示
        ，是多个通道的不像图片是三个通道的 """
        itemindex = np.argwhere(center == 0)
        itemindex = itemindex.astype(np.float32)

        ''' bev  800  700  37 '''
        """ interval = 0.1 """
        """ scales = 2 """
        scales = self.geometry['lidar_input_shape'][0] / self.geometry['knn_shape'][0]  # scales = 2
        """ x坐标设置 """
        itemindex[:, 0] = (scales * itemindex[:, 0] + 0.5) * self.geometry['interval'] + self.geometry['L1']
        """ y坐标设置 """
        itemindex[:, 1] = (scales * itemindex[:, 1] + 0.5) * self.geometry['interval'] + self.geometry['W1']
        """ z坐标设置 """
        itemindex[:, 2] = (scales * itemindex[:, 2] + 0.5) * self.geometry['interva     l'] + self.geometry['H1']

        itemindex = itemindex[:, [1, 0, 2]]
        #t = itemindex[:,0]
        #itemindex[:,0] = itemindex[:,1]
        #itemindex[:,1] = t
        """ 计算得到，刚才偏移计算时的中心位置，每一层的点 """
        center = np.reshape(itemindex, (self.geometry['knn_shape'][0], self.geometry['knn_shape'][1], self.geometry['knn_shape'][2], 3))
        size = center.shape  # 400,350,36,3

        try:
            import pcl
            # print ('itemindex', itemindex)
            itemindex = itemindex.astype(np.float32)
            # 使用pcl库，调用  knn算法查询  k近邻点
            pc_point = pcl.PointCloud(point)
            """ 虚拟的点，可能那个地方不存在真实的激光雷达采集到的点 """
            pc_center = pcl.PointCloud(itemindex)
            kd = pc_point.make_kdtree_flann()
            # find the single closest points to each point in point cloud 2
            # (and the sqr distances)
            # 以上述定义的  偏移中心点做  knn算法的中心，以此来查找k近邻个点的对应的坐标位置
            indices, _ = kd.nearest_k_search_for_cloud(pc_center, k)
            # print ('indices', indices.shape)
            # print ('point', point.shape)
            # print ('center', center.shape)

            indices = np.reshape(indices, (-1))
            k_nearest = point[indices]

            """ 在BEV表示的LiDAR模态中找到，根据指定的k（寻找几个中心点周围的最进的点）
            默认时5，那么下面的k_nearest的shape is (400,350,16,5,3)代表的含义就是：
            BEV体素表示的数据为16层，每一层相当于是图片的一层，因此每一层的H和W为400*350
            每一个中心点又存在k个最近的点，因此需要知道k个最近的点的坐标（这里的坐标是三维坐标）
            ，不要再考虑为什么不是（x，y）而是（x，y，z） """
            k_nearest = np.reshape(k_nearest, (size[0], size[1], size[2], k, size[3]))  # 400,350,36,5,3

            """ 将得到的紧邻点，映射到图片中 """
            k_nearest_image = self.velo_to_image(calib, k_nearest)  # (400,350,36,5,2)

            """ 代表的是检索到的最进的k个点和中心点坐标的位置的差距
            k_nearest shape is (400,350,16,k,3) """
            k_dif = k_nearest - center[:, :, :, np.newaxis, :]  # (400,350,36,5,3) - (400,350,36,1,3)   每一个 knn检索的点都减去中心点  k_dif size (bs,400,350,36,5,3)
        except:
            print('uninstall pcl')
            center = np.reshape(center, (size[0], size[1], size[2], 1, size[3]))
            k_nearest_image = self.velo_to_image(calib, center)
            k_dif = center - center
        return k_nearest_image, k_dif

    def velo_to_image(self, calib, point):
        size = point.shape
        point = np.reshape(point, (-1, 3))
        image, dis = calib.lidar_to_img(point)
        """ shape is 400 350 16 3 2 """
        image = np.reshape(image, (size[0], size[1], size[2], size[3], 2))
        return image

    def raw_to_tensor(self, point, image):
        calib = self.load_calib(0)
        scan = self.lidar_preprocess(point)
        bev2image, pc_diff = self.find_knn_image(calib, scan, point, k=1)

        image = torch.from_numpy(image)
        scan = torch.from_numpy(scan)
        bev2image = torch.from_numpy(bev2image)
        pc_diff = torch.from_numpy(pc_diff)
        image = image.float()
        bev2image = bev2image.float()
        pc_diff = pc_diff.float()
        image = image.permute(2, 0, 1)
        scan = scan.permute(2, 0, 1)
        return scan, image, bev2image, pc_diff
