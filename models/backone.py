from models import conv3x3 as conv3x3
import torch.nn as nn


class ImageBackBone(nn.Module):

    def __init__(self, block, num_block, use_bn=True):
        super(ImageBackBone, self).__init__()

        self.use_bn = use_bn

        # Block 1
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # Block 2-5
        self.in_planes = 64
        self.block2 = self._make_layer(block, 48, num_blocks=num_block[0])
        self.block3 = self._make_layer(block, 64, num_blocks=num_block[1])
        self.block4 = self._make_layer(block, 96, num_blocks=num_block[2])
        self.block5 = self._make_layer(block, 128, num_blocks=num_block[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 384, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu2(x)
        #print ('x', x.shape)

        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        #print ('c1', c1.shape)
        #print ('c3', c3.shape)
        #print ('c5', c5.shape)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p4 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p3 = l3 + self.deconv2(p4)

        #print ('p4', p4.shape)
        #print ('p3', p3.shape)

        return p3

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * block.expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y


class LiDAR(nn.Module):

    def __init__(self, block, num_block, geom, use_bn=True):
        super(BevBackBone, self).__init__()

        self.use_bn = use_bn
        self.fusion = geom['fusion']
        self.geom = geom
        # Block 1
        """ input shpae is  512,
                            448,
                            33 """
        self.conv1 = conv3x3(self.geom['input_shape'][2], 32)
        self.conv2 = conv3x3(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        # Block 2-5
        self.in_planes = 32
        self.block2 = self._make_layer(block, 24, num_blocks=num_block[0])
        self.block3 = self._make_layer(block, 48, num_blocks=num_block[1])
        self.block4 = self._make_layer(block, 64, num_blocks=num_block[2])
        self.block5 = self._make_layer(block, 96, num_blocks=num_block[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=(1, 0))
        # MLP
        # 2096 = 16*(128+3)
        self.image_feature_dim = self.geom['knn_shape'][2] * (128 + 3)
        self.squeeze_fusion = nn.Conv2d(self.image_feature_dim, 256, kernel_size=1, stride=1)
        self.mlp2 = self._make_mlp(256, 96)
        self.mlp3 = self._make_mlp(256, 192)
        self.mlp4 = self._make_mlp(256, 256)
        self.mlp5 = self._make_mlp(256, 384)

    def forward(self, x, y, x2y, pc_diff):
        """  
        return p3
        ---------
        p3 for anchor-free
        """
        # x: bev input
        # y: image feature map 93, 310
        # x2y: knn input map 256, 224, k, 2
        # pc_diff: the diff between knn point and center  256, 224, k, 3
        """ bev input shape is (800,700,36) """
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu2(x)

        """ c1 为输入到网络中的bev数据，经过两层卷积计算后得到的结果。 """
        if self.fusion:
            image_feature = self.bev_image_fusion(y, x2y, pc_diff)

            # bottom up layers
            c2 = self.block2(c1) + self.mlp2(image_feature)
            #print ("m2", m2.size())
            #print ("c2", c2.size())
            c3 = self.block3(c2) + self.mlp3(image_feature[:, :, ::2, ::2])
            #print ("m3", m3.size())
            #print ("c3", c3.size())
            c4 = self.block4(c3) + self.mlp4(image_feature[:, :, ::4, ::4])
            #print ("m4", m4.size())
            #print ("c4", c4.size())
            c5 = self.block5(c4) + self.mlp5(image_feature[:, :, ::8, ::8])
            #print ("m5", m5.size())
            #print ("c5", c5.size())
        else:
            c2 = self.block2(c1)
            c3 = self.block3(c2)
            c4 = self.block4(c3)
            c5 = self.block5(c4)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p4 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p3 = l3 + self.deconv2(p4)

        return p3

    def bev_image_fusion(self, y, x2y, pc_diff):
        # y: image feature map -1, 128, 96, 312
        # x2y: knn input map -1, 256, 224, 16, k, 2
        # return: 256, 224, 256
        assert list(y.size())[1:] == [128, 96, 312]
        batch_size = y.size()[0]
        y = y.permute(0, 2, 3, 1)
        # TODO can be better
        x2y[:, :, :, :, :, 0] = torch.clamp(x2y[:, :, :, :, :, 0], 0, 96)
        x2y[:, :, :, :, :, 1] = torch.clamp(x2y[:, :, :, :, :, 1], 0, 312)
        x2y = torch.round(x2y)
        x2y = x2y.view(batch_size, -1, 2)
        x2y = x2y.long()

        y_bev = torch.Tensor()
        for i in range(batch_size):
            z = y[i, x2y[i, :, 0], x2y[i, :, 1]]
            z = torch.unsqueeze(z, 0)
            if y_bev.size()[0] == 0:
                y_bev = z
            else:
                y_bev = torch.cat((y_bev, z), 0)
        if True:
            # self.geom['knn_shape']
            ''' knn shape  400 350 18 '''
            """ y bev shape now is (bs,400,350,16,-1,128) 128要跟随图像特征通道进行计算 """
            y_bev = y_bev.view(batch_size, *self.geom['knn_shape'], -1, 128)

            y_bev = torch.cat((y_bev, pc_diff), 5)
            y_bev = torch.mean(y_bev, 4).squeeze(4)
            y_bev = y_bev.view(batch_size, *self.geom['knn_shape'][0:2], -1)
            y_bev = y_bev.permute(0, 3, 1, 2)
            y_bev = self.squeeze_fusion(y_bev)
        else:
            y_bev = y_bev.view(batch_size, *self.geom['knn_shape'][0:2], -1, 256)
            y_bev = torch.mean(y_bev, 3).squeeze(3)
            y_bev = y_bev.permute(0, 3, 1, 2)
        return y_bev

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * block.expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_mlp(self, channel_in, channel_out):
        mlp = nn.Sequential(
            nn.Conv2d(channel_in, channel_in // 2, kernel_size=1, stride=1),
            nn.Conv2d(channel_in // 2, channel_out, kernel_size=1, stride=1)
        )
        return mlp

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y
