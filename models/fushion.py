def bev_image_fusion(y, x2y, pc_diff):
    """  
    description: test for fusion in knn
    """
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
