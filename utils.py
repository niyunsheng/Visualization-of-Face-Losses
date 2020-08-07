from torch import nn

def weight_init(m):
    if isinstance(m, nn.Linear):
        # print('linear',m)
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # print('conv',m)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        # print('bn',m)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        # print('parameter',m)
        nn.init.xavier_normal_(m)
