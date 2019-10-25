import torch

if torch.cuda.is_available():

    print('cuda is available!')
    print('current device:',torch.cuda.current_device())
    print('number of device:',torch.cuda.device_count())
    print('device name:',torch.cuda.get_device_name(0))

else:
    print('cuda is not available!')
    print('Need to install:',torch.version.cuda)
