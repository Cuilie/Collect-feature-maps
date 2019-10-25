from FixSeed import *
from simplecnn import *
from Settings import *
from Datasets import *
import os
import msvcrt


if __name__ == "__main__":
    device = torch.device("cuda:0")
    dataroot =  os.getcwd() + '\\data\\'
    modelroot = os.getcwd() + '\\model\\'


    '''
    Train SimpleCNN on MNIST
    Uncomment to retrain the CNN. After training, a checkpoint will be generated in ./root/model/.
    The evaluation code will use that checkpoint to evalute.
    '''
    # d = MNIST(dataroot = dataroot)
    # m = SimpleCNN().to(device)
    # s = SimpleCNN_Setting(data = d, model = m,modelroot = modelroot)
    # s.learning_rate = 0.0001
    # s.batch_size = 64
    # s.n_epochs = 8
    # s.run()


    '''
    After training, uncomment to evalute and collect the feature maps.
    After evaluation, a folder named FeatureMap will be generated in the root dir.
    Each folder in the FeatureMap collects all feature map with that label.
    '''
    # d = MNIST(dataroot = dataroot)
    # m = SimpleCNN_CollectFM().to(device)
    # s = SimpleCNN_CollectFM_Setting(data = d, model = m,modelroot = modelroot)
    # s.run()



    '''
    Train ResNet50 on CIFAR10
    '''
    ## d = ImageNet(dataroot = dataroot)


    d = CIFAR10(dataroot = dataroot)

    # ResNets50
    # m = ResNet(Bottleneck, [3, 4, 6, 3], num_classes = 10,input_channels=3).to(device)

    # ResNets18
    m = ResNet(block = ResidualBlock, layers = [2, 2, 2, 2], num_classes = 10,input_channels=3).to(device)
    s = ResNet_CIFAR10(data = d, model = m,modelroot = modelroot)
    s.learning_rate = 0.0001
    s.batch_size = 64
    s.n_epochs = 100
    s.run()
