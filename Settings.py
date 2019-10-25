import os
import time
import torch
import torchsummary
import torch.optim as optim
from torch.autograd import Variable


class Settings():
    def __init__(self, data = None, model = None):
        self.model = None
        self.data = None

    def run(self):
        return NotImplementedError


class SimpleCNN_CollectFM_Setting(Settings):

    def __init__(self, data = None, model = None,modelroot = None):
        self.data = data
        self.model = model
        self.modelroot = modelroot + 'SimpleCNN_CollectFM'
        self.learning_rate = 0.0001

    def createLossAndOptimizer(self):
        #Loss function
        loss = torch.nn.CrossEntropyLoss()
        #Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return(loss, optimizer)

    def run(self):
        # load model

        self.model.load_state_dict(torch.load(self.modelroot))
        self.model.eval()

        # print summary of modelroot
        torchsummary.summary(self.model, (1, 28, 28))

        # create optim
        loss, optimizer = self.createLossAndOptimizer()


        total_test_loss,num_correct,num_total = [0,0,0]

        if not os.path.exists('featuremap'):
            os.mkdir('featuremap')
        os.chdir("featuremap")


        for inputs, labels in self.data.test_loader:

            #enter to label dir
            dirname = str(labels.numpy()[0])
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            os.chdir(dirname)

            num_total += (inputs.shape[0])
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            #Forward pass
            test_outputs = self.model(inputs)
            test_loss_size = loss(test_outputs, labels)
            total_test_loss += test_loss_size.item()

            # convert output to predict label
            num_correct += torch.sum((labels.data == torch.argmax(test_outputs,axis = 1).data)).item()

            #return to work dir
            currentdir = os.path.dirname(os.getcwd())
            os.chdir(currentdir)

        print("Test acc = {:.2f} %".format(num_correct / num_total))


class SimpleCNN_Setting(Settings):

    def __init__(self, data = None, model = None,modelroot = None):
        self.data = data
        self.model = model
        self.modelroot = modelroot + 'SimpleCNN'
        self.learning_rate = None
        self.batch_size = None
        self.n_epochs = None


    def run(self):
        self.trainNet(self.model, self.batch_size, self.n_epochs, self.learning_rate, self.modelroot)



    def createLossAndOptimizer(self):
        #Loss function
        loss = torch.nn.CrossEntropyLoss()
        #Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return(loss, optimizer)


    def trainNet(self, net, batch_size, n_epochs, learning_rate, modelroot):
        #Print all of the hyperparameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("batch_size=", batch_size)
        print("epochs=", n_epochs)
        print("learning_rate=", learning_rate)
        print("=" * 30)

        # print summary of modelroot
        torchsummary.summary(self.model, (1, 28, 28))

        #Get training data
        train_loader = self.data.get_train_loader(batch_size)
        n_batches = len(train_loader)

        #Create our loss and optimizer functions
        loss, optimizer = self.createLossAndOptimizer()

        #Time for printing
        training_start_time = time.time()

        #Loop for n_epochs
        for epoch in range(n_epochs):
            running_loss = 0.0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            for i, data in enumerate(train_loader, 0):
                #Get inputs
                inputs, labels = data

                #Wrap them in a Variable object
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                #Set the parameter gradients to zero
                optimizer.zero_grad()

                #Forward pass, backward pass, optimize
                outputs = net(inputs)
                loss_size = loss(outputs, labels)


                loss_size.backward()
                optimizer.step()

                #Print statistics
                running_loss += loss_size.item()
                total_train_loss += loss_size.item()

                #Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.5f} took: {:.2f}s".format(
                            epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()

            # Initialize the value of loss and acc
            total_val_loss,num_correct,num_total = [0,0,0]

            #At the end of the epoch, do a pass on the validation set
            for inputs, labels in self.data.val_loader:
                num_total += (inputs.shape[0])

                #Wrap tensors in Variables
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                #Forward pass
                val_outputs = net(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.item()

                # convert output to predict label
                num_correct += torch.sum((labels.data == torch.argmax(val_outputs,axis = 1).data)).item()

            print("Validation loss = {:.5f}".format(total_val_loss / num_total))
            print("Validation acc = {:.2f} %".format(num_correct / num_total))

            # save the The model
            print(modelroot)
            torch.save(net.state_dict(), modelroot)
        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))





class ResNet_CIFAR10(Settings):

    def __init__(self, data = None, model = None,modelroot = None):
        self.data = data
        self.model = model
        self.modelroot = modelroot + 'ResNet'
        self.learning_rate = None
        self.batch_size = None
        self.n_epochs = None


    def run(self):
        self.trainNet(self.model, self.batch_size, self.n_epochs, self.learning_rate, self.modelroot)



    def createLossAndOptimizer(self):
        #Loss function
        loss = torch.nn.CrossEntropyLoss()
        #Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return(loss, optimizer)


    def trainNet(self, net, batch_size, n_epochs, learning_rate, modelroot):
        #Print all of the hyperparameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("batch_size=", batch_size)
        print("epochs=", n_epochs)
        print("learning_rate=", learning_rate)
        print("=" * 30)

        # print summary of modelroot
        torchsummary.summary(self.model, (3, 32, 32))

        #Get training data
        train_loader = self.data.get_train_loader(batch_size)
        n_batches = len(train_loader)

        #Create our loss and optimizer functions
        loss, optimizer = self.createLossAndOptimizer()

        #Time for printing
        training_start_time = time.time()

        #Loop for n_epochs
        for epoch in range(n_epochs):
            running_loss = 0.0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            for i, data in enumerate(train_loader, 0):

                #Get inputs
                inputs, labels = data

                #Wrap them in a Variable object
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                #Set the parameter gradients to zero
                optimizer.zero_grad()

                #Forward pass, backward pass, optimize
                outputs = net(inputs)
                loss_size = loss(outputs, labels)


                loss_size.backward()
                optimizer.step()

                #Print statistics
                running_loss += loss_size.item()
                total_train_loss += loss_size.item()

                #Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.5f} took: {:.2f}s".format(
                            epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()

            # Initialize the value of loss and acc
            total_val_loss,num_correct,num_total = [0,0,0]

            #At the end of the epoch, do a pass on the validation set
            for inputs, labels in self.data.val_loader:
                num_total += (inputs.shape[0])

                #Wrap tensors in Variables
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                #Forward pass
                val_outputs = net(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.item()

                # convert output to predict label
                num_correct += torch.sum((labels.data == torch.argmax(val_outputs,axis = 1).data)).item()

            print("Validation loss = {:.5f}".format(total_val_loss / num_total))
            print("Validation acc = {:.2f} %".format(num_correct / num_total))

            # save the The model
            print(modelroot)
            torch.save(net.state_dict(), modelroot)
        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
