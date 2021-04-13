import torch as tc
from datasets import CIFAR10ZagoruykoPreprocessing
from classifier import Cifar10PreactivationResNet
from runner import Runner
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10 with preprocessing as described in Section 3 of Zagoruyko and Komodakis, 2016.
training_data = CIFAR10ZagoruykoPreprocessing(root="data", train=True)
test_data = CIFAR10ZagoruykoPreprocessing(root="data", train=False)

# Create data loaders.
batch_size = 128
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# The WRN-40-4 model from Zagoruyko and Komodakis, applied to CIFAR-10.
model = Cifar10PreactivationResNet(
    img_height=32, img_width=32, img_channels=3,
    initial_num_filters=16*4, num_stages=3, num_repeats=13, num_classes=10).to(device)
print(model)

criterion = tc.nn.CrossEntropyLoss()
optimizer = tc.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0.0, weight_decay=0.0005, nesterov=True)
scheduler = tc.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.20)

try:
    model.load_state_dict(tc.load("model.pth"))
    optimizer.load_state_dict(tc.load("optimizer.pth"))
    scheduler.load_state_dict(tc.load("scheduler.pth"))
    print('successfully reloaded checkpoint. continuing training...')
except Exception:
    print('no checkpoint found. training from scratch...')

runner = Runner(max_epochs=200, verbose=True)
runner.run(model, train_dataloader, test_dataloader, device, criterion, optimizer, scheduler)
print("Done!")


model.eval() # turn batchnorm and dropout off.
for X, y in test_dataloader:
    input_example = X[0]
    input_label = y[0]

    x_features = model.visualize(
        tc.unsqueeze(input_example, dim=0))

    y_pred = tc.nn.Softmax()(model(tc.unsqueeze(input_example, dim=0)))

    print('ground truth label: {}'.format(input_label))
    print('predicted label distribution: {}'.format(y_pred))
    print(x_features)

    plt.imshow(np.transpose(input_example, [1,2,0]))
    plt.show()

    break