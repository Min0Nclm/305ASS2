import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ======================================================================================
# A. Helper class for logging
# ======================================================================================
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# ======================================================================================
# B. ResNet-50 Model Definition
# ======================================================================================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None: replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3: raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        ## origin: self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        ## arc1 change 7*7 kernel to 3*3 kernel then we can get 2*2 in conv5
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        ## origin: self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ## delete maxpooling then conv1 -> conv2 32*32 -> 32*32
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck): nn.init.constant_(m.bn3.weight, 0)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate: self.dilation *= stride; stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []; layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

# ======================================================================================
# C. Helper functions for reporting
# ======================================================================================
def plot_training_curves(loss_history, train_accuracy_history, val_accuracy_history, output_dir):
    print("Generating training curve plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(loss_history)
    ax1.set_title("Training Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(train_accuracy_history, label="Train Accuracy")
    ax2.plot(val_accuracy_history, label="Validation Accuracy")
    ax2.set_title("Train vs. Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, "resnet_training_curves.png")
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")

def visualize_and_save_predictions(net, testloader, classes, device, output_dir):
    print("Generating prediction images for the report...")
    net.eval()
    well_classified_examples, misclassified_examples = [], []
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    probabilities = F.softmax(outputs, dim=1)
    confidences = [p[predicted[i]].item() for i, p in enumerate(probabilities)]
    for i in range(images.size(0)):
        example = {"image": images[i].cpu(), "true_label": classes[labels[i]], "predicted_label": classes[predicted[i]], "confidence": confidences[i]}
        if predicted[i] == labels[i]:
            if len(well_classified_examples) < 5: well_classified_examples.append(example)
        else:
            if len(misclassified_examples) < 5: misclassified_examples.append(example)
    _plot_image_examples(well_classified_examples, "Well Classified Examples", "resnet_well_classified.png", output_dir)
    _plot_image_examples(misclassified_examples, "Misclassified Examples", "resnet_misclassified.png", output_dir)
    print("Saved classification example images.")

def _plot_image_examples(examples, title, filename, output_dir):
    if not examples: print(f"No examples found for '{title}'"); return
    fig, axes = plt.subplots(1, len(examples), figsize=(15, 3))
    if len(examples) == 1: axes = [axes]
    fig.suptitle(title, fontsize=16)
    for i, example in enumerate(examples):
        img = example["image"] / 2 + 0.5
        npimg = img.numpy()
        ax = axes[i]; ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(f"True: {example['true_label']}\nPred: {example['predicted_label']}\nConf: {example['confidence']:.2f}")
        ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)

def calculate_accuracy(loader, net, device):
    correct = 0; total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data; images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def adjust_learning_rate(optimizer, epoch, initial_lr, warmup_epochs, milestones, gamma):
    """Sets the learning rate for the given epoch with warmup and multi-step decay."""
    if epoch < warmup_epochs:
        # Linear warmup
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Multi-step decay
        lr = initial_lr
        for milestone in milestones:
            if epoch >= milestone:
                lr *= gamma
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# ======================================================================================
# D. Main Workflow for Training and Evaluation
# ======================================================================================
def main(output_dir):
    # Hyperparameters
    NUM_EPOCHS = 200
    BATCH_SIZE = 128
    INITIAL_LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    # LR Scheduler and Warmup settings
    WARMUP_EPOCHS = 5
    LR_MILESTONES = [60, 120, 160]
    LR_GAMMA = 0.2  # Divide by 5

    # Data transformation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Setup device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net = resnet50(num_classes=10)
    net.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # --- Training Loop ---
    print('Start Training')
    start_time = time.time()
    loss_history, train_accuracy_history, val_accuracy_history = [], [], []
    
    for epoch in range(NUM_EPOCHS):
        # Adjust learning rate for the current epoch
        current_lr = adjust_learning_rate(optimizer, epoch, INITIAL_LR, WARMUP_EPOCHS, LR_MILESTONES, LR_GAMMA)
        
        net.train()
        running_loss = 0.0
        for data in trainloader:
            inputs, labels = data; inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        loss_history.append(epoch_loss)
        
        net.eval()
        train_acc = calculate_accuracy(trainloader, net, device)
        val_acc = calculate_accuracy(testloader, net, device)
        train_accuracy_history.append(train_acc)
        val_accuracy_history.append(val_acc)
        
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS} - LR: {current_lr:.5f} - Loss: {epoch_loss:.3f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%')

    end_time = time.time()
    print('End Training')
    print(f'The time cost for training process is: {end_time - start_time:.2f} seconds')

    # --- Final Performance Report ---
    print('\n--- Final Performance Report ---')
    print(f'Final Validation Accuracy: {val_accuracy_history[-1]:.2f} %')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data; images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        if class_total[i] > 0:
            print(f'Accuracy of {classes[i]:>5s} : {100 * class_correct[i] / class_total[i]:2.0f} %')

    plot_training_curves(loss_history, train_accuracy_history, val_accuracy_history, output_dir)
    visualize_and_save_predictions(net, testloader, classes, device, output_dir)

if __name__ == '__main__':
    # Create a directory for results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("results", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Tee stdout to a log file
    original_stdout = sys.stdout
    log_file_path = os.path.join(output_dir, 'resnet_report.txt')
    with open(log_file_path, 'w') as f:
        sys.stdout = Tee(original_stdout, f)
        main(output_dir) # Pass the output directory to main
    sys.stdout = original_stdout
    print(f"\nScript finished. All outputs saved to '{output_dir}'.")
