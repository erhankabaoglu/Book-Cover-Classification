import string
from nltk.corpus import stopwords
import nltk
import h5py
import time
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from numpy import save, load
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class bookCoverDataset(Dataset):

    def __init__(self, csv_file, h5_root_dir, transform=None, word_to_idx=None, cnn=True):

        self.cnn = cnn
        self.word_to_idx = word_to_idx
        self.bookCover_csv = pd.read_csv(csv_file)
        self.h5_root_dir = h5_root_dir
        self.transform = transform
        self.image_dict = {}
        if self.cnn:
            with h5py.File(self.h5_root_dir, 'r') as file:
                for image_name in file.keys():
                    self.image_dict[image_name] = file[image_name][()]

    def __len__(self):
        return len(self.bookCover_csv)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.cnn:
            img_name = self.bookCover_csv.iloc[idx, 1]
            image = self.image_dict[img_name]
            category_id = self.bookCover_csv.iloc[idx, 5]
            title = self.bookCover_csv.iloc[idx, 3]
            bow_vector = make_bow_vector(title, self.word_to_idx)
            sample = {'image': image, 'category_id': category_id, 'bow_vector': bow_vector}
            if self.transform:
                sample['image'] = self.transform(sample['image'])
        else:
            title = self.bookCover_csv.iloc[idx, 3]
            category_id = self.bookCover_csv.iloc[idx, 5]
            sample = {'title': title, 'category_id': category_id}

        return sample


def bookDataLoader(batch_size, input_size, word_to_idx=None, cnn=True):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = bookCoverDataset('/content/drive/My Drive/book15-listing-train.csv',
                                     '/content/drive/My Drive/train-bookCover.h5',
                                     transform=data_transforms['train'], word_to_idx=word_to_idx, cnn=cnn)
    validation_dataset = bookCoverDataset('/content/drive/My Drive/book15-listing-validation.csv',
                                          '/content/drive/My Drive/validation-bookCover.h5',
                                          transform=data_transforms['validation'], word_to_idx=word_to_idx, cnn=cnn)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    dataloaders = {'train': train_loader, 'validation': validation_loader}

    return dataloaders


def bookTestDataLoader(batch_size, input_size, word_to_idx=None, cnn=True):
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    test_dataset = bookCoverDataset('/content/drive/My Drive/book15-listing-test.csv',
                                    '/content/drive/My Drive/test-bookCover.h5',
                                    transform=data_transforms['test'], word_to_idx=word_to_idx, cnn=cnn)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

    return test_loader


"""*************************************************************** CNN"""


def set_parameter_requires_grad(model, model_name, feature_extracting):
    if feature_extracting == False:
        return

    if model_name == "resnet18":
        for i, child in enumerate(model.children()):
            if i < 8:
                for param in child.parameters():
                    param.requires_grad = False


    elif model_name == "resnet50":
        for i, child in enumerate(model.children()):
            if i < 8:
                for param in child.parameters():
                    param.requires_grad = False

    elif model_name == "resnet152":
        for i, child in enumerate(model.children()):
            if i < 8:
                for param in child.parameters():
                    param.requires_grad = False

    elif model_name == "resnetX-101":
        for i, child in enumerate(model.children()):
            if i < 8:
                for param in child.parameters():
                    param.requires_grad = False

    elif model_name == "alexnet":
        for i, child in enumerate(model.children()):
            if i < 2:
                for param in child.parameters():
                    param.requires_grad = False

    else:
        for i, child in enumerate(model.children()):
            if i < 2:
                for param in child.parameters():
                    param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, model_name, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, model_name, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, model_name, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnetX-101":
        """ Resnet152
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, model_name, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, model_name, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, model_name, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


"""***************************************************************"""


class BookCNNbf(nn.Module):
    def __init__(self, vocab_size, model_name, num_classes, feature_extract, drop=0.0, use_pretrained=True):
        super(BookCNNbf, self).__init__()
        self.model_ft = None
        self.model_name = model_name
        self.input_size = 0
        self.resFc = None
        self.VGG_features = None
        self.VGG_avgpool = None
        self.VGG_classifier = None
        self.Alexnet_features = None
        self.Alexnet_avgpool = None
        self.Alexnet_classifier = None

        if self.model_name == "resnet18":
            """ Resnet18
            """
            self.model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(self.model_ft, model_name, feature_extract)
            in_features = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Identity()
            self.resFc = nn.Linear(in_features + vocab_size, num_classes)
            self.input_size = 224

        elif self.model_name == "resnet50":
            """ Resnet50
            """
            self.model_ft = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(self.model_ft, model_name, feature_extract)
            in_features = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Identity()
            self.resFc = nn.Linear(in_features + vocab_size, num_classes)
            self.input_size = 224

        elif self.model_name == "resnet152":
            """ Resnet152
            """
            self.model_ft = models.resnet152(pretrained=use_pretrained)
            set_parameter_requires_grad(self.model_ft, model_name, feature_extract)
            in_features = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Identity()
            self.resFc = nn.Linear(in_features + vocab_size, num_classes)
            self.input_size = 224

        elif self.model_name == "resnetX-101":
            """ Resnet152
            """
            self.model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
            set_parameter_requires_grad(self.model_ft, model_name, feature_extract)
            in_features = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Identity()
            self.resFc = nn.Linear(in_features + vocab_size, num_classes)
            self.input_size = 224

        elif self.model_name == "alexnet":
            """ Alexnet
            """
            self.model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(self.model_ft, model_name, feature_extract)
            self.Alexnet_features = self.model_ft.features
            self.Alexnet_avgpool = self.model_ft.avgpool
            self.Alexnet_classifier = self.model_ft.classifier
            in_features = self.Alexnet_classifier[1].in_features
            out_features = self.Alexnet_classifier[1].out_features
            self.Alexnet_classifier[1] = nn.Linear(in_features + vocab_size, out_features)
            num_features = self.Alexnet_classifier[6].in_features
            self.Alexnet_classifier[6] = nn.Linear(num_features, num_classes)
            self.input_size = 224

        elif self.model_name == "vgg":
            """ VGG11_bn
            """
            self.model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(self.model_ft, model_name, feature_extract)
            self.VGG_features = self.model_ft.features
            self.VGG_avgpool = self.model_ft.avgpool
            self.VGG_classifier = self.model_ft.classifier
            in_features = self.VGG_classifier[0].in_features
            out_features = self.VGG_classifier[0].out_features
            self.VGG_classifier[0] = nn.Linear(in_features + vocab_size, out_features)
            num_features = self.VGG_classifier[6].in_features
            self.VGG_classifier[6] = nn.Linear(num_features, num_classes)
            self.input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

    def forward(self, x, bow_feature_vector):
        if self.model_name == "resnet18":
            x = self.model_ft(x)
            x = x.view(x.size(0), -1)
            y = bow_feature_vector.view(bow_feature_vector.size(0), -1)
            con = torch.cat((x, y), dim=1)
            x = self.resFc(con)
            return x

        elif self.model_name == "resnet50":
            x = self.model_ft(x)
            x = x.view(x.size(0), -1)
            y = bow_feature_vector.view(bow_feature_vector.size(0), -1)
            con = torch.cat((x, y), dim=1)
            x = self.resFc(con)
            return x

        elif self.model_name == "resnet152":
            x = self.model_ft(x)
            x = x.view(x.size(0), -1)
            y = bow_feature_vector.view(bow_feature_vector.size(0), -1)
            con = torch.cat((x, y), dim=1)
            x = self.resFc(con)
            return x

        elif self.model_name == "resnetX-101":
            x = self.model_ft(x)
            x = x.view(x.size(0), -1)
            y = bow_feature_vector.view(bow_feature_vector.size(0), -1)
            con = torch.cat((x, y), dim=1)
            x = self.resFc(con)
            return x

        elif self.model_name == "alexnet":
            x = self.Alexnet_features(x)
            x = self.Alexnet_avgpool(x)
            x = x.view(x.size(0), -1)
            y = bow_feature_vector.view(bow_feature_vector.size(0), -1)
            con = torch.cat((x, y), dim=1)
            x = self.Alexnet_classifier(con)
            return x

        else:
            x = self.VGG_features(x)
            x = self.VGG_avgpool(x)
            x = x.view(x.size(0), -1)
            y = bow_feature_vector.view(bow_feature_vector.size(0), -1)
            con = torch.cat((x, y), dim=1)
            x = self.VGG_classifier(con)
            return x


"""*************************************************************** Bow"""


class BOWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size, drop=0.0):
        super(BOWClassifier, self).__init__()
        self.lin = nn.Linear(vocab_size, num_labels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(x)
        return self.lin(x)


def cleanTitle(title):
    tokens = word_tokenize(title)  # tokenize
    tokens = [w.lower() for w in tokens]  # lowe case
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]  # remove Punctuation
    words = [word for word in stripped if word.isalpha()]  # remove not alphabetic
    stop_words = stopwords.words()
    words = [w for w in words if not w in stop_words]  # remove stopwords
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]  # stemming

    return words


def createVocabulary(file_name, word_to_idx):
    bookCover_train = pd.read_csv(file_name)

    for title in bookCover_train.iloc[:, 3]:
        words = cleanTitle(title)
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)


def make_bow_vector(title, word_to_ix):
    vec = torch.zeros(len(word_to_ix))

    words = cleanTitle(title)
    for word in words:
        if word not in word_to_ix:
            raise ValueError('we have a problem')
        else:
            vec[word_to_ix[word]] += 1
    return vec


"""***************************************************************"""


def calculateTopK(outputs, labels, k):
    _, pred = torch.topk(outputs, k=k, dim=1)
    return torch.sum(torch.tensor([1 for i in range(len(labels)) if labels[i] in pred[i]]), dtype=torch.long)


def testModel(model_path, vocab_size, model_name, dataloader, save_path):
    classes = ("Arts & Photography", "Biographies & Memoirs", "Business & Money", "Calendars", "Children's Books",
               "Comics & Graphic Novels", "Computers & Technology", "Cookbooks, Food & Wine", "Crafts, Hobbies & Home",
               "Christian Books & Bibles", "Engineering & Transportation", "Health, Fitness & Dieting", "History",
               "Humor & Entertainment", "Law")
    model = BookCNNbf(vocab_size, model_name, 15, True, False)
    model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    conf_true = []
    conf_pred = []

    with torch.no_grad():
        running_corrects = 0.0
        for sample in dataloader:
            input_images, bow_vector, labels = sample['image'].to(device), \
                                               sample['bow_vector'].to(device, dtype=torch.float), \
                                               sample['category_id'].to(device)
            outputs = model(input_images, bow_vector)

            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

            conf_true[len(conf_true):] = labels.cpu().numpy()
            conf_pred[len(conf_pred):] = preds.cpu().numpy()

        epoch_acc = running_corrects.double() / len(dataloader.dataset)

    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    conf_matrix = confusion_matrix(conf_true, conf_pred)
    print(conf_matrix)
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in classes], columns=[i for i in classes])
    sn.set(font_scale=1.0)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig(save_path + '.jpg')
    plt.show()
    print('Acc: {:.4f}'.format(epoch_acc * 100))

    return epoch_acc


def plotFunction(train_acc_path, train_loss_path, val_acc_path, val_loss_path, save_path, method):
    for i, key in enumerate(train_acc_path):
        plt.plot(np.load(train_acc_path[key]), label=key + " Train")
    for i, key in enumerate(val_acc_path):
        plt.plot(np.load(val_acc_path[key]), label=key + " Val")

    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy (%)')
    plt.title(method)
    plt.legend(loc='best', prop={'size': 8})
    plt.savefig(save_path[0])
    plt.show()

    for i, key in enumerate(train_loss_path):
        plt.plot(np.load(train_loss_path[key]), label=key + " Train")
    for i, key in enumerate(val_loss_path):
        plt.plot(np.load(val_loss_path[key]), label=key + " Val")

    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title(method)
    plt.legend(loc='best', prop={'size': 8})
    plt.savefig(save_path[1])
    plt.show()


def train_model_late(model_cnn, model_name, model_bow, dataloaders, criterion, optimizer, save_path, num_epochs=25,
                     lr_sch=None):
    val_acc_history = []
    train_loss = []
    train_accuracy = []
    val_loss = []
    top_values = []

    best_model_wts_cnn = copy.deepcopy(model_cnn.state_dict())
    best_model_wts_bow = copy.deepcopy(model_bow.state_dict())
    best_acc_top1 = 0.0
    best_acc_top2 = 0.0
    best_acc_top3 = 0.0

    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model_cnn.train()  # Set model to training mode
                model_bow.train()
            else:
                model_cnn.eval()  # Set model to evaluate mode
                model_bow.eval()

            running_loss = 0.0
            running_corrects_top1 = 0
            running_corrects_top2 = 0
            running_corrects_top3 = 0
            running_corrects_cnn = 0
            running_corrects_bow = 0

            # Iterate over data.
            for sample in dataloaders[phase]:
                input_images, bow_vector, labels = sample['image'].to(device, dtype=torch.float), \
                                                   sample['bow_vector'].to(device, dtype=torch.float), \
                                                   sample['category_id'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs_cnn = model_cnn(input_images)
                    outputs_bow = model_bow(bow_vector)
                    outputs = outputs_cnn + outputs_bow

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    _, preds_cnn = torch.max(outputs_cnn, 1)
                    _, preds_bow = torch.max(outputs_bow, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * input_images.size(0)
                running_corrects_top1 += calculateTopK(outputs, labels, k=1)
                running_corrects_top2 += calculateTopK(outputs, labels, k=2)
                running_corrects_top3 += calculateTopK(outputs, labels, k=3)
                running_corrects_cnn += torch.sum(preds_cnn == labels.data)
                running_corrects_bow += torch.sum(preds_bow == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc_top1 = running_corrects_top1.double() / len(dataloaders[phase].dataset)
            epoch_acc_top2 = running_corrects_top2.double() / len(dataloaders[phase].dataset)
            epoch_acc_top3 = running_corrects_top3.double() / len(dataloaders[phase].dataset)
            epoch_acc_cnn = running_corrects_cnn.double() / len(dataloaders[phase].dataset)
            epoch_acc_bow = running_corrects_bow.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f} Acc cnn: {:.4f} Acc bow: {:.4f}'.format(phase, epoch_loss,
                                                                                       epoch_acc_top1 * 100,
                                                                                       epoch_acc_cnn * 100,
                                                                                       epoch_acc_bow * 100))

            # deep copy the model
            if phase == 'validation' and epoch_acc_top1 > best_acc_top1:
                best_acc_top1 = epoch_acc_top1
                best_acc_top2 = epoch_acc_top2
                best_acc_top3 = epoch_acc_top3
                best_model_wts_cnn = copy.deepcopy(model_cnn.state_dict())
                best_model_wts_bow = copy.deepcopy(model_bow.state_dict())

            if phase == 'validation':
                val_acc_history.append(epoch_acc_top1 * 100)
                val_loss.append(epoch_loss)
            if phase == 'train':
                train_accuracy.append(epoch_acc_top1 * 100)
                train_loss.append(epoch_loss)
        if lr_sch:
            lr_sch.step()
        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc_top1 * 100))

    top_values.append(best_acc_top1)
    top_values.append(best_acc_top2)
    top_values.append(best_acc_top3)

    # load best model weights
    model_cnn.load_state_dict(best_model_wts_cnn)
    model_bow.load_state_dict(best_model_wts_bow)
    torch.save([best_model_wts_cnn, best_model_wts_bow], save_path)
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Late trainloss.npy", np.array(train_loss))
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Late val_loss.npy", np.array(val_loss))
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Late valAccuracy.npy", np.array(val_acc_history))
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Late TrainAccuracy.npy", np.array(train_accuracy))
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Late top_values.npy", np.array(top_values))

    return model_cnn, model_bow


def train_model_early(model, model_name, dataloaders, criterion, optimizer, save_path, num_epochs=25, lr_sch=None):
    val_acc_history = []
    train_loss = []
    train_accuracy = []
    val_loss = []
    top_values = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_top1 = 0.0
    best_acc_top2 = 0.0
    best_acc_top3 = 0.0

    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects_top1 = 0
            running_corrects_top2 = 0
            running_corrects_top3 = 0

            # Iterate over data.
            for sample in dataloaders[phase]:
                input_images, bow_vector, labels = sample['image'].to(device, dtype=torch.float), \
                                                   sample['bow_vector'].to(device, dtype=torch.float), \
                                                   sample['category_id'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(input_images, bow_vector)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * input_images.size(0)
                running_corrects_top1 += calculateTopK(outputs, labels, k=1)
                running_corrects_top2 += calculateTopK(outputs, labels, k=2)
                running_corrects_top3 += calculateTopK(outputs, labels, k=3)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc_top1 = running_corrects_top1.double() / len(dataloaders[phase].dataset)
            epoch_acc_top2 = running_corrects_top2.double() / len(dataloaders[phase].dataset)
            epoch_acc_top3 = running_corrects_top3.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Top 1 Acc: {:.4f} Top 2 Acc: {:.4f} Top 3 Acc: {:.4f}'.format(phase, epoch_loss,
                                                                                                 epoch_acc_top1 * 100,
                                                                                                 epoch_acc_top2 * 100,
                                                                                                 epoch_acc_top3 * 100))

            # deep copy the model
            if phase == 'validation' and epoch_acc_top1 > best_acc_top1:
                best_acc_top1 = epoch_acc_top1
                best_acc_top2 = epoch_acc_top2
                best_acc_top3 = epoch_acc_top3
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'validation':
                val_acc_history.append(epoch_acc_top1 * 100)
                val_loss.append(epoch_loss)
            if phase == 'train':
                train_accuracy.append(epoch_acc_top1 * 100)
                train_loss.append(epoch_loss)
        if lr_sch:
            lr_sch.step()
        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc_top1 * 100))

    top_values.append(best_acc_top1)
    top_values.append(best_acc_top2)
    top_values.append(best_acc_top3)

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, save_path)
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Early trainloss.npy", np.array(train_loss))
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Early val_loss.npy", np.array(val_loss))
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Early valAccuracy.npy", np.array(val_acc_history))
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Early TrainAccuracy.npy", np.array(train_accuracy))
    save("/content/drive/My Drive/ProjeLateEarly/" + model_name + "Early top_values.npy", np.array(top_values))

    return model


word_to_idx = {}
createVocabulary('/content/drive/My Drive/book15-listing-train.csv', word_to_idx)
createVocabulary('/content/drive/My Drive/book15-listing-validation.csv', word_to_idx)
createVocabulary('/content/drive/My Drive/book15-listing-test.csv', word_to_idx)
"""**********************************************************************************************"""


def model_late(model_name):
    num_epochs = 25
    batch_size = 16
    num_classes = 15
    feature_extract = True
    save_path = "/content/drive/My Drive/ProjeLateEarly/" + model_name + "late" + ".pth"

    print("Checkpoint 1")

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract)
    model_ft.to(device)
    bow = BOWClassifier(num_classes, len(word_to_idx), drop=0.5)
    bow.to(device)

    dataloaders = bookDataLoader(batch_size, input_size, word_to_idx)

    print("Checkpoint 2")

    optimizer_ft = optim.Adam(list(filter(lambda p: p.requires_grad, model_ft.parameters())) + list(bow.parameters()))

    lr_sch = lr_scheduler.StepLR(optimizer_ft, 10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    model_ft_cnn, model_ft_bow = train_model_late(model_ft, model_name, bow, dataloaders, criterion, optimizer_ft,
                                                  save_path,
                                                  num_epochs=num_epochs, lr_sch=lr_sch)


def model_early(model_name):
    num_epochs = 25
    batch_size = 16
    num_classes = 15
    feature_extract = True
    save_path = "/content/drive/My Drive/ProjeLateEarly/" + model_name + "early" + ".pth"

    print("Checkpoint 1")

    model = BookCNNbf(len(word_to_idx), model_name, num_classes, feature_extract)
    model.to(device)

    dataloaders = bookDataLoader(batch_size, 224, word_to_idx)

    print("Checkpoint 2")

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    lr_sch = lr_scheduler.StepLR(optimizer_ft, 10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    model_ft = train_model_early(model, model_name, dataloaders, criterion, optimizer_ft, save_path, num_epochs, lr_sch)


"""plotFunction({"alexnet":"/content/drive/My Drive/ProjeLateEarly/alexnetEarly TrainAccuracy.npy", "vgg":"/content/drive/My Drive/ProjeLateEarly/vggEarly TrainAccuracy.npy", "resnet18":"/content/drive/My Drive/ProjeLateEarly/resnet18Early TrainAccuracy.npy", "resnet50":"/content/drive/My Drive/ProjeLateEarly/resnet50Early TrainAccuracy.npy", "resnet152":"/content/drive/My Drive/ProjeLateEarly/resnet152Early TrainAccuracy.npy"},
             {"alexnet":"/content/drive/My Drive/ProjeLateEarly/alexnetEarly trainloss.npy", "vgg":"/content/drive/My Drive/ProjeLateEarly/vggEarly trainloss.npy", "resnet18":"/content/drive/My Drive/ProjeLateEarly/resnet18Early trainloss.npy", "resnet50":"/content/drive/My Drive/ProjeLateEarly/resnet50Early trainloss.npy", "resnet152":"/content/drive/My Drive/ProjeLateEarly/resnet152Early trainloss.npy"},
             {"alexnet":"/content/drive/My Drive/ProjeLateEarly/alexnetEarly valAccuracy.npy", "vgg":"/content/drive/My Drive/ProjeLateEarly/vggEarly valAccuracy.npy", "resnet18":"/content/drive/My Drive/ProjeLateEarly/resnet18Early valAccuracy.npy", "resnet50":"/content/drive/My Drive/ProjeLateEarly/resnet50Early valAccuracy.npy", "resnet152":"/content/drive/My Drive/ProjeLateEarly/resnet152Early valAccuracy.npy"},
             {"alexnet":"/content/drive/My Drive/ProjeLateEarly/alexnetEarly val_loss.npy", "vgg":"/content/drive/My Drive/ProjeLateEarly/vggEarly val_loss.npy", "resnet18":"/content/drive/My Drive/ProjeLateEarly/resnet18Early val_loss.npy", "resnet50":"/content/drive/My Drive/ProjeLateEarly/resnet50Early val_loss.npy", "resnet152":"/content/drive/My Drive/ProjeLateEarly/resnet152Early val_loss.npy"},
             ["/content/drive/My Drive/ProjeLateEarly/Early Accuracy.jpg", "/content/drive/My Drive/ProjeLateEarly/Early Loss.jpg"], "Bow Feature Vector")


plotFunction({"alexnet":"/content/drive/My Drive/ProjeLateEarly/alexnetLate TrainAccuracy.npy", "vgg":"/content/drive/My Drive/ProjeLateEarly/vggLate TrainAccuracy.npy", "resnet18":"/content/drive/My Drive/ProjeLateEarly/resnet18Late TrainAccuracy.npy", "resnet50":"/content/drive/My Drive/ProjeLateEarly/resnet50Late TrainAccuracy.npy", "resnet152":"/content/drive/My Drive/ProjeLateEarly/resnet152Late TrainAccuracy.npy"},
             {"alexnet":"/content/drive/My Drive/ProjeLateEarly/alexnetLate trainloss.npy", "vgg":"/content/drive/My Drive/ProjeLateEarly/vggLate trainloss.npy", "resnet18":"/content/drive/My Drive/ProjeLateEarly/resnet18Late trainloss.npy", "resnet50":"/content/drive/My Drive/ProjeLateEarly/resnet50Late trainloss.npy", "resnet152":"/content/drive/My Drive/ProjeLateEarly/resnet152Late trainloss.npy"},
             {"alexnet":"/content/drive/My Drive/ProjeLateEarly/alexnetLate valAccuracy.npy", "vgg":"/content/drive/My Drive/ProjeLateEarly/vggLate valAccuracy.npy", "resnet18":"/content/drive/My Drive/ProjeLateEarly/resnet18Late valAccuracy.npy", "resnet50":"/content/drive/My Drive/ProjeLateEarly/resnet50Late valAccuracy.npy", "resnet152":"/content/drive/My Drive/ProjeLateEarly/resnet152Late valAccuracy.npy"},
             {"alexnet":"/content/drive/My Drive/ProjeLateEarly/alexnetLate val_loss.npy", "vgg":"/content/drive/My Drive/ProjeLateEarly/vggLate val_loss.npy", "resnet18":"/content/drive/My Drive/ProjeLateEarly/resnet18Late val_loss.npy", "resnet50":"/content/drive/My Drive/ProjeLateEarly/resnet50Late val_loss.npy", "resnet152":"/content/drive/My Drive/ProjeLateEarly/resnet152Late val_loss.npy"},
             ["/content/drive/My Drive/ProjeLateEarly/Late Accuracy.jpg", "/content/drive/My Drive/ProjeLateEarly/Late Loss.jpg"], "Fusion")"""

"""plotFunction({"alexnet":"/content/drive/My Drive/justCNN/alexnetjustCNN TrainAccurac2.npy", "vgg":"/content/drive/My Drive/justCNN/vggjustCNN TrainAccuracy2.npy", "resnet18":"/content/drive/My Drive/justCNN/resnet18justCNN TrainAccurac2.npy", "resnet50":"/content/drive/My Drive/justCNN/resnet50justCNN TrainAccuracy2.npy", "resnet152":"/content/drive/My Drive/justCNN/resnet152justCNN TrainAccuracy2.npy"},
             {"alexnet":"/content/drive/My Drive/justCNN/alexnetjustCNN trainloss2.npy", "vgg":"/content/drive/My Drive/justCNN/vggjustCNN trainloss2.npy", "resnet18":"/content/drive/My Drive/justCNN/resnet18justCNN trainloss2.npy", "resnet50":"/content/drive/My Drive/justCNN/resnet50justCNN trainloss2.npy", "resnet152":"/content/drive/My Drive/justCNN/resnet152justCNN trainloss2.npy"},
             {"alexnet":"/content/drive/My Drive/justCNN/alexnetjustCNN valAccurac2.npy", "vgg":"/content/drive/My Drive/justCNN/vggjustCNN valAccuracy2.npy", "resnet18":"/content/drive/My Drive/justCNN/resnet18justCNN valAccurac2.npy", "resnet50":"/content/drive/My Drive/justCNN/resnet50justCNN valAccuracy2.npy", "resnet152":"/content/drive/My Drive/justCNN/resnet152justCNN valAccuracy2.npy"},
             {"alexnet":"/content/drive/My Drive/justCNN/alexnetjustCNN val_los2.npy", "vgg":"/content/drive/My Drive/justCNN/vggjustCNN val_loss2.npy", "resnet18":"/content/drive/My Drive/justCNN/resnet18justCNN val_los2.npy", "resnet50":"/content/drive/My Drive/justCNN/resnet50justCNN val_loss2.npy", "resnet152":"/content/drive/My Drive/justCNN/resnet152justCNN val_loss2.npy"},
             ["/content/drive/My Drive/ProjeLateEarly/JustCNN Accuracy.jpg", "/content/drive/My Drive/ProjeLateEarly/JustCNN Loss.jpg"], "JustCNN")"""

dataloader = bookTestDataLoader(64, 224, word_to_idx)
testModel("/content/drive/My Drive/ProjeLateEarly/resnet152early.pth", len(word_to_idx), "resnet152", dataloader,
          "/content/drive/My Drive/ProjeLateEarly/confusionMatrix.jpg")
