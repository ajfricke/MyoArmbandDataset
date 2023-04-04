import numpy as np
import copy
import time
import os

import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import target_network_raw_emg_enhanced
import load_pre_training_dataset
import load_evaluation_dataset

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_score, recall_score, balanced_accuracy_score, accuracy_score, f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def scramble(examples, labels, second_labels=[]):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    if len(second_labels) == len(labels):
        new_second_labels = []
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
            new_second_labels.append(second_labels[i])
        return new_examples, new_labels, new_second_labels
    else:
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
        return new_examples, new_labels

def calculate_pre_training(examples, labels):
    list_train_dataloader = []
    list_validation_dataloader = []
    human_number = 0
    for j in tqdm(range(19)):
        examples_personne_training = []
        labels_gesture_personne_training = []
        labels_human_personne_training = []

        examples_personne_valid = []
        labels_gesture_personne_valid = []
        labels_human_personne_valid = []

        for k in range(len(examples[j])):
            if k < 21:
                examples_personne_training.extend(examples[j][k])
                labels_gesture_personne_training.extend(labels[j][k])
                labels_human_personne_training.extend(human_number * np.ones(len(labels[j][k])))
            else:
                examples_personne_valid.extend(examples[j][k])
                labels_gesture_personne_valid.extend(labels[j][k])
                labels_human_personne_valid.extend(human_number * np.ones(len(labels[j][k])))

        #print(np.shape(examples_personne_training))
        examples_personne_scrambled, labels_gesture_personne_scrambled, labels_human_personne_scrambled = scramble(
            examples_personne_training, labels_gesture_personne_training, labels_human_personne_training)

        examples_personne_scrambled_valid, labels_gesture_personne_scrambled_valid, labels_human_personne_scrambled_valid = scramble(
            examples_personne_valid, labels_gesture_personne_valid, labels_human_personne_valid)

        train = TensorDataset(torch.from_numpy(np.array(examples_personne_scrambled, dtype=np.float32)),
                              torch.from_numpy(np.array(labels_gesture_personne_scrambled, dtype=np.int64)))
        validation = TensorDataset(torch.from_numpy(np.array(examples_personne_scrambled_valid, dtype=np.float32)),
                                   torch.from_numpy(np.array(labels_gesture_personne_scrambled_valid, dtype=np.int64)))

        trainLoader = torch.utils.data.DataLoader(train, batch_size=3315, shuffle=True, drop_last=True)
        validationLoader = torch.utils.data.DataLoader(validation, batch_size=1312, shuffle=True, drop_last=True)

        list_train_dataloader.append(trainLoader)
        list_validation_dataloader.append(validationLoader)

        human_number += 1
        # print("Shape training : ", np.shape(examples_personne_scrambled))
        # print("Shape valid : ", np.shape(examples_personne_scrambled_valid))

    cnn = target_network_raw_emg_enhanced.SourceNetwork(number_of_class=7, dropout_rate=.35).to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(cnn.parameters(), lr=0.002335721469090121)
    precision = 1e-8
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15,
                                                     verbose=True, eps=precision)

    pre_train_model(cnn, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                    dataloaders={"train": list_train_dataloader, "val": list_validation_dataloader},
                    precision=precision)

def pre_train_model(cnn, criterion, optimizer, scheduler, dataloaders, num_epochs=500, precision=1e-8):
    since = time.time()

    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(len(dataloaders['val'])):
        state_dict = cnn.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batch_norm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    best_loss = float('inf')

    best_weights = copy.deepcopy(cnn.state_dict())

    patience = 30
    patience_increase = 30
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        if epoch%10 == 0:
            print("Epoch #", epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn.train(True)  # Set model to training mode
            else:
                cnn.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0

            # Get a random order for the training dataset
            random_vec = np.arange(len(dataloaders[phase]))
            np.random.shuffle(random_vec)

            for dataset_index in random_vec:
                # Retrieves the BN weights calculated so far for this dataset
                BN_weights = list_dictionaries_BN_weights[dataset_index]
                cnn.load_state_dict(BN_weights, strict=False)

                loss_over_datasets = 0.
                correct_over_datasets = 0.
                for i, data in enumerate(dataloaders[phase][dataset_index], 0):
                    # get the inputs
                    inputs, labels = data

                    inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    if phase == 'train':
                        cnn.train()
                        # forward
                        outputs = cnn(inputs)
                        _, predictions = torch.max(outputs.data, 1)
                        # backward
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        loss = loss.item()

                    else:
                        cnn.eval()
                        # forward
                        outputs = cnn(inputs)
                        _, predictions = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        loss = loss.item()
                    # Statistic for this dataset
                    loss_over_datasets += loss
                    correct_over_datasets += torch.sum(predictions == labels.data)
                    total += labels.size(0)
                # Statistic global
                running_loss += loss_over_datasets
                running_corrects += correct_over_datasets

                # Save the BN statistics for this dataset
                state_dict = cnn.state_dict()
                batch_norm_dict = {}
                for key in state_dict:
                    if "batch_norm" in key:
                        batch_norm_dict.update({key: state_dict[key]})
                list_dictionaries_BN_weights[dataset_index] = copy.deepcopy(batch_norm_dict)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.item() / total

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(cnn.state_dict())
                    patience = patience_increase + epoch
        if epoch > patience:
            break

    time_elapsed = time.time() - since

    # Save the best weights found to file
    torch.save(best_weights, '/content/drive/My Drive/544_Final_Project/best_pre_train_weights_target_raw.pt')


def calculate_fitness(examples_training, labels_training, examples_test0, labels_test0, examples_test1, labels_test_1,
                      learning_rate=.1, training_cycle=4):
    accuracy_test = []
    bal_acc_test = []
    precision_test = []
    recall_test = []
    f1_test = []
    conf_mat_test = []
    
    for test_patient in tqdm(range(17)):
        X_train = []
        y_train = []
        X_test, Y_test = [], []
        
        for j in range(17):
            for k in range(len(examples_training[j])):
                if j==test_patient:
                    X_test.extend(examples_training[j][k])
                    Y_test.extend(labels_training[j][k])
                    X_test.extend(examples_test0[j][k])
                    Y_test.extend(labels_test0[j][k])
                    X_test.extend(examples_test1[j][k])
                    Y_test.extend(labels_test1[j][k])
                else:
                    if k < 28:
                        X_train.extend(examples_training[j][k])
                        y_train.extend(labels_training[j][k])

        X_train, y_train = scramble(X_train, y_train)
        X_test, y_test = scramble(X_test, Y_test)

        X_train = X_train[0:int(len(X_train) * 0.2)]
        y_train = y_train[0:int(len(y_train) * 0.2)]

        X_acc_train = X_train[0:int(len(X_train) * 0.9)]
        y_acc_train = y_train[0:int(len(y_train) * 0.9)]
        
        X_fine_tune = X_train[int(len(X_train) * 0.9):]
        y_fine_tune = y_train[int(len(y_train) * 0.9):]

        X_test = X_test[0:int(len(X_test) * 0.2)]
        y_test = y_test[0:int(len(y_test) * 0.2)]

        print('Training size: ', len(X_acc_train))
        print('Validation size: ', len(X_fine_tune))
        print('Testing size: ', len(X_test))
        
        train = TensorDataset(torch.from_numpy(np.array(X_acc_train, dtype=np.float32)),
                                torch.from_numpy(np.array(y_acc_train, dtype=np.int64)))
        
        validation = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                                    torch.from_numpy(np.array(y_fine_tune, dtype=np.int64)))
        
        test = TensorDataset(torch.from_numpy(np.array(X_test, dtype=np.float32)),
                                torch.from_numpy(np.array(y_test, dtype=np.int64)))
        
        trainloader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
        validationloader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True, drop_last=True)

        pre_trained_weights = torch.load('/content/drive/My Drive/544_Final_Project/best_pre_train_weights_target_raw.pt')
        cnn = target_network_raw_emg_enhanced.TargetNetwork(number_of_class=7,
                                                            weights_pre_trained_convnet=pre_trained_weights,
                                                            dropout=.5).to(device)
        
        criterion = nn.CrossEntropyLoss(reduction="sum")
        optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
        
        precision = 1e-6
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                            verbose=True, eps=precision)
        
        cnn = train_model(cnn, criterion, optimizer, scheduler, dataloaders={"train": trainloader,
                                                                                "val": validationloader},
                            precision=precision)
        
        cnn.eval()
        
        test_predicted = []
        test_labels = []
        test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)
        total = 0
        correct_prediction_test = 0
        for k, data_test in enumerate(test_loader, 0):
            # get the inputs
            inputs_test, ground_truth_test = data_test
            inputs_test, ground_truth_test = Variable(inputs_test.to(device)), Variable(ground_truth_test.to(device))
            
            outputs_test = cnn(inputs_test)
            _, predicted = torch.max(outputs_test.data, 1)

            test_predicted.extend(predicted.cpu().numpy())
            test_labels.extend(ground_truth_test.data.cpu().numpy())

            correct_prediction_test += (predicted.cpu().numpy() == ground_truth_test.data.cpu().numpy()).sum()
            total += ground_truth_test.size(0)

        accuracy_test.append(accuracy_score(test_labels, test_predicted))
        bal_acc_test.append(balanced_accuracy_score(test_labels, test_predicted))
        precision_test.append(precision_score(test_labels, test_predicted, average='macro'))
        recall_test.append(recall_score(test_labels, test_predicted, average='macro'))
        f1_test.append(f1_score(test_labels, test_predicted, average='macro'))
        conf_mat_test.append(confusion_matrix(test_labels, test_predicted))

    return (accuracy_test, bal_acc_test, precision_test, recall_test, f1_test, conf_mat_test)


def train_model(cnn, criterion, optimizer, scheduler, dataloaders, num_epochs=500, precision=1e-8):
    since = time.time()
    
    best_loss = float('inf')
    
    patience = 30
    patience_increase = 10
    
    best_weights = copy.deepcopy(cnn.state_dict())
    
    for epoch in range(num_epochs):
        if epoch%10 == 0:
            print("Epoch #", epoch)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn.train(True)  # Set model to training mode
            else:
                cnn.train(False)  # Set model to evaluate mode
            
            running_loss = 0.
            running_corrects = 0
            total = 0
            
            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                
                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':
                    cnn.train()
                    # forward
                    outputs = cnn(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    # backward
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                
                else:
                    cnn.eval()
                    # forward
                    outputs = cnn(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    loss = loss.item()
                
                # statistics
                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)
        
            epoch_loss = running_loss / total
            epoch_acc = running_corrects.item() / total
            
            #print('{} Loss: {} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    #print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(cnn.state_dict())
                    patience = patience_increase + epoch
        #print("Epoch {} of {} took {:.3f}s".format(
            #epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break
    #print()
    
    time_elapsed = time.time() - since
    
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    cnn.load_state_dict(copy.deepcopy(best_weights))
    cnn.eval()
    return cnn

if __name__ == '__main__':
    if 'best_pre_train_weights_target_raw.pt' not in os.listdir('/content/drive/My Drive/544_Final_Project/'):
        print('CALCULATING PRE TRAINING')
        if "saved_evaluation_dataset_pre_training.npy" not in os.listdir("/content/drive/My Drive/544_Final_Project/"):
            print("Doing pre training")
            examples, labels = load_pre_training_dataset.read_data('/content/drive/My Drive/544_Final_Project/MyoArmbandDataset/PreTrainingDataset', type="training0")
            datasets = [examples, labels]
            np.save("/content/drive/My Drive/544_Final_Project/saved_evaluation_dataset_pre_training.npy", datasets)

        datasets_pre_training = np.load("/content/drive/My Drive/544_Final_Project/saved_evaluation_dataset_pre_training.npy", encoding="bytes", allow_pickle=True)
        examples_pre_training, labels_pre_training = datasets_pre_training

        calculate_pre_training(examples_pre_training, labels_pre_training)
        print('FINISHED PRE TRAINING')
    
    if "saved_evaluation_dataset_training.npy" not in os.listdir("/content/drive/My Drive/544_Final_Project/"):
        examples, labels = load_evaluation_dataset.read_data('/content/drive/My Drive/544_Final_Project/MyoArmbandDataset/EvaluationDataset', type="training0")
        datasets = [examples, labels]
        np.save("/content/drive/My Drive/544_Final_Project/saved_evaluation_dataset_training.npy", datasets)
    
    if "saved_evaluation_dataset_test0.npy" not in os.listdir("/content/drive/My Drive/544_Final_Project/"):
        examples, labels = load_evaluation_dataset.read_data('/content/drive/My Drive/544_Final_Project/MyoArmbandDataset/EvaluationDataset', type="Test0")
        datasets = [examples, labels]
        np.save("/content/drive/My Drive/544_Final_Project/saved_evaluation_dataset_test0.npy", datasets)
    
    if "saved_evaluation_dataset_test1.npy" not in os.listdir("/content/drive/My Drive/544_Final_Project"):
        examples, labels = load_evaluation_dataset.read_data('/content/drive/My Drive/544_Final_Project/MyoArmbandDataset/EvaluationDataset', type="Test1")
        datasets = [examples, labels]
        np.save("/content/drive/My Drive/544_Final_Project/saved_evaluation_dataset_test1.npy", datasets)
    
    
    datasets_training = np.load("/content/drive/My Drive/544_Final_Project/saved_evaluation_dataset_training.npy", encoding="bytes", allow_pickle=True)
    examples_training, labels_training = datasets_training
    
    datasets_test0 = np.load("/content/drive/My Drive/544_Final_Project/saved_evaluation_dataset_test0.npy", encoding="bytes", allow_pickle=True)
    examples_test0, labels_test0 = datasets_test0
    
    datasets_test1 = np.load("/content/drive/My Drive/544_Final_Project/saved_evaluation_dataset_test1.npy", encoding="bytes", allow_pickle=True)
    examples_test1, labels_test1 = datasets_test1

    # And here if the pre-training of the network was already completed.
    accuracy_one_by_one = []
    array_training_error = []
    array_validation_error = []
    # learning_rate=0.002335721469090121 (for network enhanced)

    with open("/content/drive/My Drive/544_Final_Project/cnn_evaluation_dataset_target_patient_independent.txt", "a") as myfile:
        myfile.write("Test")
    for training_cycle in range(4, 5):
        print(f"\n--- TRAINING CYCLE {training_cycle} of 4 ---")
        accuracy_test = []
        bal_acc_test = []
        precision_test = []
        recall_test = []
        f1_test = []
        conf_mat_test = []
        conf_mat_test1 = []
        for i in tqdm(range(3)):
            metrics = calculate_fitness(examples_training, labels_training, examples_test0,
                                        labels_test0, examples_test1, labels_test1,
                                        learning_rate=0.002335721469090121,
                                        training_cycle=training_cycle)
        
            accuracy_test.append(metrics[0])
            bal_acc_test.append(metrics[1])
            precision_test.append(metrics[2])
            recall_test.append(metrics[3])
            f1_test.append(metrics[4])
            conf_mat_test.append(metrics[5])
    
        with open("/content/drive/My Drive/544_Final_Project/cnn_evaluation_dataset_target_patient_independent.txt", "a") as myfile:
            myfile.write("--- RAW, CNN, PATIENT INDEPENDENT TARGET ---")
            myfile.write("ConvNet Training Cycle : " + str(training_cycle) + "\n\n")
            myfile.write("Accuracy: \n")
            myfile.write(str(np.mean(accuracy_test)) + '\n')

            myfile.write("Balanced Accuracy: \n")
            myfile.write(str(np.mean(bal_acc_test)) + '\n')

            myfile.write("Precision: \n")
            myfile.write(str(np.mean(precision_test)) + '\n')

            myfile.write("Recall: \n")
            myfile.write(str(np.mean(recall_test)) + '\n')

            myfile.write("F1 Score: \n")
            myfile.write(str(np.mean(f1_test)) + '\n')
            myfile.write("\n\n\n")

        import pickle
        with open('/content/drive/My Drive/544_Final_Project/conf_mat.pkl', 'wb') as f:
            pickle.dump(np.mean(conf_mat_test, axis=0), f)
