import numpy as np
import load_evaluation_dataset
import source_network_raw_emg_enhanced
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
import copy
from tqdm import tqdm
import os

from sklearn.metrics import confusion_matrix, precision_score, recall_score, balanced_accuracy_score, accuracy_score, f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def confusion_matrix(pred, Y, number_class=7):
    confusion_matrice = []
    for x in range(0, number_class):
        vector = []
        for y in range(0, number_class):
            vector.append(0)
        confusion_matrice.append(vector)
    for prediction, real_value in zip(pred, Y):
        prediction = int(prediction)
        real_value = int(real_value)
        confusion_matrice[prediction][real_value] += 1
    return np.array(confusion_matrice)


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


def calculate_fitness(examples_training, labels_training, examples_test0, labels_test0, examples_test1, labels_test1,
                      learning_rate=.1, training_cycle=4):
    accuracy_test = []
    bal_acc_test = []
    precision_test = []
    recall_test = []
    f1_test = []
    conf_mat_test = []
    
    for test_patient in tqdm(range(17)):
        examples_personne_training = []
        labels_gesture_personne_training = []
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
                    if k < training_cycle*7:
                        examples_personne_training.extend(examples_training[j][k])
                        labels_gesture_personne_training.extend(labels_training[j][k])
                        # examples_personne_training.extend(examples_test0[j][k])
                        # labels_gesture_personne_training.extend(labels_test0[j][k])
                        # examples_personne_training.extend(examples_test1[j][k])
                        # labels_gesture_personne_training.extend(labels_test1[j][k])
        
        examples_personne_scrambled, labels_gesture_personne_scrambled = scramble(examples_personne_training,
                                                                                  labels_gesture_personne_training)
        valid_examples = examples_personne_scrambled[0:int(len(examples_personne_scrambled) * 0.1)]
        labels_valid = labels_gesture_personne_scrambled[0:int(len(labels_gesture_personne_scrambled) * 0.1)]
        
        X_fine_tune = examples_personne_scrambled[int(len(examples_personne_scrambled) * 0.1):]
        Y_fine_tune = labels_gesture_personne_scrambled[int(len(labels_gesture_personne_scrambled) * 0.1):]
        
        train = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                              torch.from_numpy(np.array(Y_fine_tune, dtype=np.int64)))
        
        validation = TensorDataset(torch.from_numpy(np.array(valid_examples, dtype=np.float32)),
                                   torch.from_numpy(np.array(labels_valid, dtype=np.int64)))
        
        trainloader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
        validationloader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True, drop_last=True)
        
        cnn = source_network_raw_emg_enhanced.Net(number_of_class=7).to(device)
        
        criterion = nn.CrossEntropyLoss(reduction="sum")
        optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
        
        precision = 1e-6
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                         verbose=True, eps=precision)
        
        cnn = train_model(cnn, criterion, optimizer, scheduler, dataloaders={"train": trainloader,
                                                                             "val": validationloader},
                          precision=precision)
        
        cnn.eval()
        X_test, Y_test = scramble(X_test, Y_test)
        
        test= TensorDataset(torch.from_numpy(np.array(X_test, dtype=np.float32)),
                               torch.from_numpy(np.array(Y_test, dtype=np.int64)))
        
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
    best_loss = float('inf')
    
    patience = 30
    patience_increase = 10
    
    best_weights = copy.deepcopy(cnn.state_dict())
    
    for epoch in range(num_epochs):
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
        if epoch >= 25:
            break

    cnn.load_state_dict(copy.deepcopy(best_weights))
    cnn.eval()
    return cnn


if __name__ == '__main__':
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

    with open("/content/drive/My Drive/544_Final_Project/cnn_evaluation_dataset_target_patient_dependent.txt", "a") as myfile:
        myfile.write("Test")
    for training_cycle in range(4, 5):
        print(f"\n--- TRAINING CYCLE {training_cycle} of 4 ---")
        accuracy_test = []
        bal_acc_test = []
        precision_test = []
        recall_test = []
        f1_test = []
        conf_mat_test = []

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
    
        with open("/content/drive/My Drive/544_Final_Project/MyoArmbandDataset/PyTorchImplementation/results/cnn_evaluation_dataset_target_patient_dependent.txt", "a") as myfile:
            myfile.write("--- RAW, CNN, PATIENT INDEPENDENT SOURCE ---")
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
