from __future__ import division
import sys
sys.path.append("..")
import os
# import glob

from efficientnet_pytorch import EfficientNet
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics

from utils.data_generator import DataGenerator
from utils.custom_dataloader import FastDataLoader
# from .augmentation_setup import custom_augment
from utils.utils import FocalLoss, load_and_crop, preprocess_input, multi_threshold,\
    CustomDataParallel
from .callback import SaveModelCheckpoint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.autonotebook import tqdm
# import itertools
# import pandas as pd
import numpy as np
import random
import torch_optimizer
import shutil
import xlsxwriter
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import cv2
from datetime import datetime

class EfficientNetWrapper:
    def __init__(self, config):
        self.config = config
        self.classes = config.CLASS_NAME
        self.input_size = config.INPUT_SIZE
        self.binary_option = config.BINARY
        self.failClasses = config.FAIL_CLASSNAME
        self.passClasses = config.PASS_CLASSNAME
        self.pytorch_model = None
        self.num_of_classes = len(self.classes)
        self.data = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.id_class_mapping = None
        self.class_weights = None
        self.evaluate_generator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _build_model(self):
        try:
            model_class = {
                'B0': 'efficientnet-b0',
                'B1': 'efficientnet-b1',
                'B2': 'efficientnet-b2',
                'B3': 'efficientnet-b3',
                'B4': 'efficientnet-b4',
                'B5': 'efficientnet-b5',
                'B6': 'efficientnet-b6',
                'B7': 'efficientnet-b7',
                'B8': 'efficientnet-b8',
            }[self.config.ARCHITECTURE]
        except KeyError:
            raise ValueError('Invalid Model architecture')

        try:
            if self.config.WEIGHT_PATH:
                base_model = EfficientNet.from_pretrained(model_class, \
                    weights_path=self.config.WEIGHT_PATH, \
                    advprop=False, \
                    num_classes=len(self.classes)\
                    ,image_size=self.config.INPUT_SIZE\
                    )
            else:
                base_model = EfficientNet.from_name(model_class, \
                    num_classes=len(self.classes)\
                    ,image_size=self.config.INPUT_SIZE\
                    )
        except:
            base_model = torch.load(self.config.WEIGHT_PATH)

        return base_model

    def load_classes(self):
        if self.binary_option:
            init_class = ['Reject','Pass']
            self.classes = init_class
            self.num_of_classes = len(init_class)
            self.id_class_mapping = {class_id: class_name for class_id, class_name in enumerate(init_class)}

        else:
            self.num_of_classes = len(self.classes)
            self.id_class_mapping = {class_id: class_name for class_id, class_name in enumerate(self.classes)}

    def prepare_data(self):
        self.load_classes()

        list_Directory = [
            os.path.join(self.config.DATASET_PATH, 'Train'),
            os.path.join(self.config.DATASET_PATH, 'Validation'),
            os.path.join(self.config.DATASET_PATH, 'Test'),
            # os.path.join(self.config.DATASET_PATH, 'Part4'),
            # os.path.join(self.config.DATASET_PATH, 'Gerd_Underkill_bmp'),
        ]

        list_Generator = []
        for diRectory in list_Directory.copy():
            if not os.path.exists(diRectory) or len(os.listdir(diRectory)) == 0:
                list_Directory.remove(diRectory)

        for diRectory in list_Directory:
            generator = DataGenerator(diRectory,\
                self.classes, self.failClasses, self.passClasses,\
                self.input_size, self.binary_option, augmentation=self.config.AU_LIST if "train" in diRectory.lower() else None )
            
            list_Generator.append(generator)
        
        check_train = [list_Generator[s_value] for s_value in [value for value in [list_Directory.index(set_path) for set_path in list_Directory if "train" in set_path.lower()]]]
        self.train_generator = check_train[0] if len(check_train) > 0 else None
        
        check_val = [list_Generator[s_value] for s_value in [value for value in [list_Directory.index(set_path) for set_path in list_Directory if "validation" in set_path.lower()]]]
        self.val_generator = check_val[0] if len(check_val) > 0 else None
        
        check_test = [list_Generator[s_value] for s_value in [value for value in [list_Directory.index(set_path) for set_path in list_Directory if "test" in set_path.lower()]]]
        self.test_generator = check_test[0] if len(check_test) > 0 else None
            
        # self.evaluate_generator =  DataGenerator(list_Directory,\
        # self.classes, self.failClasses, self.passClasses,\
        # self.input_size, self.binary_option)

        # self.class_weights = compute_class_weight('balanced',self.train_generator.metadata[0], self.train_generator.metadata[1])

        return list_Generator

    def optimizer_chosen(self, model_param):
        try:
            optimizer_dict = {
                'sgd': optim.SGD(params= model_param, lr=self.config.LEARNING_RATE, momentum=0.9, nesterov=True),
                'adam': optim.Adam(params=model_param, lr=self.config.LEARNING_RATE),
                'adadelta': optim.Adadelta(params=model_param, lr=self.config.LEARNING_RATE),
                'adagrad': optim.Adagrad(params=model_param, lr=self.config.LEARNING_RATE),
                'adamax': optim.Adamax(params=model_param, lr=self.config.LEARNING_RATE),
                'adamw': optim.AdamW(params=model_param, lr=self.config.LEARNING_RATE),
                'asgd': optim.ASGD(params=model_param, lr=self.config.LEARNING_RATE),
                'rmsprop': optim.RMSprop(params=model_param, lr=self.config.LEARNING_RATE),
                'radam': torch_optimizer.RAdam(params=model_param, lr=self.config.LEARNING_RATE),
                'ranger': torch_optimizer.Ranger(params=model_param, lr=self.config.LEARNING_RATE)
            }[self.config.OPTIMIZER.lower()]

            return optimizer_dict
        except KeyError:
            print("Invalid optimizers")

    def train(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        train_checkpoint_dir = self.config.LOGS_PATH
        os.makedirs(train_checkpoint_dir,exist_ok=True)

        # trainloader = DataLoader(self.train_generator, \
        #     batch_size=self.config.BATCH_SIZE, shuffle=True,num_workers=0)
        
        # valloader = DataLoader(self.val_generator, \
        #     batch_size=self.config.BATCH_SIZE, shuffle=False,num_workers=0)

        trainloader = FastDataLoader(self.train_generator, \
            batch_size=self.config.BATCH_SIZE * self.config.GPU_COUNT, shuffle=True, num_workers=self.config.NUM_WORKERS)

        valloader = FastDataLoader(self.val_generator, \
            batch_size=self.config.BATCH_SIZE * self.config.GPU_COUNT, shuffle=False, num_workers=self.config.NUM_WORKERS)

        # evalloader = FastDataLoader(self.evaluate_generator, \
        #     batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=12)

        if self.config.GPU_COUNT > 1:
            # self.pytorch_model = self._build_model().to(self.device)
            self.pytorch_model = self._build_model().to(self.device)
            # self.pytorch_model = CustomDataParallel(self.pytorch_model, self.config.GPU_COUNT)
            self.pytorch_model = nn.DataParallel(self.pytorch_model)
        else:
            self.pytorch_model = self._build_model().to(self.device)
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        print(f"[DEBUG] class_weight : {self.class_weights}")
        # criterion = FocalLoss(torch.from_numpy(self.class_weights).float().to(self.device, non_blocking=True))
        
        model_parameters = list(self.pytorch_model.parameters())
        # print(model_parameters)

        optimizer = self.optimizer_chosen(model_parameters)

        # optimizer = optim.SGD(model.parameters(), lr=self.config['lr'], momentum=0.9)
    
        # Init tensorboard
        writer = SummaryWriter(log_dir=train_checkpoint_dir)

        
        self.failClasses = ['Reject'] if self.binary_option else self.failClasses
        self.passClasses = ['Pass'] if self.binary_option else self.passClasses
        # pass_class_index = [self.classes.index(class_) for class_ in self.passClasses]
        # fail_class_index = [self.classes.index(class_) for class_ in self.failClasses]

        start_time = datetime.now()
        
        value_best = 100

        # enumerate epoch
        try:
            for epoch in range(self.config.NO_EPOCH):

                # test_average = AverageMeter('Loss')
                running_loss = AverageMeter('Loss')
                running_correct = AverageMeter('Acc')

                running_val_loss = AverageMeter('Val_Loss')
                running_val_acc = AverageMeter('Val_Acc')

                # class_correct = list(0. for i in range(len(self.classes)))
                # class_total = list(0. for i in range(len(self.classes)))
                #
                # y_gth_list = []
                # y_score_list = []
                # y_pred_list = []

                print(f"Epoch {epoch+1}/ {self.config.NO_EPOCH}")
                print('-' * 20)

                # enumerate mini batch
                progress_bar = tqdm(trainloader)

                self.pytorch_model.train()

                for iter, data in enumerate(progress_bar):
                    inputs, labels = data[0], data[1]

                    if self.config.GPU_COUNT == 1:
                        inputs = inputs.to(self.device, non_blocking=True)
                        # labels = labels.to(self.device, non_blocking=True)
                        
                    # zero the parameter gradients
                    # print(inputs.size(0))
                    optimizer.zero_grad()
                    outputs = self.pytorch_model(inputs)

                    labels = labels.to(outputs.device, non_blocking=True)
            
                    # forward + backward + optimize
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # TODO: Fix the loss/acc and batch loss/acc

                    # print statistics on epoch end
                    # running_loss += mean(batch_loss) * batch_size (overall loss of samples)
                    # running_correct += batch_correct (how many sample in 1 batch are corret)
                    # test_average.update(loss.item(), inputs.size(0))
                    running_loss.update(loss.item() * inputs.size(0), inputs.size(0))
                    running_correct.update(torch.sum(preds == labels.data).double(), inputs.size(0))
                    # print(f"[DEBUG] correct: {type(torch.sum(preds == labels.data).data)}")
                    # running_loss += loss.item() * inputs.size(0)
                    # running_correct += torch.sum(preds == labels.data)

                    # current_samples += inputs.size(0)

                    # Update running bar
                    
                    progress_bar.set_description(\
                    'Epoch: {}/{}. {}: {:.5}. {}: {:.5}'.format(\
                    epoch + 1, self.config.NO_EPOCH, \
                    running_loss.dict_return()['name'] ,running_loss.dict_return()['avg'], \
                    running_correct.dict_return()['name'] ,running_correct.dict_return()['avg']))

                    progress_bar.update()
                
                self.pytorch_model.eval()
                with torch.no_grad():
                    for val_data in valloader:
                        inputs_val , labels_val = val_data[0], val_data[1]

                        if self.config.GPU_COUNT == 1:
                            inputs_val = inputs_val.to(self.device, non_blocking=True)
                            # labels_val = labels_val.to(self.device, non_blocking=True)

                        outputs_val = self.pytorch_model(inputs_val)
                        labels_val = labels_val.to(outputs_val.device, non_blocking=True)

                        val_score, val_preds = torch.max(outputs_val, 1)
                        # softmax_score = torch.nn.Softmax(dim=1)(outputs_val)
                        # labels_val_score = [round(softmax_score.tolist()[i][1],4) for i in range(len(outputs_val))]
                        # print(f"[DEBUG] softmax score: {softmax_score}")
                        # print(f"[DEBUG] label val score: {labels_val_score}")

                        # c = (val_preds == labels_val).squeeze()
                        # print(f"[DEBUG] c val: {c}")

                        loss = criterion(outputs_val, labels_val)
                        running_val_loss.update(loss.item() * inputs_val.size(0), inputs_val.size(0))
                        running_val_acc.update(torch.sum(val_preds == labels_val.data).double(), inputs_val.size(0))

                        # y_gth_list.extend(labels_val.tolist())
                        # y_pred_list.extend(val_preds.tolist())
                        # y_score_list.extend(labels_val_score)

                        # for i in range(min(self.config.BATCH_SIZE, inputs_val.size(0))):
                        #     label = labels_val[i]
                        #     class_correct[label] += c[i].item()
                        #     class_total[label] += 1.
            
                print(f"val Loss: {running_val_loss.dict_return()['avg']} val Acc: { running_val_acc.dict_return()['avg']}")
                
                # Evaluate metric part
                # print("====================================")
                # print("Calculating FN/FP rate.....")
                # print("====================================")
                # with torch.no_grad():
                #     for eval_data in evalloader:
                #         inputs_eval , labels_eval = eval_data[0], eval_data[1]

                #         if self.config.GPU_COUNT == 1:
                #             inputs_eval = inputs_eval.to(self.device, non_blocking=True)
                #             # labels_val = labels_val.to(self.device, non_blocking=True)

                #         outputs_eval = self.pytorch_model(inputs_eval)
                #         labels_eval = labels_eval.to(outputs_eval.device, non_blocking=True)

                #         _, eval_preds = torch.max(outputs_eval, 1)

                #         y_gth_list.extend(labels_eval.tolist())
                #         y_pred_list.extend(eval_preds.tolist())
                        
                #         # softmax_score = torch.nn.Softmax(dim=1)(outputs_val)
                #         # labels_val_score = [round(softmax_score.tolist()[i][1],4) for i in range(len(outputs_val))]
                #         # print(f"[DEBUG] softmax score: {softmax_score}")
                #         # print(f"[DEBUG] label val score: {labels_val_score}")

                #         # c = (val_preds == labels_val).squeeze()
                #         # print(f"[DEBUG] c val: {c}")

                # fail_gth_list  = [np.array(y_gth_list) == class_ for class_ in fail_class_index]
                # fail_gth_list = np.sum(fail_gth_list, axis=0)
                # total_fail = np.sum(fail_gth_list)
                # false_fail_pred_list = [np.array(y_pred_list) == class_ for class_ in fail_class_index]
                # false_fail_pred_list = np.invert(np.sum(false_fail_pred_list, axis=0).astype('bool'))
                # false_fail_pred_list = false_fail_pred_list * fail_gth_list
                # total_underkill = np.sum(false_fail_pred_list)
                # UK_ratio = (total_underkill / total_fail) * 100

                # pass_gth_list = [np.array(y_gth_list) == class_ for class_ in pass_class_index]
                # pass_gth_list = np.sum(pass_gth_list, axis=0)
                # total_pass = np.sum(pass_gth_list)
                # false_pass_pred_list = [np.array(y_pred_list) == class_ for class_ in pass_class_index]
                # false_pass_pred_list = np.invert(np.sum(false_pass_pred_list, axis=0).astype('bool'))
                # false_pass_pred_list = false_pass_pred_list  * pass_gth_list
                # total_overkill = np.sum(false_pass_pred_list)
                # OK_ratio = (total_overkill / total_pass ) * 100
                
                # for i in range(len(self.classes)):
                #     print('Accuracy of %5s : %.2f %%' %\
                #         (self.classes[i], 100 * (class_correct[i] / class_total[i]) \
                #             if class_correct[i] > 0 or class_total[i] >0 else  0))
                
                # Tensorboard part
                # General part
                # print(f"Underkill rate: {UK_ratio} %")
                # print(f"Overkill rate: {OK_ratio} %")
                writer.add_scalars('Loss',{'Train': running_loss.dict_return()['avg'],\
                                            'Val' : running_val_loss.dict_return()['avg']}, epoch)

                writer.add_scalars('Acc', {'Train': running_correct.dict_return()['avg'],\
                                            'Val' : running_val_acc.dict_return()['avg']}, epoch)

                # writer.add_scalars('Custom',{"Underkill_rate": UK_ratio,\
                #                              "Overkill_rate": OK_ratio}, epoch)

                # writer.add_scalars('Acc each class', {self.classes[i] :
                #     round(100 * (class_correct[i] / class_total[i]),2) for i in range(len(self.classes))}, epoch)
                
                # Specific part
                # writer.add_scalars('Custom', {'AUC': auc_score}, epoch)

                writer.flush()
                # Save model part
                if self.config.IS_SAVE_BEST_MODELS:
                    if value_best > running_val_loss.dict_return()['avg']:
                        value_best = running_val_loss.dict_return()['avg']
                        SaveModelCheckpoint(self.pytorch_model, self.config.LOGS_PATH, epoch, value_best, True)
                    else:
                        pass
                else:
                    SaveModelCheckpoint(self.pytorch_model, self.config.LOGS_PATH, epoch)
        except KeyboardInterrupt:
            if self.config.IS_SAVE_BEST_MODELS:
                pass
            else:
                SaveModelCheckpoint(self.pytorch_model, self.config.LOGS_PATH, epoch)
            writer.close()

        writer.close()
        end_time = datetime.now()
        print("[DEBUG] Training time: {}".format(end_time-start_time))
    
    def evaluate(self):
        pass

    def predict_one(self, img, TTA=False):
        self.pytorch_model.eval()
        with torch.no_grad():
            if TTA:
                Y_list = []
                TTA_ls = [
                    img,
                    cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                    cv2.rotate(img, cv2.ROTATE_180),
                    cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
                ]
                for i in range(len(TTA_ls)):
                    img_TTA = TTA_ls[i]
                    img_TTA = preprocess_input(img_TTA).to(self.device, non_blocking=True)
                    outputs = self.pytorch_model(img_TTA.unsqueeze(0))

                    propability = torch.nn.Softmax(dim = 1)(outputs)
                    Y_list.append(propability.tolist())
                propability = np.mean(Y_list, axis=0)

                # return propability[0]
                return self.manage_prediction(propability)
            else:

                img = preprocess_input(img).to(self.device, non_blocking=True)
                outputs = self.pytorch_model(img.unsqueeze(0))

                propability = torch.nn.Softmax(dim = 1)(outputs)
                # return propability[0]
                return self.manage_prediction(propability.tolist())
                # print(f"[DEBUG] propability :{propability}")

    def manage_prediction(self, propability_prediction):
        if self.config.CLASS_THRESHOLD is None or len(self.config.CLASS_THRESHOLD) == 0:
            prob_id = np.argmax(propability_prediction, axis=-1)
        else:
            ret = multi_threshold(np.array(propability_prediction), self.config.CLASS_THRESHOLD)
            if ret is None:
                # classID = len(self.config.CLASS_THRESHOLD)
                classID = -1
                className = "Unknown"
                all_scores = propability_prediction[0]
                return classID, all_scores, className
            else:
                prob_id, _ = ret

        class_name = self.id_class_mapping[prob_id[0]]
      
        return prob_id[0], propability_prediction[0], class_name

    def load_weight(self):
        self.load_classes()
        self.pytorch_model = self._build_model().to(self.device)
        
    def confusion_matrix_evaluate(self):
        # self.prepare_data()
        generator_list = self.prepare_data()

        self.pytorch_model.eval()
        
        workbook = xlsxwriter.Workbook("_model_result.xlsx")

        cell_format = workbook.add_format()
        cell_format.set_align('center')
        cell_format.set_align('vcenter')

        highlight_format = workbook.add_format()
        highlight_format.set_align('center')
        highlight_format.set_align('vcenter')
        highlight_format.set_bg_color("red")

        Header = ["image_id","Image","Label","Predict"]
        Header.extend(self.classes)
        Header.append("Underkill")
        Header.append("Overkill")

        self.failClasses = ["Reject"] if self.binary_option else self.failClasses
        self.passClasses = ["Pass"] if self.binary_option else self.passClasses

        fail_class_index = [self.classes.index(class_) for class_ in self.failClasses]
        pass_class_index = [self.classes.index(class_) for class_ in self.passClasses]

        for generator in generator_list:
            generator_loader = DataLoader(generator, batch_size=1, shuffle=False, num_workers=0)
            print(f"Inspecting PATH: {generator.input_dir}")

            start_row = 0
            start_column = 1
            worksheet = workbook.add_worksheet(generator.input_dir[0].split("\\")[-1])
            worksheet.write_row( start_row, start_column, Header, cell_format)
            worksheet.set_column("C:C",10)

            progress_bar = tqdm(generator_loader)
            y_gth_eval_ls = []
            y_pred_eval_ls = []

            for iter, data_eval in enumerate(progress_bar):
                with torch.no_grad():
                    Data = [0] * len(Header)
                    start_row += 1
                    worksheet.set_row(start_row, 60)
                    underkill_overkill_flag = 0

                    inputs_eval , labels_eval = data_eval[0].to(self.device, non_blocking=True), \
                            data_eval[1].to(self.device, non_blocking=True)
                    
                    image_path = data_eval[2][0]
                    image_name = data_eval[2][0].split("\\")[-1]

                    img, gt_name = load_and_crop(image_path, self.input_size)

                    pred_id, all_scores, pred_name = self.predict_one(img)
                    
                    # outputs_eval = self.pytorch_model(inputs_eval)

                    # probability = torch.nn.Softmax(dim=1)(outputs_eval)

                    # pred_id, pred_score, pred_name = self.manage_prediction(probability)

                    # all_scores = probability.tolist()

                    gt_id = labels_eval.tolist()[0]
                    
                    if self.binary_option:
                        gt_name = 'Reject' if gt_name in self.failClasses else 'Pass'
                    else:
                        pass

                    if gt_id in fail_class_index and (pred_id in pass_class_index or pred_id == len(self.classes)):    # Underkill
                        underkill_path = os.path.join("_Result",image_path.split("\\")[-2],"UK")
                        os.makedirs(underkill_path, exist_ok=True)
                        image_output_path = os.path.join(underkill_path,image_name)
                        cv2.imwrite(image_output_path, img)
                        shutil.copy(image_path + ".json", os.path.join(underkill_path,image_name+".json"))
                        underkill_overkill_flag = -1

                    elif gt_id in pass_class_index and pred_id in fail_class_index:     # Overkill
                        overkill_path = os.path.join("_Result",image_path.split("\\")[-2],"OK")
                        os.makedirs(overkill_path, exist_ok=True)
                        image_output_path = os.path.join(overkill_path,image_name)
                        cv2.imwrite(image_output_path, img)
                        shutil.copy(image_path + ".json", os.path.join(overkill_path,image_name+".json"))
                        underkill_overkill_flag = 1
                    
                    else:                                                               # Correct result
                        result_path = os.path.join("_Result", image_path.split("\\")[-2])
                        os.makedirs(result_path, exist_ok=True)
                        image_output_path = os.path.join(result_path,image_name)
                        cv2.imwrite(image_output_path, img)
                        shutil.copy(image_path + ".json", os.path.join(result_path,image_name + ".json"))

                    y_gth_eval_ls.extend(labels_eval.tolist())
                    y_pred_eval_ls.extend([pred_id])

                    Data[0] = image_name.split(".")[0]
                    Data[2] = gt_name
                    Data[3] = pred_name
                    Data[4:4+len(self.classes)] = all_scores
                    Data[-2] = True if underkill_overkill_flag == -1 else False
                    Data[-1] = True if underkill_overkill_flag == 1 else False
                    # print(f"[DEBUG]:\n{Data}")
                    for index, info in enumerate(Data):
                        
                        excel_format = highlight_format if (Data[index] == True and isinstance(Data[index],bool)) else cell_format

                        worksheet.insert_image(start_row, index + 1, image_output_path, {'x_scale': 0.5,'y_scale': 0.5, 'x_offset': 5, 'y_offset': 5,'object_position':1}\
                            ) if index == 1 else worksheet.write(start_row, index + 1, Data[index], excel_format)

                    progress_bar.update()

            header = [{'header': head} for head in Header]

            worksheet.add_table(0, 1, start_row, len(Header), {'columns':header})
            worksheet.freeze_panes(1,0)
            worksheet.hide_gridlines(2)

            confusion_matrix = metrics.confusion_matrix(y_gth_eval_ls, y_pred_eval_ls)
            print(f"Confusion matrix : \n{confusion_matrix}")

        workbook.close()


class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self, name):
        self.name = name 
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def dict_return(self):
        return {'name': self.name, 'value':self.val, 'count': self.count,'avg': self.avg}    