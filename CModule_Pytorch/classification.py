import os
import sys
import json
import time
import shutil
import glob
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from classification_module import config_loader
from classification_module.classification_wrapper import EfficientNetWrapper
from utils.utils import load_and_crop
from utils.utils import set_GPU
from datetime import datetime
###################
# Global Constant #
###################


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train EfficientNet.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="C:\\DLCApplication\\Dataset",
                        help='Directory of the DLModel dataset')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--weight', required=False,
                        metavar="/home/simon/logs/weights.h5",
                        help="Path to weights .h5")
    parser.add_argument('--config', default=os.path.join(ROOT_DIR, 'efficient_module/default_config.json'),
                        help='Path to config json file')
    parser.add_argument('--savePath', default="NONE", required=False,
                        help='Path to save predicted images')
    parser.add_argument('--au_list', required=False,
                        help='a list of used augment technique')
    parser.add_argument('--binary', required=False, default=0, type=int,
                        help='binary/mutli classification option 1/0')
    parser.add_argument('--gpu', required=False, default=None, type=int,
                        help='declare number of gpus')
    args = parser.parse_args()
    colors = [(255,0,0),(255,215,0),(255,140,0),(255,69,0),(0,255,0),(255,255,0),(0,255,255),(0,0,255), (252,186,3),(136,3,252)]
    
    try:
    
        if args.command == "train":
            param = config_loader.LoadConfig(args.config)

            class TrainConfig:
                NO_EPOCH = param.NO_EPOCH
                GPU_COUNT = param.NUM_GPU if args.gpu == None else args.gpu
                LEARNING_RATE = param.LEANING_RATE
                LEARNING_MOMENTUM = param.MOMENTUM
                WEIGHT_DECAY = param.DECAY
                OPTIMIZER = param.OPTIMIZER
                NUM_CLASSES = len(param.CLASS_NAME)
                CLASS_NAME = param.CLASS_NAME
                INPUT_SIZE = param.CHANGE_BOX_SIZE
                IMAGES_PER_GPU = param.BATCH_SIZE
                THRESHOLDS = param.CLASS_THRESHOLD
                AU_LIST = param.AUGMENT_LIST
                if AU_LIST == [] or AU_LIST == None:
                    AU_LIST = None
                else:
                    AU_LIST = param.AUGMENT_LIST
                ARCHITECTURE = param.ARCHITECTURE
                BATCH_SIZE = param.BATCH_SIZE
                LOGS_PATH = args.logs
                DATASET_PATH = args.dataset
                WEIGHT_PATH = args.weight if args.weight else None
                FAIL_CLASSNAME = param.FAILCLASS_NAME
                PASS_CLASSNAME = param.PASSCLASS_NAME
                BINARY = bool(args.binary)
                IS_SAVE_BEST_MODELS = param.IS_SAVE_BEST_MODELS
                NUM_WORKERS = param.NUM_WORKERS

            config = TrainConfig()
            set_GPU(config.GPU_COUNT)
            model = EfficientNetWrapper(config)
            model.prepare_data()
            # _init_t =  input("[DEBUG] Init train ?(Y/N)\nYour answer: ")
            # if _init_t.lower() == "y":
                # if config.WEIGHT_PATH:
                #     model.resume_training()
                # else:

            model.train()

            # else:
            #     pass
            print("\nTrain Done")

        elif args.command == "cm":
            param = config_loader.LoadConfig(args.config)
            
            class InferConfig:
                NO_EPOCH = param.NO_EPOCH
                GPU_COUNT = 1
                LEARNING_RATE = param.LEANING_RATE
                LEARNING_MOMENTUM = param.MOMENTUM
                WEIGHT_DECAY = param.DECAY
                OPTIMIZER = param.OPTIMIZER
                NUM_CLASSES = len(param.CLASS_NAME)
                CLASS_NAME = param.CLASS_NAME
                INPUT_SIZE = param.CHANGE_BOX_SIZE
                IMAGES_PER_GPU = param.BATCH_SIZE
                CLASS_THRESHOLD = param.CLASS_THRESHOLD
                
                AU_LIST = None

                ARCHITECTURE = param.ARCHITECTURE
                BATCH_SIZE = param.BATCH_SIZE
                LOGS_PATH = args.logs
                DATASET_PATH = args.dataset
                WEIGHT_PATH = args.weight if args.weight else None
                FAIL_CLASSNAME = param.FAILCLASS_NAME
                PASS_CLASSNAME = param.PASSCLASS_NAME
                BINARY = bool(args.binary) # Hardcode
                NUM_WORKERS = param.NUM_WORKERS

            config = InferConfig()
            set_GPU(config.GPU_COUNT)
            model = EfficientNetWrapper(config)
            model.load_weight()
            # Test with 1 img
            # img, _ = load_and_crop("D:\\Coding_practice\\_Data\\KLA_Infineon\\Dataset_model_1\\Test\\1_Burr_Defect_2 (1).bmp")
            # output = model.predict_one(img)
            # print(f"Result : {output}")
            model.confusion_matrix_evaluate()

        elif args.command == "labelling":
            param = config_loader.LoadConfig(args.config)
            
            class InferConfig:
                NO_EPOCH = param.NO_EPOCH
                GPU_COUNT = 1
                LEARNING_RATE = param.LEANING_RATE
                LEARNING_MOMENTUM = param.MOMENTUM
                WEIGHT_DECAY = param.DECAY
                OPTIMIZER = param.OPTIMIZER
                NUM_CLASSES = len(param.CLASS_NAME)
                CLASS_NAME = param.CLASS_NAME
                INPUT_SIZE = param.CHANGE_BOX_SIZE
                IMAGES_PER_GPU = param.BATCH_SIZE
                CLASS_THRESHOLD = param.CLASS_THRESHOLD
                AU_LIST = param.AUGMENT_LIST
                if AU_LIST == [] or AU_LIST == None:
                    AU_LIST = False
                else:
                    AU_LIST = True
                ARCHITECTURE = param.ARCHITECTURE
                BATCH_SIZE = param.BATCH_SIZE
                LOGS_PATH = args.logs
                DATASET_PATH = args.dataset
                WEIGHT_PATH = args.weight if args.weight else None
                FAIL_CLASSNAME = param.FAILCLASS_NAME
                PASS_CLASSNAME = param.PASSCLASS_NAME
                BINARY = bool(args.binary) # Hardcode

            config = InferConfig()
            model = EfficientNetWrapper(config)
            model.load_weight()
            model.labelling_raw_data()

        elif args.command == "test":
            param = config_loader.LoadConfig(args.config)
            assert args.weight

            class TestConfig:
                NO_EPOCH = param.NO_EPOCH
                GPU_COUNT = 1
                LEARNING_RATE = param.LEANING_RATE
                LEARNING_MOMENTUM = param.MOMENTUM
                WEIGHT_DECAY = param.DECAY
                OPTIMIZER = param.OPTIMIZER
                NUM_CLASSES = len(param.CLASS_NAME)
                CLASS_NAME = param.CLASS_NAME
                INPUT_SIZE = param.CHANGE_BOX_SIZE
                IMAGES_PER_GPU = param.BATCH_SIZE
                CLASS_THRESHOLD = param.CLASS_THRESHOLD
                AU_LIST = param.AUGMENT_LIST
                ARCHITECTURE = param.ARCHITECTURE
                BATCH_SIZE = param.BATCH_SIZE
                LOGS_PATH = args.logs
                DATASET_PATH = args.dataset
                WEIGHT_PATH = args.weight
                FAIL_CLASSNAME = param.FAILCLASS_NAME
                PASS_CLASSNAME = param.PASSCLASS_NAME
                BINARY = bool(args.binary) # Hardcode


            config = TestConfig()
            test_dir = os.path.join(config.DATASET_PATH, 'Test')
            model = EfficientNetWrapper(config)
            model.load_weight()

            detail = {'Reject': 0, 'Pass': 0, 'Overkill': 0, 'Underkill': 0}
            for k in detail.keys():
                folder = os.path.join(test_dir, k)
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder)
                    except OSError:
                        pass
                os.makedirs(folder, exist_ok=True)

            for filename in os.listdir(test_dir):
                if filename.split('.')[-1] == 'json':
                    img_path = os.path.join(test_dir, filename.split('.json')[0])
                    with open(os.path.join(test_dir, filename), encoding='utf-8') as jf:
                        json_data = json.load(jf)
                        gt_name = json_data['classId'][0]
                        gt_id = -1
                        for i, label in enumerate(config.CLASS_NAME):
                            if label == gt_name:
                                gt_id = i
                        box = dict(json_data['box'])
                        img, _ = load_and_crop(img_path, config.INPUT_SIZE)
                        pred_id, pred_score, pred_name = model.predict_one(img)

                        pred_dir = ''
                        if gt_name == 'Pass' and pred_name == 'Reject':  # FP - OK
                            pred_dir = os.path.join(test_dir, 'Overkill')
                            detail['Overkill'] += 1
                        elif gt_name == 'Reject' and pred_name == 'Pass':  # FN - UK
                            pred_dir = os.path.join(test_dir, 'Underkill')
                            detail['Underkill'] += 1
                        elif gt_name == pred_name:
                            pred_dir = os.path.join(test_dir, pred_name)
                            detail[pred_name] += 1

                        cv2.imwrite(os.path.join(pred_dir, filename.split('.json')[0]), img)

            print(detail)

        elif args.command == "classify":
            param = config_loader.LoadConfig(args.config)

            class ClassifyConfig:
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                LEARNING_RATE = param.LEANING_RATE
                LEARNING_MOMENTUM = param.MOMENTUM
                WEIGHT_DECAY = param.DECAY
                OPTIMIZER = param.OPTIMIZER
                NUM_CLASSES = len(param.CLASS_NAME)
                CLASS_NAME = param.CLASS_NAME
                INPUT_SIZE = param.CHANGE_BOX_SIZE
                IMAGES_PER_GPU = param.BATCH_SIZE
                CLASS_THRESHOLD = param.CLASS_THRESHOLD
                ARCHITECTURE = param.ARCHITECTURE
                BATCH_SIZE = param.BATCH_SIZE
                WEIGHT_PATH = args.weight
                FAIL_CLASSNAME = param.FAILCLASS_NAME
                PASS_CLASSNAME = param.PASSCLASS_NAME
                BINARY = bool(args.binary) # Hardcode

            config = ClassifyConfig()
            model = EfficientNetWrapper(config)
            model.load_weight()

            # Skip the first prediction
            draft_img = np.zeros((config.INPUT_SIZE, config.INPUT_SIZE, 3)).astype(np.float32)
            model.predict_one(draft_img)

            print('[INFO] Ready..', file=sys.stdout)
            while True:
                input_command = input("Image path or quit ")  # image path
                if input_command == "quit":
                    break
                image_name = input_command
                center = input("Center point ('256 256') ")
                if center == "":
                    center = "256 256"

                start = time.time()
                x_center, y_center = np.fromstring(center, dtype=int, sep=' ')
                anchor = np.array([
                    x_center - config.INPUT_SIZE / 2,
                    y_center - config.INPUT_SIZE / 2,
                    x_center + config.INPUT_SIZE / 2,
                    y_center + config.INPUT_SIZE / 2,
                    ], dtype=np.int32)

                image, _ = load_and_crop(image_name, config.INPUT_SIZE, (x_center, y_center))

                try:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                except:
                    pass

                pred_id, all_scores, pred_name = model.predict_one(image)
                end = time.time()

                result_dict = {
                    "Classes"   : model.classes,
                    "Scores"    : all_scores,
                    "Box"       : anchor.tolist(),
                    "FinalResult"    : pred_name
                }

                sendResult = ""
                sendResult += "Classification result%"
                sendResult += r"Result&&&" + json.dumps(result_dict, sort_keys=True) +"%"
                sendResult += "Done in predict"

                print(sendResult)

        elif args.command == "testConfusionMatrix":
            param = config_loader.LoadConfig(args.config)
            

            class ClassifyConfig:
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                LEARNING_RATE = param.LEANING_RATE
                LEARNING_MOMENTUM = param.MOMENTUM
                WEIGHT_DECAY = param.DECAY
                OPTIMIZER = param.OPTIMIZER
                NUM_CLASSES = len(param.CLASS_NAME)
                CLASS_NAME = param.CLASS_NAME
                INPUT_SIZE = param.CHANGE_BOX_SIZE
                IMAGES_PER_GPU = param.BATCH_SIZE
                CLASS_THRESHOLD = param.CLASS_THRESHOLD
                ARCHITECTURE = param.ARCHITECTURE
                BATCH_SIZE = param.BATCH_SIZE
                WEIGHT_PATH = args.weight
                FAIL_CLASSNAME = param.FAILCLASS_NAME
                PASS_CLASSNAME = param.PASSCLASS_NAME
                BINARY = bool(args.binary) # Hardcode

            config = ClassifyConfig()
            model = EfficientNetWrapper(config)
            model.load_weight()

            while True:
                input_command = input("Image directory:")  # image path
                savePath = args.savePath
                result = ""
                if input_command != "quit":      
                    # Initialize confusion matrix. numClass = 2
                    # [Count, Reject, Pass, Classification Accuracy, Underkill, Overkill
                    #  ....., ......, ...., ......................%, ........%, .......%,
                    #  ....., ......, ...., ......................%, ........%, .......%]
                    confusionMatrix = np.zeros((2, 2 + 4))
                    
                    imageDir = input_command
                    lastString = imageDir.split("\\")[-1]
                    print(lastString)
                    imagePaths = []
                    if lastString == "Train" or lastString == "train":
                        if savePath != "NONE":
                            savePath = os.path.join(savePath,"Train")
                        for subDir in os.listdir(imageDir):
                            for path in glob.glob(imageDir + "\\" + subDir + "\\*.bmp"):
                                imagePaths.append(path)
                    else:
                        imagePaths = glob.glob(input_command + "\\*.bmp")
                        if savePath != "NONE":
                            if lastString == "Validation" or lastString == "validation":
                                savePath = os.path.join(savePath,"Validation")
                            elif lastString == "Test" or lastString == "test":
                                savePath = os.path.join(savePath,"Test")
                    
                    start = time.time()

                    for imagePath in imagePaths:
                        jsonPath = imagePath + ".json"
                        with open(jsonPath, encoding='utf-8') as jsonFile:
                            jsonData = json.load(jsonFile)
                            box = jsonData['box']
                            gt_name = jsonData['classId'][0]
                            gt_id = config.CLASS_NAME.index(gt_name)
                            centerX = box['centerX'][0]
                            centerY = box['centerY'][0]
                            widthBox = box['widthBox'][0]
                            heightBox = box['heightBox'][0]
                        
                        anchors = " ".join(str(a) for a in [centerX, centerY, widthBox, heightBox])                   
                        anchors = np.fromstring(anchors, dtype=int, sep=' ')
                        values = np.reshape(anchors, (anchors.shape[0], 1))
                        keys = ['centerX', 'centerY', 'widthBox', 'heightBox']
                        box = dict(zip(keys, values))
                        image, _ = load_and_crop(imagePath, config.INPUT_SIZE)

                        try:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        except:
                            pass
                        
                        textToDraw = ""
                        if gt_name == "Pass":
                            confusionMatrix[1][0] += 1
                        else:
                            confusionMatrix[0][0] += 1

                        pred_id, pred_score, pred_name = model.predict_one(image)
                        imageName = os.path.basename(imagePath)
                        print(f"Predicting image: {imagePath}")
                        if gt_name == "Pass":
                            if gt_name != pred_name:
                                if pred_name == "Unknown":
                                    confusionMatrix[1][2] += 1
                                    textToDraw = "Correct_Pass_With_Unknown%Score_{}".format(pred_score)
                                    color = colors[gt_id]
                                else:
                                    confusionMatrix[1][1] += 1
                                    textToDraw = "Overkill_{}%Score_{}".format(pred_name, pred_score)
                                    imageName = "Overkill_{}".format(imageName)
                                    color = colors[config.NUM_CLASSES + 1]
                            else:
                                confusionMatrix[1][2] += 1
                                textToDraw = "Correct_Pass%Score_{}".format(pred_score)
                                color = colors[gt_id]
                        else:
                            if pred_name == "Pass" or pred_name == "Unknown":
                                confusionMatrix[0][2] += 1
                                if pred_name == "Pass":
                                    textToDraw = "Underkill_{}%Score_{}".format(pred_name, pred_score)
                                else:
                                    textToDraw = "Underkill_{}%Score_{}".format(pred_name, pred_score)
                                imageName = "Underkill_{}".format(imageName)
                                color = colors[config.NUM_CLASSES + 5]
                            else:
                                confusionMatrix[0][1] += 1
                                textToDraw = "Correct_{}%Score_{}".format(pred_name, pred_score)
                                color = colors[gt_id]
                        
                        if savePath != "NONE":
                            originalImage = cv2.imread(imagePath)

                            x1 = int(centerX - widthBox/2)
                            y1 = int(centerY - heightBox/2)
                            x2 = int(centerX + widthBox/2)
                            y2 = int(centerY + heightBox/2)

                            # Write some Text

                            font                   = cv2.FONT_HERSHEY_SIMPLEX
                            bottomLeftCornerOfText = (x1-3, y1-3)
                            fontScale              = 0.3

                            cv2.putText(originalImage,textToDraw, 
                                bottomLeftCornerOfText, 
                                font, 
                                fontScale,
                                color=color)

                            cv2.rectangle(originalImage, (x1,y1), (x2,y2), color, 2)
                            print(os.path.join(savePath,imageName))
                            cv2.imwrite(os.path.join(savePath,imageName) ,originalImage)
                    end = time.time()

                    def ZeroDivisionError(a, b):
                        return a / b if b else 0

                    confusionMatrix[0][3] = ZeroDivisionError(confusionMatrix[0][1], confusionMatrix[0][0])
                    confusionMatrix[1][3] = ZeroDivisionError(confusionMatrix[1][2], confusionMatrix[1][0])
                    confusionMatrix[0][4] = ZeroDivisionError(confusionMatrix[0][2], confusionMatrix[0][0])
                    confusionMatrix[1][5] = ZeroDivisionError(confusionMatrix[1][1], confusionMatrix[1][0])
                    
                    result = "Matrix~%"
                    stringValueArray = []
                    for i in range(0,confusionMatrix.shape[0],1):
                        stringValueArray.append([])

                    for indexIn2DMatrix, value in np.ndenumerate(confusionMatrix):
                        # print("real value: {} {}".format(value,value.__class__))
                        value = np.nan_to_num(value)
                        if value >= 1.0:
                            stringValueArray[indexIn2DMatrix[0]].append(int(value))
                        elif value == 0.0:
                            stringValueArray[indexIn2DMatrix[0]].append(0)
                        else:
                            stringValueArray[indexIn2DMatrix[0]].append(np.round(value,decimals=2))
                    for i in stringValueArray:
                        result += "Row" + "_" + str(i) + "_" + "%"
                    result += "Done in test"
                    print(result)
                
                else:
                    result+="Finish to generate confusion matrix"
                    print("Finish to generate confusion matrix")
                    break
    except Exception as e:
        print(traceback.format_exc())
        print(f'[ERROR] {e}', file=sys.stderr)
    finally:
        print('[INFO] End of process.')