import numpy as np
import json
import argparse
import os
import re
import shutil
from datetime import datetime
import warnings

# Suppress warnings to clean up output
warnings.filterwarnings("ignore")

# Import evaluation functions for different tasks
from src.captioning import evaluate_captioning
from src.classification import evaluate_classification
from src.segmentation import evaluate_segmentation
from src.detection import evaluate_detection
from src.detection_obb import evaluate_detection2
from src.vqa import evaluate_vqa
import const

# Load constants from the `const` module
temp_folder = const.TEMP_FOLDER
cls_tags = const.CLS_TAGS
cap_tags = const.CAP_TAGS
vqa_tags = const.VQA_TAGS
det_tags = const.DET_TAGS
seg_tags = const.SEG_TAGS


def extract_roi (input_string, pattern = r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(\d+)>"):
    # Regular expression pattern to capture the required groups
    pattern = pattern
    # Find all matches
    matches = re.findall(pattern, input_string)
    
    # Extract the values
    extracted_values = [match for match in matches]

    return extracted_values

def round_up_to_nearest(x):
    if x <= 0:
        return 0  # Handle non-positive numbers
    magnitude = 10 ** (len(str(x)) - 1)
    if x == magnitude:
        return x
    else:
        return magnitude * 10


def calculate_iou_hbb(box1, box2):
    x_min_inter = float(max(float(box1[0]), float(box2[0])))
    y_min_inter = float(max(float(box1[1]), float(box2[1])))
    x_max_inter = float(min(float(box1[2]), float(box2[2])))
    y_max_inter = float(min(float(box1[3]), float(box2[3])))

    inter_area = float(max(0, x_max_inter - x_min_inter)) * float(max(0, y_max_inter - y_min_inter))
    box1_area = (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1]))
    box2_area = (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1]))
    union_area = box1_area + box2_area - inter_area+0.1

    iou = float(inter_area) / float(union_area)
    return iou

def prepare_detection_format(file_name = '/home/datadisk/evaluation/results/mPLUG-Owl2/json/'+'en_mPLUG-Owl2_rs_reg_det.json', analytic = False):

    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    print ("save HBB", folder_name)
    temp_dir = temp_folder+folder_name+'/det/'
    file_json = open(file_name)
    json_obj =  json.load(file_json)
    
    try:
        info_json = json_obj['info']
        result_task = info_json['task']
    except:
        result_task = json_obj['task']
    data_json = json_obj['data']

    #delete old data
    if os.path.exists(temp_dir+"ground-truth"):
        shutil.rmtree(temp_dir+"ground-truth")
    if os.path.exists(temp_dir+"detection-results"):
        shutil.rmtree(temp_dir+"detection-results")

    #create new data for new test
    if not os.path.exists(temp_dir+"ground-truth"):
        os.makedirs(temp_dir+"ground-truth")
    if not os.path.exists(temp_dir+"detection-results"):
        os.makedirs(temp_dir+"detection-results")


    max_answer = 0
    max_gt = 0
    
    multiplyer_ans = 1
    multiplyer_gt = 1
    
    for data_j in data_json:
        answer_roi = data_j['answer']
        gt_roi = data_j['gt']
        if not isinstance(gt_roi, list):  
            gt_roi = extract_roi(gt_roi.replace(" ",""), pattern = '<(\d+)><(\d+)><(\d+)><(\d+)>')
    
        if not isinstance(answer_roi, list):
            answer_roi = extract_roi(answer_roi.replace(" ",""), pattern = '<(\d+)><(\d+)><(\d+)><(\d+)>')
    
        for gt in gt_roi:
            if max_gt < int(max(gt)):
                max_gt = int(max(gt))
    
        for ans in answer_roi:
            if max_answer < int(max(ans)):
                max_answer = int(max(ans))
    
    
    max_answer = round_up_to_nearest(max_answer)
    max_gt = round_up_to_nearest(max_gt)
    try:
        if max_gt>max_answer:
            multiplyer_ans=max_gt/max_answer
        elif max_gt<max_answer:
            multiplyer_gt = max_answer/max_gt
    except:
        None

    ##TEST
    if True:
        multiplyer_ans = 1
        multiplyer_gt = 1
    for data_j in data_json:
        class_name = 'PIPE'
        
        image_file = data_j['image']
        if isinstance(image_file, list):
            image_file=image_file[0]

       
        image_name = image_file[image_file.rfind('/')+1: image_file.rfind('.')]
        #print (temp_dir+"ground-truth/"+image_name+".txt")
        time_now_file = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        ground_file = open(temp_dir+"ground-truth/"+image_name+time_now_file+".txt", 'a')
        res_file = open(temp_dir+"detection-results/"+image_name+time_now_file+'.txt', 'a')
        
        answer_roi = data_j['answer']
        gt_roi = data_j['gt']
        if not isinstance(gt_roi, list):
            gt_roi = extract_roi(gt_roi.replace(" ",""), pattern = '<(\d+)><(\d+)><(\d+)><(\d+)>')

        if not isinstance(answer_roi, list):
            answer_roi = extract_roi(answer_roi.replace(" ",""), pattern = '<(\d+)><(\d+)><(\d+)><(\d+)>')

        ###calculate IOU
        answer_iou = np.zeros(len(answer_roi))
        for i, pred_box in enumerate(answer_roi):
            iou_threshold = 0
            for j, gt_box in enumerate(gt_roi):
                iou = calculate_iou_hbb(pred_box, gt_box)
                if iou >= iou_threshold:
                    answer_iou[i] = iou
                    iou_threshold = iou

        ###
        
        if analytic:
            if len(gt_roi) >2:
                class_name = class_name+'s'#question_dic['text'][question_dic['text'].index("<p>")+3:question_dic['text'].index("</p>")]
            else:
                class_name = str(len(gt_roi))+'_'+class_name

        HBB = True
        for a_roi, a_iou in zip(answer_roi,answer_iou):
            cx = (int(a_roi[0])+int(a_roi[2]))/2
            cy = (int(a_roi[1])+int(a_roi[3]))/2
            wi = int(a_roi[2])-int(a_roi[0])
            hi = int(a_roi[3])-int(a_roi[1])
            #hbb_box = obb_to_hbb(cx,cy,wi,hi,int(a_roi[4]))
            
            if HBB:
                write_result = class_name+' '+str(a_iou)+' '+str(int(a_roi[0])*multiplyer_ans)+' '+str(int(a_roi[1])*multiplyer_ans)+' '+str(int(a_roi[2])*multiplyer_ans)+' '+str(int(a_roi[3])*multiplyer_ans)
            else:
                write_result = class_name+' '+str(a_iou)+' '+str(int(hbb_box[0])*multiplyer_ans)+' '+str(int(hbb_box[1])*multiplyer_ans)+' '+str(int(hbb_box[2])*multiplyer_ans)+' '+str(int(hbb_box[3])*multiplyer_ans)
                
            res_file.write(write_result+"\n")
            
        for a_roi in gt_roi:
            write_result = class_name+' '+str(int(a_roi[0])*multiplyer_gt)+' '+str(int(a_roi[1])*multiplyer_gt)+' '+str(int(a_roi[2])*multiplyer_gt)+' '+str(int(a_roi[3])*multiplyer_gt)
            ground_file.write(write_result+"\n")
    
        ground_file.close()
        res_file.close()
    return temp_dir+"ground-truth/", temp_dir+"detection-results/", temp_dir, temp_folder+folder_name

def read_json_result(file_name):
    
    file_json = open(file_name)
    json_obj =  json.load(file_json)
    
    try:
        info_json = json_obj['info']
        result_task = info_json['task']
        result_model = info_json['model']
        result_dataset = info_json['dataset']
    except:
        result_task = json_obj['task']
        result_model = json_obj['model']
        result_dataset = json_obj['dataset']
    data_json = json_obj['data']
    y_true = []
    y_pred = []
    class_names = []
    isCAP = False
    isList = False
    image_size = []
    if result_task in cap_tags:
        isCAP = True
    elif result_task in seg_tags:
        isList = True
    for data_i in data_json:
        try:
            if isCAP:
                y_true.append(data_i['gt'].lower())
                #print ('CAP')
            elif isList:
                y_true.append(data_i['gt'])
            else:
                y_true.append(data_i['gt'].lower().replace(".",""))
        except:
            if isCAP:
                y_true.append(str(data_i['gt']).lower())
            elif isList:
                y_true.append(data_i['gt'])
            else:
                y_true.append(str(data_i['gt']).lower().replace(".","").replace("<",""))
                
        try:
            if isCAP:
                y_pred.append(data_i['answer'].lower())
            elif isList:
                y_pred.append(data_i['answer'])
                image_size.append(data_i['crop'])
            else:
                y_pred.append(data_i['answer'].lower().replace(".","").replace("<",""))

        except:
            if isCAP:
                y_pred.append(str(data_i['answer']).lower())
            elif isList:
                y_pred.append(data_i['answer'])
                image_size.append(data_i['crop'])
            else:
                y_pred.append(str(data_i['answer']).lower().replace(".","").replace("<",""))

        try:
            if data_i['gt'].lower().replace(".","") not in class_names:
                class_names.append(data_i['gt'].lower().replace(".",""))
        except:
            continue
    
    if result_task in cls_tags:
        
        accuracy, precision_macro, recall_macro, f1_macro, micro = evaluate_classification(y_true, y_pred, class_names)
        result_dic = {'info':{'task':result_task,
                              'model':result_model,
                              'dataset':result_dataset},
                     'results':{'accuracy':accuracy,
                               'precision':precision_macro,
                               'recall':recall_macro,
                               'f1':f1_macro,
                               'micro':micro}}
        
        
    elif result_task in cap_tags:

        bleu, metero_score, rouge_score, rouge_l, CIDEr = evaluate_captioning(y_true, y_pred)
        
        result_dic = {'info':{'task':result_task,
                              'model':result_model,
                              'dataset':result_dataset},
                     'results':{'bleu':bleu,
                               'metero':metero_score,
                               'rouge':rouge_score,
                                'rouge_l':rouge_l,
                                'CIDEr':CIDEr
                               }}
    elif result_task in seg_tags:
        m_dice, m_iou, m_vc8, m_vc16 = evaluate_segmentation(y_true, y_pred, image_size)
        result_dic = {'info':{'task':result_task,
                              'model':result_model,
                              'dataset':result_dataset},
                     'results':{'mdice':m_dice,
                                'miou':m_iou,
                                #'mvc8':m_vc8,
                                #'mvc16':m_vc16
                               }}
        
    elif result_task in det_tags:
        isOBB = False
        isBox = False
        pre_box = False
        if "obb" in result_task.lower():
            isOBB = True

        answer_roi = data_json[0]['answer']
        gt_roi = data_json[0]['gt']
        if not isinstance(gt_roi, list):  
            isBox = True
            
        if not isinstance(answer_roi, list):  
            pre_box = True
        detection_result = {}
        iou_res = {}
        if isOBB:
            detection_result = evaluate_detection2(data_json, pre_box=pre_box, is_box=isBox, is_obb=isOBB)
        elif "vg" in result_task.lower():# or True:
            #print ("VG")
            detection_result = evaluate_detection2(data_json, pre_box=pre_box, is_box=isBox, is_obb=isOBB)
        
        else:
            # ##Detection v. 1
            print ("HBB",file_name)
            gt_roi_folder, res_roi_folder, temp_save_folder, master_folder = prepare_detection_format(file_name)
            
            ious = [0.5]#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            
            for iou in ious:
                print (f'IoU@{iou}')
                det_map = evaluate_detection(GT_PATH = gt_roi_folder, 
                           DR_PATH=res_roi_folder, 
                           TEMP_FILES_PATH = temp_save_folder+"temps", 
                           output_files_path= temp_save_folder+"output_detection/",  
                           iou=iou,
                           draw_plot = False)
                iou_res['mAP@'+str(iou)] = det_map

            #delete old data
            if os.path.exists(gt_roi_folder):
                shutil.rmtree(gt_roi_folder)
            if os.path.exists(res_roi_folder):
                shutil.rmtree(res_roi_folder)
            if os.path.exists(temp_save_folder):
                shutil.rmtree(temp_save_folder)
            if os.path.exists(master_folder):
                shutil.rmtree(master_folder)
        
        # ## end of detection V.1
        result_dic = {'info':{'task':result_task,
                              'model':result_model,
                              'dataset':result_dataset},
                     'results':{**detection_result, **iou_res}}#detection_result.update(iou_res)}#{'Evaluation of detection task currently under maintenance. Please contact Mr.Pipe'}}

    elif result_task in vqa_tags:
        
        accuracy, precision_macro, recall_macro, f1_macro, micro = evaluate_classification(y_true, y_pred, class_names)
        accuracy, precision, recall, f1 = evaluate_vqa(y_true, y_pred)

        result_dic = {'info':{'task':result_task,
                              'model':result_model,
                              'dataset':result_dataset},
                     'results':{'accuracy':accuracy,
                                #'precision':precision,
                                #'recall':recall,
                               'f1':f1
                               }}
    else:
        print ('error', result_task)
        result_dic = {'info':{'task':result_task,
                              'model':result_model,
                              'dataset':result_dataset}}

    return result_dic


def evaluation(args):
    evaluation_folder = args.evaluation_folder
    debuging = False
    isReplace = args.isReplace
    if args.debug == "yes":
        debuging = True
    save_evaluation_result = args.save_path
    save_result_folder_dir = save_evaluation_result+args.model_name+"/"
    if not os.path.exists(save_result_folder_dir):
        os.makedirs(save_result_folder_dir)
    results_files = os.listdir(save_result_folder_dir)
    if evaluation_folder == '':
        evaluation_path = args.evaluation_file
        if evaluation_path == '':
            result_dic = 'please spacific --evaluation-file or --evaluation-folder'
        else:
            if not debuging:
                file_name = evaluation_path.split('/')[-1]
                save_file_name = '['+args.model_name+"]"+file_name
                if save_file_name in os.listdir(save_result_folder_dir) and not isReplace:
                    print(f'{file_name} finish already!')
                else:
                    result_dic = read_json_result(evaluation_path)
                    print (f'{result_dic}')
        
                    final_json = json.dumps(result_dic)
                    with open(save_result_folder_dir+'['+args.model_name+"]"+file_name, "w") as outfile:
                        outfile.write(final_json)
            else:
                result_dic = read_json_result(evaluation_path)
                file_name = evaluation_path.split('/')[-1]
                print (f'{result_dic}')

                print (result_dic)
                final_json = json.dumps(result_dic)
    else:
        with open(save_result_folder_dir+"log.txt", "w") as log_file:
            log_file.write("missing file:\n")
        for jsons in os.listdir(evaluation_folder):
            if (jsons.endswith(".json")):
                save_file_name = '['+args.model_name+"]"+jsons
                if save_file_name in os.listdir(save_result_folder_dir) and not isReplace:
                    print(f'{jsons} finish already!')
                else:
                    evaluation_path = evaluation_folder+"/"+jsons
                    print(f'Evaluate:{evaluation_path}')

                    if not debuging:
                        try:
                            result_dic = read_json_result(evaluation_path)
                            
                            print ('done read')
                            print(f'{result_dic}')
                            print ()
                        
                            final_json = json.dumps(result_dic)
                            with open(save_result_folder_dir+'['+args.model_name+"]"+jsons, "w") as outfile:
                                outfile.write(final_json)
                        except Exception as e:
                            with open(save_result_folder_dir+"log.txt", "a") as log_file:
                                log_file.write(f"Error processing {jsons}: {e}\n")
                    else:
                        result_dic = read_json_result(evaluation_path)
                        
                        print ('done read')
                        print(f'{result_dic}')
                        print ()
                    
                        final_json = json.dumps(result_dic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-file", type=str, default="")
    parser.add_argument("--evaluation-folder", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--isReplace", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default='results/20241010/')
    parser.add_argument("--debug", type=str, default='no')
    args = parser.parse_args()

    evaluation(args)
