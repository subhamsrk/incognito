import os
import math
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from PIL import Image, ImageEnhance
from doctr.models import ocr_predictor
from tensorflow.keras.models import load_model
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
model_cropper = ocr_predictor(det_arch = 'db_resnet50',    
                      reco_arch = 'crnn_vgg16_bn',
                      pretrained = True
                     )
model_ocr = ocr_predictor(det_arch = 'db_resnet50',    
                      reco_arch = 'crnn_vgg16_bn',
                      pretrained = True,
                      resolve_blocks=False,
                      resolve_lines=False,
                      paragraph_break= 0.07
                     )
model_classifier = load_model('models/2_4')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model_handwritten = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten',cache_dir='/data/OCR/models/')
 
def save_imgs(dirname,imgs):
    with open(dirname,'wb') as f:
        pickle.dump(imgs,f)
    f.close()
   
def convert_coordinates(geometry, page_dim=(2500,2500)):
    len_x = page_dim[1]
    len_y = page_dim[0]
    (x_min, y_min) = geometry[0]
    (x_max, y_max) = geometry[1]
    x_min = math.floor(x_min * len_x)
    x_max = math.ceil(x_max * len_x)
    y_min = math.floor(y_min * len_y)
    y_max = math.ceil(y_max * len_y)
    return [x_min, x_max, y_min, y_max]
 
def seal_number(s):
    if s.isnumeric():
        return True
    elif s.isalpha():
        return False
    return ((any(char.isdigit() for char in s)) and (len(s) > 6))
   
def model_run(image,processor,model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    outputs = model.generate(pixel_values, max_new_tokens=12, output_scores=True, return_dict_in_generate=True)
    generated_text = processor.batch_decode(outputs.sequences,skip_special_tokens=True)[0]
   
    return generated_text
 
dirname =  'incognito/'
print(len(os.listdir(dirname)))
dirname_crp = os.path.join(dirname,'bol_cropped')
imgs_pkl_path = os.path.join(dirname,'images.pkl')
imgs = {}
if(not os.path.exists(dirname_crp)):
    print('Creating directory for cropped bols')
    os.mkdir(dirname_crp)
    os.mkdir(os.path.join(dirname_crp,'bols'))
print('Cropped bols directory created')
if(os.path.exists(imgs_pkl_path)):
    print('Pickle file loading from memory')
    f = open(imgs_pkl_path,'rb')
    imgs = pickle.load(f)
    f.close()
    print(len(imgs))
po_matched_df = pd.read_csv('additionalmatchedpo.csv')
po_matched_files = set(po_matched_df['File name'].unique())
print(len(po_matched_files))
for c,file in enumerate(os.listdir(dirname)):
    if('.jpeg' not in file):
        continue
    if(file in imgs or file not in po_matched_files):
        continue
    img_path = dirname+file
    img = DocumentFile.from_images(img_path)
    result = model_cropper(img)
    imgs[file] = result.export()
    if(len(imgs)%100 == 0):
        save_imgs(imgs_pkl_path,imgs)
        print('Saved at {}'.format(len(imgs)))
save_imgs(imgs_pkl_path,imgs)
print(len(imgs))
seal_dict = {}
for outputs in imgs.keys():
    seal_list = []
    for page in imgs[outputs]['pages'][0]['blocks']:
        for line in page['lines']:
            for word in line['words']:
                if('seal' in word['value'].lower() and 'sealy' not in word['value'].lower() and 'sealed' not in word['value'].lower() and word['confidence'] >= 0.9):
                    seal_list.append(word)
    seal_dict[outputs] = seal_list if(len(seal_list) != 0) else None
print(len(seal_dict))
boxes_dict = {}
for seal_list in seal_dict.keys():
    boxes = []
    if(seal_dict[seal_list] == None):
        boxes_dict[seal_list] = None
        continue
    for seal in seal_dict[seal_list]:
        boxes.append(convert_coordinates(seal['geometry'],imgs[seal_list]['pages'][0]['dimensions']))
    boxes_dict[seal_list] = boxes
no_seal = []
for img_path in boxes_dict.keys():
    image = Image.open(dirname+img_path)
    c = 1
    if(boxes_dict[img_path] == None):
        no_seal.append(img_path)
        continue
    for box in boxes_dict[img_path]:
        div = (int(2500/image.size[0]) if(int(2500/image.size[0])>1) else 1,int(3300/image.size[1]) if(int(3300/image.size[1]) > 1) else 1)
        img = image.crop((box[0],box[2]-10/div[1],box[0]+750/div[0],box[2]+55/div[1])).resize((750,65))
        if(div[0]>1 and div[1]>1):
            plt.imshow(img)
            plt.show()
            cont_obj = ImageEnhance.Contrast(img)
            img = cont_obj.enhance(1.5)
            plt.imshow(img)
            plt.show()
            shrp_obj = ImageEnhance.Sharpness(img)
            img = shrp_obj.enhance(10.0)
        plt.imshow(img)
        plt.show()
#         print(img.size)
        img.save(os.path.join(dirname_crp,'bols','{}__{}'.format(c,img_path)))
        c+=1
print('Seal not found or not confident for {} records'.format(len(no_seal)))
with open(dirname+'no_seal.txt','w') as f:
    for i in no_seal:
        f.write(i+'\n')
f.close()
data_gen = ImageDataGenerator(rescale=1/255)
generator = data_gen.flow_from_directory(dirname_crp,batch_size=1,shuffle=False,class_mode='categorical',target_size=(65,750))
id2label = {0: 'printed', 1: 'empty', 2: 'handwritten', 3: 'stamp_details'}
n = len(generator.filenames)
print(n)
predictions = model_classifier.predict(generator,steps=n,use_multiprocessing=True,workers=-1)
classes = predictions.argmax(axis=-1)
df = pd.DataFrame(columns=['File','Type','Type Confidence','Value','Value Confidence'])
# result of missed docs from previous models
data = []
for c,img in enumerate(generator.filenames):
    if(c%100 == 0):
        df.to_csv(os.path.join(dirname,'seal_data.csv'),index=False)
    data.append(img.split('__')[1])
    if('.jpeg' not in img):
        continue
    img_add = os.path.join(dirname_crp,img)
    image = Image.open(img_add)
    plt.imshow(image)
    plt.show()
    print("Class predicted is '{}' with probability {} \n".format(id2label[classes[c]],round(predictions[c][classes[c]],2)))
    data.append(id2label[classes[c]])
    data.append(round(predictions[c][classes[c]],2))
    img = DocumentFile.from_images(os.path.join(dirname_crp,img))
    result = model_ocr(img).export()
    data.append('')
    data.append('NA')
    if(id2label[classes[c]] == 'printed' and round(predictions[c][classes[c]],2) >= 0.9):
        #extract with algorithm
        for word in result['pages'][0]['blocks'][0]['lines'][0]['words']:
            if(not seal_number(word['value'])):
                continue
#             if(word['confidence'] < 0.6):
#                 continue
#             print('Seal Number is {}'.format(word['value']))
            if(len(data[3]) >= 4 and len(data[3]) <= 10 and (len(word['value'])<4 or word['confidence'] < 0.6 or word['confidence'] < data[4])):
                continue
            val = word['value']
            if(':' in val and val[len(val)-1] != ':'):
                val = val.split(':')[1]
            elif('#' in val and val[len(val)-1] != '#'):
                val = val.split('#')[1]
            elif(')' in val and val[len(val)-1] != ')'):
                val = val.split(')')[1]
            elif('(' in val and val[len(val)-1] != '('):
                val = val.split('(')[1]
            val = val.replace(':','').replace('#','').replace('.','').replace('NUMBER','').replace('SEAL','')
            data[3] = '{}'.format(val) if(len(val) > 3 and len(val) <= 10) else 'SE#'
            data[4] = word['confidence']
    elif(id2label[classes[c]] == 'empty' and round(predictions[c][classes[c]],2) >= 0.9):
        data[3] = 'Empty'
        data[4] = 1
    df.loc[len(df)] = data
    data = []
df.loc[(df['Type Confidence'] >= 0.9) & (df['Type'].isin(['empty','printed']))]
df.to_csv(os.path.join(dirname,'data.csv'),index=False)
df_empty = pd.DataFrame(columns=['File','Seal Detected','Confidence'])
for i in no_seal:
    ls = [i]
    found = False
    for page in imgs[i]['pages'][0]['blocks']:
        for line in page['lines']:
            for word in line['words']:
                if('seal' in word['value'].lower() and 'sealy' not in word['value'].lower() and 'sealed' not in word['value'].lower()):
                    ls.append('Y')
                    ls.append(word['confidence'])
                    found = True
                    break
            if(found):
                break
        if(found):
            break
    if(not found):
        ls.append('N')
        ls.append('NA')
    df_empty.loc[len(df_empty)] = ls
df_empty.to_csv(os.path.join(dirname,'withoutseal.csv'),index=False)
seal_ann_2 = pd.read_csv(os.path.join(dirname,'withoutseal.csv'))
len(seal_ann_2)
for c,img in enumerate(generator.filenames):
    if('.jpg' not in img):
        continue
    image = Image.open(os.path.join(dirname_crp,img))
    plt.imshow(image)
    plt.show()
    print("Class predicted is '{}' with probability {} \n".format(id2label[classes[c]],round(predictions[c][classes[c]],2)))
    if(id2label[classes[c]] == 'printed' and round(predictions[c][classes[c]],2) >= 0.9):
        #extract with algorithm
        img = DocumentFile.from_images(os.path.join(dirname_crp,img))
        result = model_ocr(img).export()
        helper_dict = {}
        for word in result['pages'][0]['blocks'][0]['lines'][0]['words']:
            if(not seal_number(word['value'])):
                continue
            if(word['confidence'] < 0.6):
                continue
            print('Seal Number is {}'.format(word['value']))
for c,img in enumerate(generator.filenames):
    if('.jpeg' not in img):
        continue
    image = Image.open(os.path.join(dirname_crp,img))
    plt.imshow(image)
    plt.show()
    print("Class predicted is '{}' with probability {} \n".format(id2label[classes[c]],round(predictions[c][classes[c]],2)))
    if(id2label[classes[c]] == 'printed' and round(predictions[c][classes[c]],2) >= 0.9):
        #extract with algorithm
        img = DocumentFile.from_images(os.path.join(dirname_crp,img))
        result = model_ocr(img).export()
        helper_dict = {}
        for word in result['pages'][0]['blocks'][0]['lines'][0]['words']:
            if(not seal_number(word['value'])):
                continue
            if(word['confidence'] < 0.6):
                continue
            print('Seal Number is {}'.format(word['value']))
    if(id2label[classes[c]] == 'handwritten' and round(predictions[c][classes[c]],2) >= 0.9):
        print('Seal Number is {}'.format(model_run(image,processor,model_handwritten)))
