from __future__ import absolute_import
from __future__ import division
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc, signal, spatial,sparse,io
from skimage.filters import gaussian
import cv2
import math
from time import time
import sys
from keras.models import Model
from keras.layers import Input,Concatenate,GlobalAveragePooling2D
from keras import layers
from keras.layers.core import Flatten,Activation,Lambda, Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D,UpSampling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from functools import reduce
import glob
import logging
import json
import CNAI
import shutil
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adadelta
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
import collections

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
sess = K.tf.Session(config=config)
K.set_session(sess)

mode = 'inference'
root=os.getcwd()
# Can use multiple folders for deploy, inference
inference_set = ['../Dataset/CoarseNet_test/',]

pretrain_dir = '../Models/CoarseNet.h5'
cnai_model_dir='../Models/cnai_weight.h5'
output_dir = '../output_CoarseNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')

FineNet_dir = '../Models/FineNet.h5'

def main():
    if mode == 'inference':
       # output_dir = '../output_CoarseNet/inferenceResults/' +datetime.now().strftime('%Y%m%d-%H%M%S')
        logging = init_log(output_dir)
        for i, folder in enumerate(inference_set):
            inference(folder, output_dir=output_dir, model_path=pretrain_dir, FineNet_path=FineNet_dir, file_ext='.bmp',
                      isHavingFineNet=False)
    else:
        pass

def init_log(output_dir):
    re_mkdir(output_dir)
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        filename=os.path.join(output_dir, 'log.log'),
        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def re_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def get_files_in_folder(folder, file_ext=None):
    files = glob.glob(os.path.join(folder, "*" + file_ext))
    files_name = []
    for i in files:
        _, name = os.path.split(i)
        name, ext = os.path.splitext(name)
        files_name.append(name)
    return np.asarray(files), np.asarray(files_name)
def nextpow2(x):
    return int(math.ceil(math.log(x, 2)))
def smooth_dir_map(dir_map,sigma=2.0,mask = None):

    cos2Theta = np.cos(dir_map * 2)
    sin2Theta = np.sin(dir_map * 2)
    if mask is not None:
        assert (dir_map.shape[0] == mask.shape[0])
        assert (dir_map.shape[1] == mask.shape[1])
        cos2Theta[mask == 0] = 0
        sin2Theta[mask == 0] = 0

    cos2Theta = gaussian(cos2Theta, sigma, multichannel=False, mode='reflect')
    sin2Theta = gaussian(sin2Theta, sigma, multichannel=False, mode='reflect')

    dir_map = np.arctan2(sin2Theta,cos2Theta)*0.5


    return dir_map

def compute_gradient_norm(input):
    input = input.astype(np.float32)

    Gx, Gy = np.gradient(input)
    out = np.sqrt(Gx * Gx + Gy * Gy) + 0.000001
    return out
def get_ridge_flow_top(local_info):

    blkH,blkW = local_info.shape
    dir_map = np.zeros((blkH,blkW)) - 10
    fre_map = np.zeros((blkH, blkW)) - 10
    for i in range(blkH):
        for j in range(blkW):
            if local_info[i,j].ori is None:
                continue

            dir_map[i,j] = local_info[i,j].ori[0] #+ math.pi*0.5
            fre_map[i,j] = local_info[i,j].fre[0]
    return dir_map,fre_map

def inference(deploy_set, output_dir, model_path, FineNet_path=None, set_name=None, file_ext='.bmp', isHavingFineNet = False):
    if set_name is None:
        set_name = deploy_set.split('/')[-2]


    mkdir(output_dir + '/'+ set_name + '/')
    mkdir(output_dir + '/' + set_name + '/mnt_results/')
    mkdir(output_dir + '/'+ set_name + '/seg_results/')
    mkdir(output_dir + '/' + set_name + '/OF_results/')

    logging.info("Predicting %s:" % (set_name))

    _, img_name = get_files_in_folder(deploy_set+ 'img_files/', file_ext)
    print(deploy_set)

    time_c = []

    main_net_model = CoarseNetmodel((None, None, 1), model_path, mode='inference')

    for i in range(0, len(img_name)):
        print(i)

        image = misc.imread(deploy_set + 'img_files/'+ img_name[i] + file_ext, mode='L')  # / 255.0

        img_size = image.shape
        img_size = np.array(img_size, dtype=np.int32) // 8 * 8

        # read the mask from files
        try:
            mask = misc.imread(deploy_set + 'seg_files/' + img_name[i] + '.jpg', mode='L') / 255.0
        except:
            mask = np.ones((img_size[0],img_size[1]))
        image = image[:img_size[0], :img_size[1]]
        mask = mask[:img_size[0], :img_size[1]]

        original_image = image.copy()

        texture_img = FastEnhanceTexture(image, sigma=2.5, show=False)
        dir_map, fre_map = get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)
        image = image*mask

        logging.info("%s %d / %d: %s" % (set_name, i + 1, len(img_name), img_name[i]))
        time_start = time()

        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out \
            = main_net_model.predict(image)
        time_afterconv = time()

        # If use mask from model
        round_seg = np.round(np.squeeze(seg_out))
        seg_out = 1 - round_seg
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_out = cv2.dilate(seg_out, kernel)
        # If use mask from outside
        max_num_minu = 20
        min_num_minu = 6

        early_minutiae_thres = 0.5
        # New adaptive threshold
        mnt = label2mnt(np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out)), mnt_w_out, mnt_h_out, mnt_o_out,
                        thresh=0)

        # Previous exp: 0.2
        mnt_nms_1 = py_cpu_nms(mnt, 0.5)
        mnt_nms_2 = nms(mnt)
        mnt_nms_1.view('f8,f8,f8,f8').sort(order=['f3'], axis=0)
        mnt_nms_1 = mnt_nms_1[::-1]

        mnt_nms_1_copy = mnt_nms_1.copy()
        mnt_nms_2_copy = mnt_nms_2.copy()
        # Adaptive threshold goes here
        # Make sure the maximum number of minutiae is max_num_minu

        # Sort minutiae by score
        while early_minutiae_thres > 0:
            mnt_nms_1 = mnt_nms_1_copy[mnt_nms_1_copy[:, 3] > early_minutiae_thres, :]
            mnt_nms_2 = mnt_nms_2_copy[mnt_nms_2_copy[:, 3] > early_minutiae_thres, :]

            if mnt_nms_1.shape[0]>max_num_minu or mnt_nms_2.shape[0]>max_num_minu:
                mnt_nms_1 = mnt_nms_1[:max_num_minu,:]
                mnt_nms_2 = mnt_nms_2[:max_num_minu, :]
            if mnt_nms_1.shape[0] > min_num_minu and mnt_nms_2.shape[0] > min_num_minu:
                break

            early_minutiae_thres = early_minutiae_thres - 0.05

        mnt_nms = fuse_nms(mnt_nms_1, mnt_nms_2)

        final_minutiae_score_threashold = early_minutiae_thres - 0.05

        print(early_minutiae_thres, final_minutiae_score_threashold)

        mnt_nms_backup = mnt_nms.copy()
        mnt_nms = np.array(mnt_nms)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:,3]>final_minutiae_score_threashold,:]

        final_mask = ndimage.zoom(np.round(np.squeeze(seg_out)), [8, 8], order=0)
        # Show the orientation
        show_orientation_field(original_image, dir_map + np.pi, mask=final_mask, fname="%s/%s/OF_results/%s_OF.jpg" % (output_dir, set_name, img_name[i]))

        fuse_minu_orientation(dir_map, mnt_nms, mode=3)

        time_afterpost = time()
        mnt_writer(mnt_nms, img_name[i], img_size, "%s/%s/mnt_results/%s.mnt"%(output_dir, set_name, img_name[i]))
        draw_minutiae(original_image, mnt_nms, "%s/%s/%s_minu.jpg"%(output_dir, set_name, img_name[i]),saveimage=True)

        misc.imsave("%s/%s/seg_results/%s_seg.jpg" % (output_dir, set_name, img_name[i]), final_mask)

        time_afterdraw = time()
        time_c.append([time_afterconv - time_start, time_afterpost - time_afterconv, time_afterdraw - time_afterpost])
        logging.info(
            "load+conv: %.3fs, seg-postpro+nms: %.3f, draw: %.3f" % (time_c[-1][0], time_c[-1][1], time_c[-1][2]))
    return

def draw_minutiae(image, minutiae, fname, saveimage= False, r=15, drawScore=True):
    image = np.squeeze(image)
    fig = plt.figure()
    

    plt.imshow(image,cmap='gray')
    
    # Check if no minutiae
    if minutiae.shape[0] > 0:
        plt.plot(minutiae[:, 0], minutiae[:, 1], 'rs', fillstyle='none', linewidth=1)
        for x, y, o, s in minutiae:
            #plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'r-')
            if drawScore == True:
                plt.text(x - 10, y - 10, '%.2f' % s, color='yellow', fontsize=4)

    plt.axis([0,image.shape[1],image.shape[0],0])
    plt.axis('off')
    if saveimage:
        plt.savefig(fname, dpi=500, bbox_inches='tight', pad_inches = 0)
        plt.close(fig)
    else:
        plt.show()
    return
def mnt_writer(mnt, image_name, image_size, file_name):
    f = open(file_name, 'w')
    f.write('%s\n'%(image_name))
    f.write('%d %d %d\n'%(mnt.shape[0], image_size[0], image_size[1]))
    for i in range(mnt.shape[0]):
        f.write('%d %d %.6f %.4f\n'%(mnt[i,0], mnt[i,1], mnt[i,2], mnt[i,3]))
    f.close()
    return
def copy_file(path_s, path_t):
    shutil.copy(path_s, path_t)  
def conv_bn_prelu(bottom, w_size, name, strides=(1,1), dilation_rate=(1,1)):
    if dilation_rate == (1,1):
        conv_type = 'conv'
    else:
        conv_type = 'atrousconv'

    top = Conv2D(w_size[0], (w_size[1],w_size[2]),
        kernel_regularizer=l2(5e-5),
        padding='same',
        strides=strides,
        dilation_rate=dilation_rate,
        name=conv_type+name)(bottom)
    top = BatchNormalization(name='bn-'+name)(top)
    top = PReLU(alpha_initializer='zero', shared_axes=[1,2], name='prelu-'+name)(top)
    # top = Dropout(0.25)(top)
    return top

def CoarseNetmodel(input_shape=(400,400,1), weights_path=None, mode='train'):
    # Change network architecture here!!
    img_input=Input(input_shape)
    bn_img=Lambda(img_normalization, name='img_normalized')(img_input)

    # Main part
    conv = conv_bn_prelu(bn_img, (64, 5, 5), '1_0')
    conv = conv_bn_prelu(conv, (64, 3, 3), '1_1')
    conv = conv_bn_prelu(conv, (64, 3, 3), '1_2')
    conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

    # =======Block 1 ========
    conv1 = conv_bn_prelu(conv, (128, 3, 3), '2_1')
    conv = conv_bn_prelu(conv1, (128, 3, 3), '2_2')
    conv = conv_bn_prelu(conv, (128, 3, 3), '2_3')
    conv = layers.add([conv, conv1])

    conv1 = conv_bn_prelu(conv, (128, 3, 3), '2_1b')
    conv = conv_bn_prelu(conv1, (128, 3, 3), '2_2b')
    conv = conv_bn_prelu(conv, (128, 3, 3), '2_3b')
    conv = layers.add([conv, conv1])

    conv1 = conv_bn_prelu(conv, (128, 3, 3), '2_1c')
    conv = conv_bn_prelu(conv1, (128, 3, 3), '2_2c')
    conv = conv_bn_prelu(conv, (128, 3, 3), '2_3c')
    conv = layers.add([conv, conv1])

    conv_block1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)
    # ==========================

    # =======Block 2 ========
    conv1 = conv_bn_prelu(conv_block1, (256,3,3), '3_1')
    conv = conv_bn_prelu(conv1, (256,3,3), '3_2')
    conv = conv_bn_prelu(conv, (256,3,3), '3_3')
    conv = layers.add([conv, conv1])

    conv1 = conv_bn_prelu(conv, (256, 3, 3), '3_1b')
    conv = conv_bn_prelu(conv1, (256, 3, 3), '3_2b')
    conv = conv_bn_prelu(conv, (256, 3, 3), '3_3b')
    conv = layers.add([conv, conv1])

    conv_block2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)
    # ==========================

    # =======Block 3 ========
    conv1 = conv_bn_prelu(conv_block2, (512, 3, 3), '3_1c')
    conv = conv_bn_prelu(conv1, (512, 3, 3), '3_2c')
    conv = conv_bn_prelu(conv, (512, 3, 3), '3_3c')
    conv = layers.add([conv, conv1])
    conv_block3 = conv_bn_prelu(conv, (256, 3, 3), '3_4c')

    #conv_block3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)
    # ==========================


    # multi-scale ASPP
    level_2=conv_bn_prelu(conv_block3, (256,3,3), '4_1', dilation_rate=(1,1))
    ori_1=conv_bn_prelu(level_2, (128,1,1), 'ori_1_1')
    ori_1=Conv2D(90, (1,1), padding='same', name='ori_1_2')(ori_1)
    seg_1=conv_bn_prelu(level_2, (128,1,1), 'seg_1_1')
    seg_1=Conv2D(1, (1,1), padding='same', name='seg_1_2')(seg_1)

    level_3=conv_bn_prelu(conv_block2, (256,3,3), '4_2', dilation_rate=(4,4))
    ori_2=conv_bn_prelu(level_3, (128,1,1), 'ori_2_1')
    ori_2=Conv2D(90, (1,1), padding='same', name='ori_2_2')(ori_2)
    seg_2=conv_bn_prelu(level_3, (128,1,1), 'seg_2_1')
    seg_2=Conv2D(1, (1,1), padding='same', name='seg_2_2')(seg_2)

    level_4=conv_bn_prelu(conv_block2, (256,3,3), '4_3', dilation_rate=(8,8))
    ori_3=conv_bn_prelu(level_4, (128,1,1), 'ori_3_1')
    ori_3=Conv2D(90, (1,1), padding='same', name='ori_3_2')(ori_3)
    seg_3=conv_bn_prelu(level_4, (128,1,1), 'seg_3_1')
    seg_3=Conv2D(1, (1,1), padding='same', name='seg_3_2')(seg_3)

    # sum fusion for ori
    ori_out=Lambda(merge_sum)([ori_1, ori_2, ori_3])
    ori_out_1=Activation('sigmoid', name='ori_out_1')(ori_out)
    ori_out_2=Activation('sigmoid', name='ori_out_2')(ori_out)

    # sum fusion for segmentation
    seg_out=Lambda(merge_sum)([seg_1, seg_2, seg_3])
    seg_out=Activation('sigmoid', name='seg_out')(seg_out)

    # ----------------------------------------------------------------------------
    # enhance part
    filters_cos, filters_sin = gabor_bank(stride=2, Lambda=8)

    filter_img_real = Conv2D(filters_cos.shape[3],(filters_cos.shape[0],filters_cos.shape[1]),
        weights=[filters_cos, np.zeros([filters_cos.shape[3]])], padding='same',
        name='enh_img_real_1')(img_input)
    filter_img_imag = Conv2D(filters_sin.shape[3],(filters_sin.shape[0],filters_sin.shape[1]),
        weights=[filters_sin, np.zeros([filters_sin.shape[3]])], padding='same',
        name='enh_img_imag_1')(img_input)

    ori_peak = Lambda(ori_highest_peak)(ori_out_1)
    ori_peak = Lambda(select_max)(ori_peak) # select max ori and set it to 1

    # Use this function to upsample image
    upsample_ori = UpSampling2D(size=(8,8))(ori_peak)
    seg_round = Activation('softsign')(seg_out)


    upsample_seg = UpSampling2D(size=(8,8))(seg_round)
    mul_mask_real = Lambda(merge_mul)([filter_img_real, upsample_ori])

    enh_img_real = Lambda(reduce_sum, name='enh_img_real_2')(mul_mask_real)
    mul_mask_imag = Lambda(merge_mul)([filter_img_imag, upsample_ori])

    enh_img_imag = Lambda(reduce_sum, name='enh_img_imag_2')(mul_mask_imag)
    enh_img = Lambda(atan2, name='phase_img')([enh_img_imag, enh_img_real])

    enhance_img = Lambda(merge_concat, name='phase_seg_img')([enh_img, upsample_seg])
    # ----------------------------------------------------------------------------
    # mnt part
    # =======Block 1 ========
    mnt_conv1 = conv_bn_prelu(enhance_img, (64, 9, 9), 'mnt_1_1')
    mnt_conv = conv_bn_prelu(mnt_conv1, (64, 9, 9), 'mnt_1_2')
    mnt_conv = conv_bn_prelu(mnt_conv, (64, 9, 9), 'mnt_1_3')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv1 = conv_bn_prelu(mnt_conv, (64, 9, 9), 'mnt_1_1b')
    mnt_conv = conv_bn_prelu(mnt_conv1, (64, 9, 9), 'mnt_1_2b')
    mnt_conv = conv_bn_prelu(mnt_conv, (64, 9, 9), 'mnt_1_3b')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv)
    # ==========================

    # =======Block 2 ========
    mnt_conv1 = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_1')
    mnt_conv = conv_bn_prelu(mnt_conv1, (128, 5, 5), 'mnt_2_2')
    mnt_conv = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_3')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv1 = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_1b')
    mnt_conv = conv_bn_prelu(mnt_conv1, (128, 5, 5), 'mnt_2_2b')
    mnt_conv = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_3b')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv)
    # ==========================

    # =======Block 3 ========
    mnt_conv1 = conv_bn_prelu(mnt_conv, (256, 3, 3), 'mnt_3_1')
    mnt_conv2 = conv_bn_prelu(mnt_conv1, (256, 3, 3), 'mnt_3_2')
    mnt_conv3 = conv_bn_prelu(mnt_conv2, (256, 3, 3), 'mnt_3_3')
    mnt_conv3 = layers.add([mnt_conv3, mnt_conv1])
    mnt_conv4 = conv_bn_prelu(mnt_conv3, (256, 3, 3), 'mnt_3_4')
    mnt_conv4 = layers.add([mnt_conv4, mnt_conv2])

    mnt_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv4)
    # ==========================


    mnt_o_1=Lambda(merge_concat)([mnt_conv, ori_out_1])
    mnt_o_2=conv_bn_prelu(mnt_o_1, (256,1,1), 'mnt_o_1_1')
    mnt_o_3=Conv2D(180, (1,1), padding='same', name='mnt_o_1_2')(mnt_o_2)
    mnt_o_out=Activation('sigmoid', name='mnt_o_out')(mnt_o_3)

    mnt_w_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_w_1_1')
    mnt_w_2=Conv2D(8, (1,1), padding='same', name='mnt_w_1_2')(mnt_w_1)
    mnt_w_out=Activation('sigmoid', name='mnt_w_out')(mnt_w_2)

    mnt_h_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_h_1_1')
    mnt_h_2=Conv2D(8, (1,1), padding='same', name='mnt_h_1_2')(mnt_h_1)
    mnt_h_out=Activation('sigmoid', name='mnt_h_out')(mnt_h_2)

    mnt_s_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_s_1_1')
    mnt_s_2=Conv2D(1, (1,1), padding='same', name='mnt_s_1_2')(mnt_s_1)
    mnt_s_out=Activation('sigmoid', name='mnt_s_out')(mnt_s_2)

      
    model = Model(inputs=[img_input,], outputs=[enh_img,enh_img_imag,enhance_img,ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out])

    if weights_path != None:
        model.load_weights(weights_path, by_name=True)
    return model
def LowpassFiltering(img,L):
    h,w = img.shape
    h2,w2 = L.shape

    img = cv2.copyMakeBorder(img, 0, h2-h, 0, w2-w, cv2.BORDER_CONSTANT, value=0)

    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    img_fft = img_fft * L
    rec_img = np.fft.ifft2(np.fft.fftshift(img_fft))
    rec_img = np.real(rec_img)
    rec_img = rec_img[:h,:w]

    return rec_img

def FastEnhanceTexture(img,sigma=2.5,show=False):
    img = img.astype(np.float32)
    h, w = img.shape
    h2 = 2 ** nextpow2(h)
    w2 = 2 ** nextpow2(w)

    FFTsize = np.max([h2, w2])
    x, y = np.meshgrid(range(int(-FFTsize / 2), int(FFTsize / 2)), range(int(-FFTsize / 2), int(FFTsize / 2)))
    r = np.sqrt(x * x + y * y) + 0.0001
    r = r/FFTsize

    L = 1. / (1 + (2 * math.pi * r * sigma)** 4)
    img_low = LowpassFiltering(img, L)

    gradim1=  compute_gradient_norm(img)
    gradim1 = LowpassFiltering(gradim1,L)

    gradim2=  compute_gradient_norm(img_low)
    gradim2 = LowpassFiltering(gradim2,L)

    diff = gradim1-gradim2
    ar1 = np.abs(gradim1)
    diff[ar1>1] = diff[ar1>1]/ar1[ar1>1]
    diff[ar1 <= 1] = 0

    cmin = 0.3
    cmax = 0.7

    weight = (diff-cmin)/(cmax-cmin)
    weight[diff<cmin] = 0
    weight[diff>cmax] = 1

    u = weight * img_low + (1-weight)* img
    temp = img - u
    lim = 20
    temp1 = (temp + lim) * 255 / (2 * lim)
    temp1[temp1 < 0] = 0
    temp1[temp1 >255] = 255
    v = temp1
    if show:
        plt.imshow(v,cmap='gray')
        plt.show()
    return v

def get_maps_STFT(img,patch_size = 64,block_size = 16, preprocess = False):
    assert len(img.shape) == 2

    nrof_dirs = 16
    ovp_size = (patch_size-block_size)//2
    if preprocess:
        img = FastEnhanceTexture(img, sigma=2.5, show=False)

    img = np.lib.pad(img, (ovp_size,ovp_size),'symmetric')
    h,w = img.shape
    blkH = (h - patch_size)//block_size+1
    blkW = (w - patch_size)//block_size+1
    local_info = np.empty((blkH,blkW),dtype = object)

    x, y = np.meshgrid(range(int(-patch_size / 2),int(patch_size / 2)), range(int(-patch_size / 2),int(patch_size / 2)))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    r = np.sqrt(x*x + y*y) + 0.0001
    RMIN = 3  # min allowable ridge spacing
    RMAX = 18 # maximum allowable ridge spacing
    FLOW = patch_size / RMAX
    FHIGH = patch_size / RMIN
    dRLow = 1. / (1 + (r / FHIGH) ** 4)
    dRHigh = 1. / (1 + (FLOW / r) ** 4)
    dBPass = dRLow * dRHigh  # bandpass

    dir = np.arctan2(y,x)
    dir[dir<0] = dir[dir<0] + math.pi
    dir_ind = np.floor(dir/(math.pi/nrof_dirs))
    dir_ind = dir_ind.astype(np.int,copy=False)
    dir_ind[dir_ind==nrof_dirs] = 0
    dir_ind_list = []
    for i in range(nrof_dirs):
        tmp = np.argwhere(dir_ind == i)
        dir_ind_list.append(tmp)
    sigma = patch_size/3
    weight = np.exp(-(x*x + y*y)/(sigma*sigma))
    for i in range(0,blkH):
        for j in range(0,blkW):
            patch =img[i*block_size:i*block_size+patch_size,j*block_size:j*block_size+patch_size].copy()
            local_info[i,j] = local_STFT(patch,weight,dBPass)
            local_info[i, j].analysis(r,dir_ind_list)
    # get the ridge flow from the local information
    dir_map,fre_map = get_ridge_flow_top(local_info)
    dir_map = smooth_dir_map(dir_map)

    return dir_map, fre_map



def label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5):
    mnt_s_out = np.squeeze(mnt_s_out)
    mnt_w_out = np.squeeze(mnt_w_out)
    mnt_h_out = np.squeeze(mnt_h_out)
    mnt_o_out = np.squeeze(mnt_o_out)
    assert len(mnt_s_out.shape)==2 and len(mnt_w_out.shape)==3 and len(mnt_h_out.shape)==3 and len(mnt_o_out.shape)==3

    # get cls results
    mnt_sparse = sparse.coo_matrix(mnt_s_out>thresh)
    mnt_list = np.array(list(zip(mnt_sparse.row, mnt_sparse.col)), dtype=np.int32)
    if mnt_list.shape[0] == 0:
        return np.zeros((0, 4))

    # get regression results
    mnt_w_out = np.argmax(mnt_w_out, axis=-1)
    mnt_h_out = np.argmax(mnt_h_out, axis=-1)
    mnt_o_out = np.argmax(mnt_o_out, axis=-1) # TODO: use ori_highest_peak(np version)

    # get final mnt
    mnt_final = np.zeros((len(mnt_list), 4))
    mnt_final[:, 0] = mnt_sparse.col*8 + mnt_w_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 1] = mnt_sparse.row*8 + mnt_h_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 2] = (mnt_o_out[mnt_list[:,0], mnt_list[:,1]]*2-89.)/180*np.pi
    mnt_final[mnt_final[:, 2]<0.0, 2] = mnt_final[mnt_final[:, 2]<0.0, 2]+2*np.pi
    # New one
    mnt_final[:, 2] = (-mnt_final[:, 2]) % (2*np.pi)
    mnt_final[:, 3] = mnt_s_out[mnt_list[:,0], mnt_list[:, 1]]

    return mnt_final

def py_cpu_nms(det, thresh):
    if det.shape[0]==0:
        return det
    dets = det.tolist()
    dets.sort(key=lambda x:x[3], reverse=True)
    dets = np.array(dets)


    box_sz = 25
    x1 = np.reshape(dets[:,0],[-1,1]) -box_sz
    y1 = np.reshape(dets[:,1],[-1,1]) -box_sz
    x2 = np.reshape(dets[:,0],[-1,1]) +box_sz
    y2 = np.reshape(dets[:,1],[-1,1]) +box_sz
    scores = dets[:, 2]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep, :]

def distance(y_true, y_pred, max_D=16, max_O=np.pi/6):
    D = spatial.distance.cdist(y_true[:, :2], y_pred[:, :2], 'euclidean')
    O = spatial.distance.cdist(np.reshape(y_true[:, 2], [-1, 1]), np.reshape(y_pred[:, 2], [-1, 1]), angle_delta)
    return (D<=max_D)*(O<=max_O)

def angle_delta(A, B, max_D=np.pi*2):
    delta = np.abs(A - B)
    delta = np.minimum(delta, max_D-delta)
    return delta 
def nms(mnt):
    if mnt.shape[0]==0:
        return mnt
    # sort score
    mnt_sort = mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    # cal distance
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=np.pi/6).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in range(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(np.bool), :]

def fuse_nms(mnt, mnt_set_2):
    if mnt.shape[0]==0:
        return mnt
    # sort score
    all_mnt = np.concatenate((mnt, mnt_set_2))

    mnt_sort = all_mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    # cal distance
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=2*np.pi).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in range(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(np.bool), :]

def show_orientation_field(img,dir_map,mask=None,fname=None):
    h,w = img.shape[:2]

    if mask is None:
        mask = np.ones((h,w),dtype=np.uint8)
    blkH, blkW = dir_map.shape

    blk_size = h/blkH

    R = blk_size/2*0.8
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    for i in range(blkH):
        y0 = i*blk_size + blk_size/2
        y0 = int(y0)
        for j in range(blkW):
            x0 = j*blk_size + blk_size/2
            x0 = int(x0)
            ori = dir_map[i,j]
            if mask[y0,x0] == 0:
                continue
            if ori<-9:
                continue
            x1 = x0 - R * math.cos(ori)
            x2 = x0 + R * math.cos(ori)
            y1 = y0 - R * math.sin(ori)
            y2 = y0 + R * math.sin(ori)
            plt.plot([x1, x2], [y1, y2], 'b-', lw=2)
    plt.axis('off')
    if fname is not None:
        fig.savefig(fname,dpi = 500, bbox_inches='tight', pad_inches = 0)
        plt.close()
    else:
        plt.show(block=True)
def fuse_minu_orientation(dir_map, mnt, mode=1,block_size=16):
    # mode is the way to fuse output minutiae with orientation
    # 1: use orientation; 2: use minutiae; 3: fuse average
    blkH, blkW = dir_map.shape
    dir_map = dir_map%(2*np.pi)

    if mode == 1:
        for k in range(mnt.shape[0]):
            # Choose nearest orientation
            ori_value = dir_map[int(mnt[k, 1]//block_size),int(mnt[k, 0]//block_size)]
            if 0 < mnt[k, 2] and mnt[k, 2] <= np.pi/2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    mnt[k, 2] = ori_value
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (ori_value - mnt[k, 2]) < (np.pi - ori_value + mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    mnt[k, 2] = ori_value - np.pi
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    if (np.pi*2 - ori_value + mnt[k, 2]) < (ori_value - np.pi - mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
            if np.pi/2 < mnt[k, 2] and mnt[k, 2] <= np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi - ori_value + mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    mnt[k, 2] = ori_value
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    mnt[k, 2] = ori_value - np.pi
            if np.pi < mnt[k, 2] and mnt[k, 2] <= 3*np.pi/2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (mnt[k, 2] - ori_value) < (ori_value + np.pi - mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    mnt[k, 2] = ori_value
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
            if 3*np.pi/2 < mnt[k, 2] and mnt[k, 2] <= 2*np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (np.pi - mnt[k, 2] + ori_value) < (mnt[k, 2] - np.pi - ori_value):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    if (mnt[k, 2] - ori_value) < (np.pi*2 - mnt[k, 2] + ori_value - np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    mnt[k, 2] = ori_value


    elif mode == 2:
        return
    elif mode ==3:
        for k in range(mnt.shape[0]):
            # Choose nearest orientation

            ori_value = dir_map[int(mnt[k, 1] // block_size), int(mnt[k, 0] // block_size)]
            if 0 < mnt[k, 2] and mnt[k, 2] <= np.pi / 2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    fixed_ori = ori_value
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (ori_value - mnt[k, 2]) < (np.pi - ori_value + mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (np.pi * 2 - ori_value + mnt[k, 2]) < (ori_value - np.pi - mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
            if np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi - ori_value + mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    fixed_ori = ori_value
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    fixed_ori = ori_value - np.pi
            if np.pi < mnt[k, 2] and mnt[k, 2] <= 3 * np.pi / 2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (mnt[k, 2] - ori_value) < (ori_value + np.pi - mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    fixed_ori = ori_value
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
            if 3 * np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= 2 * np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (np.pi - mnt[k, 2] + ori_value) < (mnt[k, 2] - np.pi - ori_value):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi * 2 - mnt[k, 2] + ori_value - np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    fixed_ori = ori_value

            mnt[k, 2] = (mnt[k, 2] + fixed_ori)/2.0
    else:
        return


def gausslabel(length=180, stride=2):
    gaussian_pdf = signal.gaussian(length+1, 3)
    label = np.reshape(np.arange(stride/2, length, stride), [1,1,-1,1])
    y = np.reshape(np.arange(stride/2, length, stride), [1,1,1,-1])
    delta = np.array(np.abs(label - y), dtype=int)
    delta = np.minimum(delta, length-delta)+int(length/2)
    return gaussian_pdf[delta]
def select_max(x):
    x = x / (K.max(x, axis=-1, keepdims=True)+K.epsilon())
    x = K.tf.where(K.tf.greater(x, 0.999), x, K.tf.zeros_like(x)) # select the biggest one
    x = x / (K.sum(x, axis=-1, keepdims=True)+K.epsilon()) # prevent two or more ori is selected
    return x
def merge_mul(x):
    return reduce(lambda x,y:x*y, x)
def merge_sum(x):
    return reduce(lambda x,y:x+y, x)
def reduce_sum(x):
    return K.sum(x,axis=-1,keepdims=True)
kernal2angle = np.reshape(np.arange(1, 180, 2, dtype=float), [1,1,1,90])/90.*np.pi #2angle = angle*2
sin2angle, cos2angle = np.sin(kernal2angle), np.cos(kernal2angle)
def ori2angle(ori):
    sin2angle_ori = K.sum(ori*sin2angle, -1, keepdims=True)
    cos2angle_ori = K.sum(ori*cos2angle, -1, keepdims=True)
    modulus_ori = K.sqrt(K.square(sin2angle_ori)+K.square(cos2angle_ori))
    return sin2angle_ori, cos2angle_ori, modulus_ori

# Group with depth
def merge_concat(x):
    return K.tf.concat(x,3)

def ori_highest_peak(y_pred, length=180):
    glabel = gausslabel(length=length,stride=2).astype(np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    ori_gau = K.conv2d(y_pred,glabel,padding='same')
    return ori_gau

def img_normalization(img_input, m0=0.0, var0=1.0):
    m = K.mean(img_input, axis=[1,2,3], keepdims=True)
    var = K.var(img_input, axis=[1,2,3], keepdims=True)
    after = K.sqrt(var0*K.tf.square(img_input-m)/var)
    image_n = K.tf.where(K.tf.greater(img_input, m), m0+after, m0-after)
    return image_n





def gabor_bank(stride=2,Lambda=8):

    filters_cos = np.ones([25,25,int(180/stride)], dtype=float)
    filters_sin = np.ones([25,25,int(180/stride)], dtype=float)

    for n, i in enumerate(range(-90,90,stride)):
        theta = i*np.pi/180.
        kernel_cos, kernel_sin = gabor_fn((24,24),4.5, -theta, Lambda, 0, 0.5)
        filters_cos[..., n] = kernel_cos
        filters_sin[..., n] = kernel_sin

    filters_cos = np.reshape(filters_cos,[25,25,1,-1])
    filters_sin = np.reshape(filters_sin,[25,25,1,-1])
    return filters_cos, filters_sin

def atan2(y_x):
    y, x = y_x[0], y_x[1]+K.epsilon()
    atan = K.tf.atan(y/x)
    angle = K.tf.where(K.tf.greater(x,0.0), atan, K.tf.zeros_like(x))
    angle = K.tf.where(K.tf.logical_and(K.tf.less(x,0.0),  K.tf.greater_equal(y,0.0)), atan+np.pi, angle)
    angle = K.tf.where(K.tf.logical_and(K.tf.less(x,0.0),  K.tf.less(y,0.0)), atan-np.pi, angle)
    return angle
def get_document_id(file_name): 
    return file_name.split('.')[0]
#def tocsv(mnt_path=None):
#    s=mnt_path+'/csv_results/'
#    
#    mkdir(s)
#    print(os.listdir(mnt_path))
 #      if file.endswith('.mnt'):
  #          print(file)
   #         print("found")
    #        data = np.loadtxt(file, skiprows=2)
     ####    f_name=s+f_name
         #   np.savetxt(f_name, data, delimiter = ',')
   # return

def work_CNAI(mnt_path=None):
    

    l = CNAI.CNAI()
    index_table = dict()
    data = dict()
    path = mnt_path #+'/csv_results/'

    file_list = os.listdir(path)
    os.chdir(path)

    for file in file_list:
        print(file)
        if file.endswith('.mnt'):
            doc_id = get_document_id(file)
            l.points(file)
            point_id = 0
            for pt in l.pts:
                point_id = point_id + 1
                m_neighbours = l.m_neighbours(l.sort_neighbours(pt,l.n_neighbours(pt)))
                for m in m_neighbours:
                    desc = l.discreet_affine_vector(l.compute_descriptor(l.f_neighbours(m)))
                    index = l.compute_index(desc)
                    data['doc_id'] = doc_id
                    data['point_id'] = point_id
                    data['desc'] = desc
                    index_table[index] = data
                    data = dict()

    root1=os.getcwd()               
    with open('test.json', 'w') as fp:
        json.dump(index_table, fp, sort_keys=True, indent=4)
    os.chdir(root)
    model_path=cnai_model_dir
    num_classes=10
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(15,1)))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(Conv1D(256, kernel_size=3, activation='relu'))
    model.add(Conv1D(256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.load_weights(model_path)
    os.chdir(root1)
    with open('test.json', 'r') as test1:
        test_data = json.load(test1)

    x_try = []
    for i in test_data.keys():
        x_try.append(test_data[i]['desc'])
    
    num_test_samples = len(x_try)
    time_len=15
    num_channels=1
    x_try = np.array(x_try)
    print("input shape")
    print(x_try.shape)


    x_try = x_try.reshape(num_test_samples,time_len, num_channels)

    p = model.predict(x_try)
    p = np.argmax(p, axis = 1) 
    
    a=collections.Counter(p)
    print(a)
    maxa=0
    for i in range(9):
        if maxa<=a[i]:
            maxa=a[i]
            ind=i
    print("\nPredicted Fingerprint belong to: ")
    ind=ind+101
    print(ind)
   # print("With probability of:")
   # prob=(a[ind-101])/(len(x_try))
    #print(prob)
    return

def gabor_fn(ksize, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    # Bounding box
    nstds = 3
    xmax = ksize[0]/2
    ymax = ksize[1]/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb_cos = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    gb_sin = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(2 * np.pi / Lambda * x_theta + psi)
    return gb_cos, gb_sin

class local_STFT:
    def __init__(self,patch,weight = None, dBPass = None):


        if weight is not None:
            patch = patch * weight
        patch = patch - np.mean(patch)
        norm = np.linalg.norm(patch)
        patch = patch / (norm+0.000001)

        f = np.fft.fft2(patch)
        fshift = np.fft.fftshift(f)
        if dBPass is not None:
            fshift = dBPass * fshift

        self.patch_FFT = fshift
        self.patch = patch
        self.ori = None
        self.fre = None
        self.confidence = None
        self.patch_size = patch.shape[0]

    def analysis(self,r,dir_ind_list=None,N=2):

        assert(dir_ind_list is not None)
        energy = np.abs(self.patch_FFT)
        energy = energy / (np.sum(energy)+0.00001)
        nrof_dirs = len(dir_ind_list)

        ori_interval = math.pi/nrof_dirs
        ori_interval2 = ori_interval/2


        pad_size = 1
        dir_norm = np.zeros((nrof_dirs + 2,))
        for i in range(nrof_dirs):
            tmp = energy[dir_ind_list[i][:, 0], dir_ind_list[i][:, 1]]
            dir_norm[i + 1] = np.sum(tmp)

        dir_norm[0] = dir_norm[nrof_dirs]
        dir_norm[nrof_dirs + 1] = dir_norm[1]

        # smooth dir_norm
        smoothed_dir_norm = dir_norm
        for i in range(1, nrof_dirs + 1):
            smoothed_dir_norm[i] = (dir_norm[i - 1] + dir_norm[i] * 4 + dir_norm[i + 1]) / 6

        smoothed_dir_norm[0] = smoothed_dir_norm[nrof_dirs]
        smoothed_dir_norm[nrof_dirs + 1] = smoothed_dir_norm[1]

        den = np.sum(smoothed_dir_norm[1:nrof_dirs + 1]) + 0.00001  # verify if den == 1
        smoothed_dir_norm = smoothed_dir_norm/den  # normalization if den == 1, this line can be removed

        ori = []
        fre = []
        confidence = []

        wenergy = energy*r
        for i in range(1, nrof_dirs+1):
            if smoothed_dir_norm[i] > smoothed_dir_norm[i-1] and smoothed_dir_norm[i] > smoothed_dir_norm[i+1]:
                tmp_ori = (i-pad_size)*ori_interval + ori_interval2 + math.pi/2
                ori.append(tmp_ori)
                confidence.append(smoothed_dir_norm[i])
                tmp_fre = np.sum(wenergy[dir_ind_list[i-pad_size][:, 0], dir_ind_list[i-pad_size][:, 1]])/dir_norm[i]
                tmp_fre = 1/(tmp_fre+0.00001)
                fre.append(tmp_fre)


        if len(confidence)>0:
            confidence = np.asarray(confidence)
            fre = np.asarray(fre)
            ori = np.asarray(ori)
            ind = confidence.argsort()[::-1]
            confidence = confidence[ind]
            fre = fre[ind]
            ori = ori[ind]
            if len(confidence) >= 2 and confidence[0]/confidence[1]>2.0:

                self.ori = [ori[0]]
                self.fre = [fre[0]]
                self.confidence = [confidence[0]]
            elif len(confidence)>N:
                fre = fre[:N]
                ori = ori[:N]
                confidence = confidence[:N]
                self.ori = ori
                self.fre = fre
                self.confidence = confidence
            else:
                self.ori = ori
                self.fre = fre
                self.confidence = confidence

    def get_features_of_topN(self,N=2):
        if self.confidence is None:
            self.border_wave = None
            return
        candi_num = len(self.ori)
        candi_num = np.min([candi_num,N])
        patch_size = self.patch_FFT.shape
        for i in range(candi_num):

            kernel = gabor_kernel(self.fre[i], theta=self.ori[i], sigma_x=10, sigma_y=10)

            kernel_f = np.fft.fft2(kernel.real, patch_size)
            kernel_f = np.fft.fftshift(kernel_f)
            patch_f = self.patch_FFT * kernel_f

            patch_f = np.fft.ifftshift(patch_f)  # *np.sqrt(np.abs(fshift)))
            rec_patch = np.real(np.fft.ifft2(patch_f))


            plt.subplot(121), plt.imshow(self.patch, cmap='gray')
            plt.title('Input patch'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(rec_patch, cmap='gray')
            plt.title('filtered patch'), plt.xticks([]), plt.yticks([])
            plt.show()

    def reconstruction(self,weight=None):
        f_ifft = np.fft.ifftshift(self.patch_FFT)  # *np.sqrt(np.abs(fshift)))
        rec_patch = np.real(np.fft.ifft2(f_ifft))
        if weight is not None:
            rec_patch = rec_patch * weight
        return rec_patch

    def gabor_filtering(self,theta,fre,weight=None):

        patch_size = self.patch_FFT.shape
        kernel = gabor_kernel(fre, theta=theta,sigma_x=4,sigma_y=4)

        f = kernel.real
        f = f - np.mean(f)
        f = f / (np.linalg.norm(f)+0.0001)


        kernel_f = np.fft.fft2(f,patch_size)
        kernel_f = np.fft.fftshift(kernel_f)
        patch_f = self.patch_FFT*kernel_f

        patch_f = np.fft.ifftshift(patch_f)  # *np.sqrt(np.abs(fshift)))
        rec_patch = np.real(np.fft.ifft2(patch_f))
        if weight is not None:
            rec_patch = rec_patch * weight
        return rec_patch

if __name__ =='__main__':
    main()
    a=output_dir+"/CoarseNet_test/mnt_results"
    work_CNAI(mnt_path=a)


