import numpy as np
import csv
import cv2
import math
import os

gray_level = 32


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def demo(lst, k):
    return lst[k:] + lst[:k]


def RotLBP(image, mask):
    W, H = image.shape
    xx = [-1, 0, 1, 1, 1, 0, -1, -1]
    yy = [-1, -1, -1, 0, 1, 1, 1, 0]
    res = np.zeros((W, H), dtype="uint8")
    for i in range(W):
        for j in range(H):
            if mask[i][j] == 255:
                temp = ""
                for m in range(8):
                    Xtemp = xx[m] + i
                    Ytemp = yy[m] + j
                    if image[Xtemp, Ytemp] > image[i, j]:
                        temp = temp + '1'
                    else:
                        temp = temp + '0'
                res[i - 1][j - 1] = int(temp, 2)
                for m in range(8):
                    temp = demo(temp, 1)
                    if res[i - 1][j - 1] > int(temp, 2):
                        res[i - 1][j - 1] = int(temp, 2)
    return res


def LBP(image, mask):
    W, H = image.shape
    xx = [-1, 0, 1, 1, 1, 0, -1, -1]
    yy = [-1, -1, -1, 0, 1, 1, 1, 0]
    res = np.zeros((W, H), dtype="uint8")
    for i in range(W):
        for j in range(H):
            if mask[i][j] == 255:
                temp = ""
                for m in range(8):
                    Xtemp = xx[m] + i
                    Ytemp = yy[m] + j
                    if image[Xtemp, Ytemp] > image[i, j]:
                        temp = temp + '1'
                    else:
                        temp = temp + '0'
                    res[i - 1][j - 1] = int(temp, 2)
    return res


def maxGrayLevel(img, mask):
    max_gray_level = 0
    (height, width) = img.shape
    # print(height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level and mask[y][x] == 255:
                max_gray_level = img[y][x]
    # print(max_gray_level)
    return max_gray_level + 1


def getGlcm(input, mask, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape
    max_gray_level = maxGrayLevel(input, mask)
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                if mask[j][i] == 255:
                    srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level
    for j in range(height):
        for i in range(width):
            if mask[j][i] == 255:
                rows = (int)((srcdata[j][i]))
                if j + d_y in range(height) and i + d_x in range(width) and mask[j + d_y][i + d_x] == 255:
                    cols = (int)((srcdata[j + d_y][i + d_x]))
                else:
                    continue
                ret[rows][cols] += 1.0
                ret[cols][rows] += 1.0

    # for i in range(gray_level):
    #     for j in range(gray_level):
    #        ret[i][j] /= float(height * width)
    # print(ret)
    return ret


def feature_computer(p, gray_level):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    Cor = 0.0
    Pro = 0.0
    Cont = 0.0
    Gen = 0.0
    Var = 0.0
    Mea = 0.0
    Sha = 0.0
    total = 0.0
    # print(p[gray_level-1][gray_level-1])
    for i in range(gray_level):
        for j in range(gray_level):
            total += p[i][j]
            # print(total)
    for i in range(gray_level):
        for j in range(gray_level):
            if total != 0:
                p[i][j] = p[i][j] / total
    # print(p[gray_level-1][gray_level-1])
    for i in range(gray_level):
        for j in range(gray_level):
            n = float(abs(i - j) * abs(i - j))
            Gen += float(p[i][j]) / (1 + abs(i - j))
            Cont += float(p[i][j]) * n
            Mea += (i + j) * p[i][j]
            Con += float((i - j) * (i - j)) * p[i][j]  # 对比度
            Asm += p[i][j] * p[i][j]  # 能量
            if i != j:
                Idm += float(p[i][j]) / float((i - j) * (i - j))  # 一致性
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])  # 熵
    Sigx = 0.0
    Sigy = 0.0
    ux = 0.0
    uy = 0.0
    for j in range(gray_level):
        for i in range(gray_level):
            ux += i * p[j][i]
            uy += j * p[j][i]
    for j in range(gray_level):
        for i in range(gray_level):
            Sigx += (i - ux) * (i - ux) * p[j][i]
            Sigy += (j - uy) * (j - uy) * p[j][i]
    for i in range(gray_level):
        for j in range(gray_level):
            Cor += float(i - ux) * float(j - uy) * p[i][j]
    if Sigx != 0 and Sigy != 0:
        Cor /= Sigx
        Cor /= Sigy
    for i in range(gray_level):
        for j in range(gray_level):
            Pro += (i + j - ux - uy) * (i + j - ux - uy) * (i + j - ux - uy) * (i + j - ux - uy) * p[i][j]
            Var += (i - ux) * (i - ux) * p[i][j] + (j - uy) * (j - uy) * p[i][j]
            Sha += (i + j - ux - uy) * (i + j - ux - uy) * (i + j - ux - uy) * p[i][j]
    Sha = math.fabs(Sha)
    # print(Asm)
    return Asm, Con, -Eng, Idm, Cor, Pro, Cont, Gen, Var, Mea, Sha



def parseScript():
    #name_arr = ['lh_rrt', 'ljw_flt', 'ljw_rlt', 'ljw_frm', 'ljw_rrm', 'ljw_frt', 'ljw_rrt', 'qyh_fli', 'qyh_rli',
    #            'qyh_fri', 'qyh_rri', 'syj_fli', 'syj_rli', 'syj_flt', 'syj_rlt', 'syj_frt', 'syj_rrt', 'th_frt',
    #            'th_rrt', 'wc_fli', 'wc_rli', 'wc_fri', 'wc_rri', 'wc_frm', 'wc_rrm', 'xwy_flt', 'xwy_rlt', 'zlf_flm',
    #            'zlf_rlm', 'zlf_frm', 'zlf_rrm']
    # name = 'cyk_fri'
    name_arr = ['cyk_fri','cyk_frt','cyk_ri','cyk_rt','syj_fli','syj_flt','syj_frt','syj_li','syj_lt','syj_rt']
    inputPathh = ['/home/asrathor/Downloads/robustimageoutput/bottom/','/home/asrathor/Downloads/robustimageoutput/cold/','/home/asrathor/Downloads/robustimageoutput/dark/','/home/asrathor/Downloads/robustimageoutput/gentlepressure/','/home/asrathor/Downloads/robustimageoutput/hardpressure/','/home/asrathor/Downloads/robustimageoutput/left/','/home/asrathor/Downloads/robustimageoutput/light/','/home/asrathor/Downloads/robustimageoutput/normal/','/home/asrathor/Downloads/robustimageoutput/right/','/home/asrathor/Downloads/robustimageoutput/top/','/home/asrathor/Downloads/robustimageoutput/warm/','/home/asrathor/Downloads/robustimageoutput/wet/','/home/asrathor/Downloads/robustimageoutput/wetlittle/']
    for inputPath in inputPathh:
        for name in name_arr:
            total_fingerprint_feature_arr = np.zeros((50, 44 * 3))
            total_glcm_feature_arr = np.zeros((50, 44))
            for number in range(1, 51):
    
                if not (os.path.isfile(inputPath + name + '_' + str(number) + '_orient.png')):
                    continue
                try:
                    print(name, number)
                    mask = cv2.imread(inputPath + name + '_' + str(number) + '_mask.png')
                    height = int(mask.shape[0] * 0.2)
                    width = int(mask.shape[1] * 0.2)
                    mask = cv2.resize(mask, (width, height), cv2.INTER_CUBIC)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
                    img_orient = cv2.imread(inputPath + name + '_' + str(number) + '_orient.png', 0)
                    img_ridge = cv2.imread(inputPath + name + '_' + str(number) + '_ridge.png', 0)
                    img_minutiae = cv2.imread(inputPath + name + '_' + str(number) + '_minutiae.png', 0)
    
                    img_orient = cv2.resize(img_orient, (width, height), cv2.INTER_CUBIC)
                    try:
                        img_orient = RotLBP(img_orient, mask)
                    except Exception:
                        pass
                    glcm_0 = getGlcm(img_orient, mask, 1, 0)
                    glcm_1 = getGlcm(img_orient, mask, 0, 1)
                    glcm_2 = getGlcm(img_orient, mask, 1, 1)
                    glcm_3 = getGlcm(img_orient, mask, -1, 1)
                    asm0, con0, eng0, idm0, cor0, pro0, cont0, gen0, var0, mea0, sha0 = feature_computer(glcm_0, 32)
                    asm1, con1, eng1, idm1, cor1, pro1, cont1, gen1, var1, mea1, sha1 = feature_computer(glcm_1, 32)
                    asm2, con2, eng2, idm2, cor2, pro2, cont2, gen2, var2, mea2, sha2 = feature_computer(glcm_2, 32)
                    asm3, con3, eng3, idm3, cor3, pro3, cont3, gen3, var3, mea3, sha3 = feature_computer(glcm_3, 32)
                    feature_arr_orient = [asm0, con0, eng0, idm0, cor0, pro0, cont0, gen0, var0, mea0, sha0, asm1, con1,
                                          eng1, idm1,
                                          cor1, pro1, cont1, gen1, var1, mea1, sha1, asm2, con2, eng2, idm2, cor2, pro2,
                                          cont2, gen2,
                                          var2, mea2, sha2, asm3, con3, eng3, idm3, cor3, pro3, cont3, gen3, var3, mea3,
                                          sha3]
    
                    img_ridge = cv2.resize(img_ridge, (width, height), cv2.INTER_CUBIC)
                    try:
                        img_ridge = RotLBP(img_ridge, mask)
                    except Exception:
                        pass
                    glcm_0 = getGlcm(img_ridge, mask, 1, 0)
                    glcm_1 = getGlcm(img_ridge, mask, 0, 1)
                    glcm_2 = getGlcm(img_ridge, mask, 1, 1)
                    glcm_3 = getGlcm(img_ridge, mask, -1, 1)
                    asm0, con0, eng0, idm0, cor0, pro0, cont0, gen0, var0, mea0, sha0 = feature_computer(glcm_0, 32)
                    asm1, con1, eng1, idm1, cor1, pro1, cont1, gen1, var1, mea1, sha1 = feature_computer(glcm_1, 32)
                    asm2, con2, eng2, idm2, cor2, pro2, cont2, gen2, var2, mea2, sha2 = feature_computer(glcm_2, 32)
                    asm3, con3, eng3, idm3, cor3, pro3, cont3, gen3, var3, mea3, sha3 = feature_computer(glcm_3, 32)
                    feature_arr_ridge = [asm0, con0, eng0, idm0, cor0, pro0, cont0, gen0, var0, mea0, sha0, asm1, con1,
                                         eng1,
                                         idm1,
                                         cor1, pro1, cont1, gen1, var1, mea1, sha1, asm2, con2, eng2, idm2, cor2, pro2,
                                         cont2,
                                         gen2,
                                         var2, mea2, sha2, asm3, con3, eng3, idm3, cor3, pro3, cont3, gen3, var3, mea3,
                                         sha3]
    
                    img_minutiae = cv2.resize(img_minutiae, (width, height), cv2.INTER_CUBIC)
                    try:
                        img_minutiae = RotLBP(img_minutiae, mask)
                    except Exception:
                        pass
                    glcm_0 = getGlcm(img_minutiae, mask, 1, 0)
                    glcm_1 = getGlcm(img_minutiae, mask, 0, 1)
                    glcm_2 = getGlcm(img_minutiae, mask, 1, 1)
                    glcm_3 = getGlcm(img_minutiae, mask, -1, 1)
                    asm0, con0, eng0, idm0, cor0, pro0, cont0, gen0, var0, mea0, sha0 = feature_computer(glcm_0, 32)
                    asm1, con1, eng1, idm1, cor1, pro1, cont1, gen1, var1, mea1, sha1 = feature_computer(glcm_1, 32)
                    asm2, con2, eng2, idm2, cor2, pro2, cont2, gen2, var2, mea2, sha2 = feature_computer(glcm_2, 32)
                    asm3, con3, eng3, idm3, cor3, pro3, cont3, gen3, var3, mea3, sha3 = feature_computer(glcm_3, 32)
                    feature_arr_minutiae = [asm0, con0, eng0, idm0, cor0, pro0, cont0, gen0, var0, mea0, sha0, asm1, con1,
                                            eng1,
                                            idm1,
                                            cor1, pro1, cont1, gen1, var1, mea1, sha1, asm2, con2, eng2, idm2, cor2, pro2,
                                            cont2,
                                            gen2,
                                            var2, mea2, sha2, asm3, con3, eng3, idm3, cor3, pro3, cont3, gen3, var3, mea3,
                                            sha3]
    
                    # asm0, con0, eng0, idm0, cor0, pro0, cont0, gen0, var0, mea0, sha0 = feature_computer(img_orient,16)
                    # print(0)
                    # asm1, con1, eng1, idm1, cor1, pro1, cont1, gen1, var1, mea1, sha1 = feature_computer(img_ridge,128)
                    # print(1)
                    # asm2, con2, eng2, idm2, cor2, pro2, cont2, gen2, var2, mea2, sha2 = feature_computer(img_minutiae,128)
                    # print(2)
                    # img_orient = cv2.resize(img_orient,(32,32),cv2.INTER_CUBIC)
                    # img_ridge = cv2.resize(img_ridge, (32, 32), cv2.INTER_CUBIC)
                    # img_minutiae = cv2.resize(img_minutiae, (32, 32), cv2.INTER_CUBIC)
    
                    feature_arr = feature_arr_orient + feature_arr_ridge + feature_arr_minutiae
    
                    total_fingerprint_feature_arr[number - 1, :] = feature_arr
    
                    img_org = cv2.imread(inputPath + name + '_' + str(number) + '.png', 0)
                    img_org = cv2.resize(img_org, (width, height), cv2.INTER_CUBIC)
    
                    ret, threshold = cv2.threshold(img_org, 240, 255, 0)
                    mask = cv2.bitwise_not(threshold)
                    try:
                        img_gray = RotLBP(img_org, mask)
                    except Exception:
                        pass
                    glcm_0 = getGlcm(img_gray, mask, 1, 0)
                    glcm_1 = getGlcm(img_gray, mask, 0, 1)
                    glcm_2 = getGlcm(img_gray, mask, 1, 1)
                    glcm_3 = getGlcm(img_gray, mask, -1, 1)
                    asm0, con0, eng0, idm0, cor0, pro0, cont0, gen0, var0, mea0, sha0 = feature_computer(glcm_0, 32)
                    asm1, con1, eng1, idm1, cor1, pro1, cont1, gen1, var1, mea1, sha1 = feature_computer(glcm_1, 32)
                    asm2, con2, eng2, idm2, cor2, pro2, cont2, gen2, var2, mea2, sha2 = feature_computer(glcm_2, 32)
                    asm3, con3, eng3, idm3, cor3, pro3, cont3, gen3, var3, mea3, sha3 = feature_computer(glcm_3, 32)
                    feature_arr_2 = [asm0, con0, eng0, idm0, cor0, pro0, cont0, gen0, var0, mea0, sha0, asm1, con1, eng1,
                                     idm1, cor1, pro1, cont1, gen1, var1, mea1, sha1, asm2, con2, eng2, idm2, cor2, pro2,
                                     cont2, gen2, var2, mea2, sha2, asm3, con3, eng3, idm3, cor3, pro3, cont3, gen3, var3,
                                     mea3, sha3]
                    total_glcm_feature_arr[number - 1, :] = feature_arr_2
    
                    with open(inputPath + name + '_fingerprintfeatures.csv', 'w', newline='') as f:
    
                        writer = csv.writer(f)
                        writer.writerows(total_fingerprint_feature_arr)
    
                    with open(inputPath + name + '_glcmfeatures.csv', 'w', newline='') as f:
    
                        writer = csv.writer(f)
                        writer.writerows(total_glcm_feature_arr)
    
    
                except Exception:
                    pass
    
            print(name + ' Done')
        print(inputPath + 'Done')

parseScript()
