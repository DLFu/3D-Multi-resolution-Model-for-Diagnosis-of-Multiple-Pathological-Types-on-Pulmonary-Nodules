# coding=utf-8
#该程序是将实验室数据转化为NPY文件的程序，并且里面含有注释掉的归一化到[-1000,400]上的程序，但是这段程序只适用于LIDC数据
# from NoneGroup_GN_Dual_Path_3D_Networks import DPN92
import math
import os
import boto
import sys
from glob import glob
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np  # linear algebra
from tqdm import tqdm
import scipy
import copy
import xlrd
import string
import time
import xlwt
from collections import defaultdict
import re
import warnings
import argparse

# 归一化重建--uint16
def matrix2int16(matrix):
    '''
    matrix must be a numpy array NXN
    Returns uint16 version
    '''
    m_min = np.min(matrix)
    m_max = np.max(matrix)
    #     print(m_min, m_max)
    matrix = matrix - m_min  # 拔0
    return np.array(np.rint(matrix / float(m_max - m_min) * 65535.0), dtype=np.uint16)  # 归一 --> 归0 - 65535


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # 直接截断-1548以下像素


    image[image < -1548] = -1000   # 这个地方是把hu值进行归一化，归一化到-1000到400，超过400的任何东西都是我们不感兴趣的
    image[image > 400]=400
    image2 = np.zeros(image.shape, dtype=np.uint16)
    for n, im in enumerate(image):
        im = matrix2int16(im)
        image2[n, ...] = im
        # plt.imshow(im, cmap=plt.cm.gray)
        # plt.show()
        # plt.imshow(image[n], cmap=plt.cm.gray)
        # plt.show()
        mmax = np.max(image2)
        mmin = np.min(image2)
        # print(mmax)
        # print(mmin)

    return image
    plt.show(image)


def resample(image, scan, new_spacing=None, special=None):
    #         3D  , 各种信息 ,   新建间隔
    # Determine current pixel spacing确定当前像素间距
    spacing_flag = 0
    if new_spacing is None:
        new_spacing = [1, 1, 1]#重建层厚到1mm
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))

    spacing = np.array(list(spacing))
    if special == 84:
        spacing[0] = 1.25

    if spacing[0] > 2.5:
        spacing_flag = 1

    resize_factor = spacing / new_spacing
    #     print(resize_factor)
    new_real_shape = image.shape * resize_factor

    # 重采样后的数据格式（一般来说x,y不变，只变z）
    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    #     image = image[...,crop[0]:-crop[0], crop[1]:-crop[1]]
    return image, new_spacing, spacing_flag


class Dicom_ersatz:
    def __init__(self, spacing, pixel_array, direction):
        # 切片厚度（z 轴spacing）
        self.SliceThickness = spacing[2]
        # x,y 坐标（真实）
        self.pixel_array = pixel_array
        # x,y 的spacing，list 表
        self.PixelSpacing = list(spacing[:2])
        # direction 神秘代码（表示方向性，一般不用管）
        self.direction = direction


def load_scan(path, Excel_contain):

    GenName = os.path.basename(path)
    GenList = GenName.split('_')
    GenNum = GenList[-1]

    path+='\\'

    # print(path)
    reader = sitk.ImageSeriesReader()

    seriesID = reader.GetGDCMSeriesIDs(path)

    dicom_names = reader.GetGDCMSeriesFileNames(path,seriesID[0])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    itk_img = sitk.GetArrayFromImage(image)  # indexes are z,y,x (notice the ordering)

    spacing = np.array(image.GetSpacing())  # 世界坐标系下单像素间距：spacing of voxels in world coor. (mm)

    num_z, height, width = itk_img.shape

    direction = np.array(image.GetDirection())  # 世界坐标系下(x,y,z)

    origin = np.array(image.GetOrigin())  # 世界坐标系下图像各方向的起点(x,y,z)(mm)

    slices = []
    for iz, s in enumerate(itk_img):
        # 这里才是 3D 的合辑，其中包括 z 个（spacing,x,y,direction,mask)
        slices += [Dicom_ersatz(spacing, s, direction)]
    tumor_coords = []

    # 把 csv 文件中统计的坐标（x,y,z)和diam存入 nodes 中
    flag = 0
    nodes = []
    label=[]

    for index, content in enumerate(Excel_contain['住院ID号']):
        #cc=str(int(content))
        #cc = int(content)
        if GenNum ==str(content):# str(content):#将DICOM文件名与excel名比较，找出对应的那一对
            flag = 1
            # row 为4（x,y,z,diam)
            XYZ = Excel_contain["坐标"][index]
            XYZ = XYZ[1:]
            XYZ = XYZ[:-1]
            XYZ_list=XYZ.split(',')
            #print(XYZ_list[0])#仅显示x坐标
            node_x = float(XYZ_list[0])
            node_y = float(XYZ_list[1])
            node_z = float(XYZ_list[2])
            # 只需要X,Y,Z,diam 其他不要
            nodes += [(node_x, node_y, node_z, index)]

        else:
            if flag == 0:
                continue
            else:
                break


    for node in nodes:
        # node[0]:x  node[1]:y  node[2]:z  node[3]:diam
        node2 = direction[0] * node[0], direction[0] * node[1], node[2]

        center = (np.array([node2[0], node2[1], node2[2], node[3]])).astype('int16')

        center[0], center[1] = center[1], center[0]####python里面显示的肺结节跟ITKSNAP里面显示的xy坐标是反着的，所以这地方需要变换一下xy'坐标系

        tumor_coords += [(center)]

    return slices, tumor_coords, GenNum


def main():

    LIDC_path = ''#############这个地方还需要修改,这个地方的文件名不能是中文
    output_path30 = r''
    output_path20 = r''
    output_path45 = r''
    n_list = glob(LIDC_path + '*')#glob加*代表路径查找，*代表匹配0个或多个字符

    for i, ii in enumerate(n_list):
        base_l = os.path.basename(ii)
        base_list = base_l.split('_')  #[ID,44064]
        n_list[i] += '\\'
        # n_list[i] += base_l##dicom如果是两个文件夹嵌套的话需要加上这句话
        # n_list[i] += '\\'
    total_path = n_list
    ssum = 0

    xls_path = r''
    table1 = xlrd.open_workbook(xls_path)
    temp = table1.sheet_by_name('Sheet1')
    Excel_contain = {}
    for i in range(temp.ncols):
        temp1 = temp.col_values(i)
        Excel_contain[temp1[0]] = temp1[1:]

    Total_num = 1
    spacing_flag = 0
    for num, content in tqdm(enumerate(total_path)):#這地方是病例文件里的

        content=content[:-1]#content是路径
        print(content)
        first_patient, tumor_info, subname = load_scan(content, Excel_contain)####first_patient是什么

        print(tumor_info)###需要看看tumor_infor是什么,是[array([349, 362, 121,  24], dtype=int16)]，nodes的信息

        # 返回值为从 HU 值转化为uint16 后的 3D 图像（z,x,y）
        first_patient_pixels = get_pixels_hu(first_patient)

        # 获取mask（int8）

        # 输出 pix_resampled：重采样后图像（3D）     # spacing：重采样后间隔（z,x,y) 或（z,y,x)？？？？？？？？为什么重采样以后变成了这样？
        pix_resampled = first_patient_pixels

        for n_t, tumor in tqdm(enumerate(tumor_info)):#tumor坐标位置是（y,x,z的形式）
            #   编号 肺结节数据（中心，直径）
            tumor_coords = tumor#需要看看tumor_coords是什么，是坐标值+索引值
            offset45 = 22
            offset30 = 15
            offset20 = 10

            z_border_left30 = tumor_coords[2] - offset30
            z_border_right30 = tumor_coords[2] + offset30
            x_border_left30 = tumor_coords[0] - offset30
            x_border_right30 = tumor_coords[0] + offset30
            y_border_left30 = tumor_coords[1] - offset30
            y_border_right30 = tumor_coords[1] + offset30

            z_border_left20 = tumor_coords[2] - offset20
            z_border_right20 = tumor_coords[2] + offset20
            x_border_left20 = tumor_coords[0] - offset20
            x_border_right20 = tumor_coords[0] + offset20
            y_border_left20 = tumor_coords[1] - offset20
            y_border_right20 = tumor_coords[1] + offset20

            z_border_left45 = tumor_coords[2] - offset45
            z_border_right45 = tumor_coords[2] + offset45
            x_border_left45 = tumor_coords[0] - offset45
            x_border_right45 = tumor_coords[0] + offset45
            y_border_left45 = tumor_coords[1] - offset45
            y_border_right45 = tumor_coords[1] + offset45

            # 提取 30*30*30 块
            if z_border_left30 >= 0 and z_border_right30 <= pix_resampled.shape[
                0] and x_border_left30 >= 0 and x_border_right30 <= 512 and y_border_left30 >= 0 and y_border_right30 <= 512:
                tumor_cube30 = pix_resampled[tumor_coords[2] - offset30: tumor_coords[2] + offset30,
                               tumor_coords[0] - offset30: tumor_coords[0] + offset30,
                               tumor_coords[1] - offset30: tumor_coords[1] + offset30]

                if spacing_flag == 0:
                    #np.save(output_path30 + '{0:04d}.npy'.format(Total_num), tumor_cube30)#{0:04d}.npy的意思是按照四位数进行保存，不够的左边用0补齐
                    np.save(output_path30 + '{}.npy'.format(subname), tumor_cube30)#subname来自于表格中

                    mal_xls.append(subname)#这地方是自己写的，为了把住院号加入表格中，这地方感觉有问题

                    #mal_xls.append(1 if Excel_contain['肺结节良恶性'][tumor_coords[3]]=='恶' else 0)###########################？？？？？？？？？？？？？？？
                    if Excel_contain['病理类型分类'][tumor_coords[3]] == '炎症':
                        mal_xls.append(0)
                    # elif Excel_contain['病理类型分类'][tumor_coords[3]] == '错构瘤':
                    #     mal_xls.append(1)
                    elif Excel_contain['病理类型分类'][tumor_coords[3]] == '鳞癌':
                        mal_xls.append(1)
                    elif Excel_contain['病理类型分类'][tumor_coords[3]] == '腺癌':
                        mal_xls.append(2)
                    # elif Excel_contain['病理类型分类'][tumor_coords[3]] == '小细胞癌':
                    #     mal_xls.append(4)
                    else:
                        mal_xls.append(3)
                    #mal_xls.append(0 if Excel_contain['病理类型分类'][tumor_coords[3]] == '炎症' else 0)


                    # sitk_img = sitk.GetImageFromArray(tumor_cube30, isVector=False)
                    # sitk.WriteImage(sitk_img, os.path.join('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}.mhd'.format(Total_num)))
                    # if os.path.exists('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}.mhd'.format(Total_num)):
                    #    os.remove('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}.mhd'.format(Total_num))

                    #print('saving to:{0:04d}'.format(Total_num + 1), '--Extracted from {}'.format(subname))
                    print('saving to:{}'.format(subname), '--Extracted from {}'.format(subname))

            else:
                tumor_cube30 = pix_resampled[
                               max(0, tumor_coords[2] - offset30): min(tumor_coords[2] + offset30, 511) + 1,
                               max(0, tumor_coords[0] - offset30): min(tumor_coords[0] + offset30, 511) + 1,
                               max(0, tumor_coords[1] - offset30): min(tumor_coords[1] + offset30, 511) + 1]

                if spacing_flag == 0:
                    # #np.save(output_path30 + '{0:04d}--1.npy'.format(Total_num), tumor_cube30)
                    # np.save(output_path30 + '{}--1.npy'.format(subname), tumor_cube30)
                    # mal_xls.append(1 if Excel_contain['肺结节良恶性'][tumor_coords[3]]=='恶' else 0)###########################？？？？？？？？？？？？？？？
                    np.save(output_path30 + '{}.npy'.format(subname), tumor_cube30)  # subname来自于表格中
                    mal_xls.append(subname)  # 这地方是自己写的，为了把住院号加入表格中
                    # mal_xls.append(1 if Excel_contain['肺结节良恶性'][tumor_coords[3]]=='恶' else 0)###########################？？？？？？？？？？？？？？？
                    if Excel_contain['病理类型分类'][tumor_coords[3]] == '炎症':
                        mal_xls.append(0)
                    # elif Excel_contain['病理类型分类'][tumor_coords[3]] == '错构瘤':
                    #     mal_xls.append(1)
                    elif Excel_contain['病理类型分类'][tumor_coords[3]] == '鳞癌':
                        mal_xls.append(1)
                    elif Excel_contain['病理类型分类'][tumor_coords[3]] == '腺癌':
                        mal_xls.append(2)
                    # elif Excel_contain['病理类型分类'][tumor_coords[3]] == '小细胞癌':
                    #     mal_xls.append(4)
                    else:
                        mal_xls.append(3)
                    # # sitk_img = sitk.GetImageFromArray(tumor_cube30, isVector=False)
                    # # sitk.WriteImage(sitk_img, os.path.join('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}--1.mhd'.format(Total_num)))
                    # # if os.path.exists('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}--1.mhd'.format(Total_num)):
                    # #     os.remove('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}--1.mhd'.format(Total_num))


                    #print('saving to:{0:04d}--1'.format(Total_num + 1), '--Extracted from {}'.format(subname))
                    print('saving to:{}--1'.format(subname), '--Extracted from {}'.format(subname))

            # 提取 20*20*20 块
            if z_border_left20 >= 0 and z_border_right20 <= pix_resampled.shape[
                0] and x_border_left20 >= 0 and x_border_right20 <= 512 and y_border_left20 >= 0 and y_border_right20 <= 512:
                tumor_cube20 = pix_resampled[tumor_coords[2] - offset20: tumor_coords[2] + offset20,
                               tumor_coords[0] - offset20: tumor_coords[0] + offset20,
                               tumor_coords[1] - offset20: tumor_coords[1] + offset20]

                if spacing_flag == 0:
                    #np.save(output_path20 + '{0:04d}.npy'.format(Total_num), tumor_cube20)
                    np.save(output_path20 + '{}.npy'.format(subname), tumor_cube20)

                    # sitk_img = sitk.GetImageFromArray(tumor_cube30, isVector=False)
                    # sitk.WriteImage(sitk_img, os.path.join('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}.mhd'.format(Total_num)))
                    # if os.path.exists('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}.mhd'.format(Total_num)):
                    #    os.remove('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}.mhd'.format(Total_num))


                    #print('saving to:{0:04d}'.format(Total_num + 1), '--Extracted from {}'.format(subname))
                    print('saving to:{}'.format(subname), '--Extracted from {}'.format(subname))


            else:
                tumor_cube20 = pix_resampled[
                               max(0, tumor_coords[2] - offset20): min(tumor_coords[2] + offset20, 511) + 1,
                               max(0, tumor_coords[0] - offset20): min(tumor_coords[0] + offset20, 511) + 1,
                               max(0, tumor_coords[1] - offset20): min(tumor_coords[1] + offset20, 511) + 1]

                if spacing_flag == 0:
                    #np.save(output_path20 + '{0:04d}--1.npy'.format(Total_num), tumor_cube20)
                    np.save(output_path20 + '{}--1.npy'.format(subname), tumor_cube20)

                    # sitk_img = sitk.GetImageFromArray(tumor_cube30, isVector=False)
                    # sitk.WriteImage(sitk_img, os.path.join('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}--1.mhd'.format(Total_num)))
                    # if os.path.exists('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}--1.mhd'.format(Total_num)):
                    #     os.remove('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}--1.mhd'.format(Total_num))


                    #print('saving to:{0:04d}--1'.format(Total_num + 1), '--Extracted from {}'.format(subname))
                    print('saving to:{}--1'.format(subname), '--Extracted from {}'.format(subname))


            # 提取 45*45*45 块
            if z_border_left45 >= 0 and z_border_right45 <= pix_resampled.shape[
                0] - 1 and x_border_left45 >= 0 and x_border_right45 <= 511 and y_border_left45 >= 0 and y_border_right45 <= 511:
                tumor_cube45 = pix_resampled[tumor_coords[2] - offset45: tumor_coords[2] + offset45 + 1,
                               tumor_coords[0] - offset45: tumor_coords[0] + offset45 + 1,
                               tumor_coords[1] - offset45: tumor_coords[1] + offset45 + 1]

                if spacing_flag == 0:
                    #np.save(output_path45 + '{0:04d}.npy'.format(Total_num), tumor_cube45)
                    np.save(output_path45 + '{}.npy'.format(subname), tumor_cube45)

                    # sitk_img = sitk.GetImageFromArray(tumor_cube30, isVector=False)
                    # sitk.WriteImage(sitk_img, os.path.join('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}.mhd'.format(Total_num)))
                    # if os.path.exists('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}.mhd'.format(Total_num)):
                    #    os.remove('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}.mhd'.format(Total_num))


                    #print('saving to:{0:04d}'.format(Total_num + 1), '--Extracted from {}'.format(subname))
                    print('saving to:{}'.format(subname), '--Extracted from {}'.format(subname))


            else:
                tumor_cube45 = pix_resampled[
                               max(0, tumor_coords[2] - offset45): min(tumor_coords[2] + offset45, 511) + 1,
                               max(0, tumor_coords[0] - offset45): min(tumor_coords[0] + offset45, 511) + 1,
                               max(0, tumor_coords[1] - offset45): min(tumor_coords[1] + offset45, 511) + 1]

                if spacing_flag == 0:
                    #np.save(output_path45 + '{0:04d}--1.npy'.format(Total_num), tumor_cube45)#{0：04d}说明生成的文件名的数字至少有4位，不足的在左边由0补齐
                    np.save(output_path45 + '{}--1.npy'.format(subname), tumor_cube45)

                    # sitk_img = sitk.GetImageFromArray(tumor_cube30, isVector=False)
                    # sitk.WriteImage(sitk_img, os.path.join('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}--1.mhd'.format(Total_num)))
                    # if os.path.exists('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}--1.mhd'.format(Total_num)):
                    #     os.remove('D:\\LIDC30_Reconstruction_raws_npy\\{0:04d}--1.mhd'.format(Total_num))


                    #print('saving to:{0:04d}--1'.format(Total_num + 1), '--Extracted from {}'.format(subname))
                    print('saving to:{}--1'.format(subname), '--Extracted from {}'.format(subname))


            Total_num += 1


if __name__ == '__main__':
    mal_xls=[]
    main()

    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('Sheet1')
    worksheet.write(0, 0, '住院号')
    worksheet.write(0, 1, 'malignancy')
    #worksheet.write(0, 2, 'sum_nod')

    for num2, mm in enumerate(mal_xls):
        if mm == '':
            print('ERROR!,num {} has no malignancy!'.format(num2))
        else:
            worksheet.write(num2 + 1, 0, mm)

    workbook.save('')
