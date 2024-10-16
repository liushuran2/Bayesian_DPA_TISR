# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from PIL import Image
import os
from skimage import io, exposure, img_as_uint, img_as_float
import h5py

rec_header_dtd = \
    [
        ("nx", "i4"),  # Number of columns
        ("ny", "i4"),  # Number of rows
        ("nz", "i4"),  # Number of sections

        ("mode", "i4"),  # Types of pixels in the image. Values used by IMOD:
        #  0 = unsigned or signed bytes depending on flag in imodFlags
        #  1 = signed short integers (16 bits)
        #  2 = float (32 bits)
        #  3 = short * 2, (used for complex data)
        #  4 = float * 2, (used for complex data)
        #  6 = unsigned 16-bit integers (non-standard)
        # 16 = unsigned char * 3 (for rgb data, non-standard)

        ("nxstart", "i4"),  # Starting point of sub-image (not used in IMOD)
        ("nystart", "i4"),
        ("nzstart", "i4"),

        ("mx", "i4"),  # Grid size in X, Y and Z
        ("my", "i4"),
        ("mz", "i4"),

        ("xlen", "f4"),  # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        ("ylen", "f4"),
        ("zlen", "f4"),

        ("alpha", "f4"),  # Cell angles - ignored by IMOD
        ("beta", "f4"),
        ("gamma", "f4"),

        # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
        ("mapc", "i4"),  # map column  1=x,2=y,3=z.
        ("mapr", "i4"),  # map row     1=x,2=y,3=z.
        ("maps", "i4"),  # map section 1=x,2=y,3=z.

        # These need to be set for proper scaling of data
        ("amin", "f4"),  # Minimum pixel value
        ("amax", "f4"),  # Maximum pixel value
        ("amean", "f4"),  # Mean pixel value

        ("ispg", "i4"),  # space group number (ignored by IMOD)
        ("next", "i4"),  # number of bytes in extended header (called nsymbt in MRC standard)
        ("creatid", "i2"),  # used to be an ID number, is 0 as of IMOD 4.2.23
        ("extra_data", "V30"),  # (not used, first two bytes should be 0)

        # These two values specify the structure of data in the extended header; their meaning depend on whether the
        # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
        # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
        # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
        ("nint", "i2"),
        # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
        ("nreal", "i2"),  # Number of reals per section (Agard format) or bit
        # Number of reals per section (Agard format) or bit
        # flags for which types of short data (SerialEM format):
        # 1 = tilt angle * 100  (2 bytes)
        # 2 = piece coordinates for montage  (6 bytes)
        # 4 = Stage position * 25    (4 bytes)
        # 8 = Magnification / 100 (2 bytes)
        # 16 = Intensity * 25000  (2 bytes)
        # 32 = Exposure dose in e-/A2, a float in 4 bytes
        # 128, 512: Reserved for 4-byte items
        # 64, 256, 1024: Reserved for 2-byte items
        # If the number of bytes implied by these flags does
        # not add up to the value in nint, then nint and nreal
        # are interpreted as ints and reals per section

        ("extra_data2", "V20"),  # extra data (not used)
        ("imodStamp", "i4"),  # 1146047817 indicates that file was created by IMOD
        ("imodFlags", "i4"),  # Bit flags: 1 = bytes are stored as signed

        # Explanation of type of data
        ("idtype", "i2"),  # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
        ("lens", "i2"),
        # ("nd1", "i2"),  # for idtype = 1, nd1 = axis (1, 2, or 3)
        # ("nd2", "i2"),
        ("nphase", "i4"),
        ("vd1", "i2"),  # vd1 = 100. * tilt increment
        ("vd2", "i2"),  # vd2 = 100. * starting angle

        # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
        # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
        ("triangles", "f4", 6),  # 0,1,2 = original:  3,4,5 = current

        ("xorg", "f4"),  # Origin of image
        ("yorg", "f4"),
        ("zorg", "f4"),

        ("cmap", "S4"),  # Contains "MAP "
        ("stamp", "u1", 4),  # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian

        ("rms", "f4"),  # RMS deviation of densities from mean density

        ("nlabl", "i4"),  # Number of labels with useful data
        ("labels", "S80", 10)  # 10 labels of 80 charactors
    ]


def read_mrc(filename, filetype='image'):

    fd = open(filename, 'rb')
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]

    if header[0][3] == 1:
        data_type = 'int16'
    elif header[0][3] == 2:
        data_type = 'float32'
    elif header[0][3] == 4:
        data_type = 'single'
        nx = nx * 2
    elif header[0][3] == 6:
        data_type = 'uint16'

    data = np.ndarray(shape=(nx, ny, nz))
    imgrawdata = np.fromfile(fd, data_type)
    fd.close()

    if filetype == 'image':
        for iz in range(nz):
            data_2d = imgrawdata[nx*ny*iz:nx*ny*(iz+1)]
            data[:, :, iz] = data_2d.reshape(nx, ny, order='F')
    else:
        data = imgrawdata

    return header, data


def write_mrc(filename, img_data, header):

    if img_data.dtype == 'int16':
        header[0][3] = 1
    elif img_data.dtype == 'float32':
        header[0][3] = 2
    elif img_data.dtype == 'uint16':
        header[0][3] = 6

    fd = open(filename, 'wb')
    for i in range(len(rec_header_dtd)):
        header[rec_header_dtd[i][0]].tofile(fd)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]
    imgrawdata = np.ndarray(shape=(nx*ny*nz), dtype='uint16')
    for iz in range(nz):
        imgrawdata[nx*ny*iz:nx*ny*(iz+1)]=img_data[:,:,iz].reshape(nx*ny, order='F')
    imgrawdata.tofile(fd)

    fd.close()
    return

def readmrc_rawSIM(filename, img_size = 512, timepoints = 7):
    lr = np.zeros((timepoints, 1, img_size, img_size)).astype(np.float32)

    data = read_mrc(filename, filetype='image')[1]
    len = data.shape[2]
    y = np.split(data, len, axis=2)

    for timepoint in range(1, timepoints + 1):
        image_WF = np.zeros((data.shape[0], data.shape[1]), dtype=data.dtype)
        for i in range(3*3):
            image_out = y[i + timepoint * 9].reshape(data.shape[0], data.shape[1])
            image_WF += image_out

        image_WF = image_WF / 9
        image_WF = np.rot90(image_WF, 1)
        image_WF = image_WF.astype(np.uint16)

        lr[timepoint - 1,0,::] = image_WF
    return lr

def readmrc_GTSIM(filename, img_size = 512, timepoints = 7, factor = 2):
    hr = np.zeros((timepoints, 1, img_size*factor, img_size*factor)).astype(np.float32)

    data = read_mrc(filename, filetype='image')[1]
    len = data.shape[2]
    y = np.split(data, len, axis=2)

    for timepoint in range(1, timepoints + 1):
        image_GT = y[timepoint].reshape(data.shape[0], data.shape[1])
        image_GT = np.rot90(image_GT, 1)
        image_GT = image_GT.astype(np.float32)

        plt.imshow(image_GT, cmap='gray')
        plt.show()
        hr[timepoint - 1,0,::] = image_GT
    return hr


def FindAllSuffix(path: str, suffix: str, verbose: bool = False) -> list:
    ''' find all files have specific suffix under the path

    :param path: target path
    :param suffix: file suffix. e.g. ".json"/"json"
    :param verbose: whether print the found path
    :return: a list contain all corresponding file path (relative path)
    '''
    result = []
    if not suffix.startswith("."):
        suffix = "." + suffix
    for root, dirs, files in os.walk(path, topdown=False):
        # print(root, dirs, files)
        for file in files:
            if suffix in file:
                file_path = os.path.join(root, file)
                #os.remove(file_path)
                result.append(file_path)
                if verbose:
                    print(file_path)

    return result


def count_subfolders(folder_path):
    # 初始化子文件夹计数器
    subfolder_count = 0

    # 遍历文件夹中的所有内容
    for root, dirs, files in os.walk(folder_path):
        # 只统计根文件夹的子文件夹数量
        if root == folder_path:
            subfolder_count += len(dirs)
            break  # 只计算顶层文件夹，不深入子目录

    return subfolder_count


if __name__ == '__main__':
    mrc_path = 'F:\liushuran\DATA\VSR_Data\Ensconsin'
    save_path = 'F:\liushuran\DATA\VSR_Data\Ensconsin_h5'
    cell_count_valid = 5
    SNR_flag = 1 #SNR level = [1,2,3](2D) | [1,2](3D)

    n_frame = 7
    patchsize = 512
    factor = 2

    outputfile_train = os.path.join(save_path, 'train.h5')
    outputfile_valid = os.path.join(save_path, 'valid.h5')


    filelist = FindAllSuffix(mrc_path, '.mrc')
    cell_count = count_subfolders(mrc_path)
    num_of_SNR = len(filelist) // cell_count - 1


    #TRAIN
    h5_file = h5py.File(outputfile_train, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    patch_idx = 0

    for i in range(cell_count_valid, cell_count):
        lr_file = filelist[i * (num_of_SNR + 1) + SNR_flag - 1]
        lr_in = readmrc_rawSIM(lr_file, img_size=patchsize, timepoints=n_frame)

        hr_file = filelist[i * (num_of_SNR + 1) + num_of_SNR]
        hr_in = readmrc_GTSIM(hr_file, img_size=patchsize, timepoints=n_frame, factor=factor)

        lr_in = (lr_in - np.min(lr_in)) / (np.max(lr_in) - np.min(lr_in))
        hr_in = hr_in / np.max(hr_in)

        lr_group.create_dataset(str(patch_idx), data=lr_in)
        hr_group.create_dataset(str(patch_idx), data=hr_in)

        patch_idx = patch_idx + 1


