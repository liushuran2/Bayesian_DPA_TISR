import os
import struct
import torch
import numpy as np
import bitarray as ba
from utils.option import OptionClass, class2dict

"""
% The MRC image header has a fixed size of 1024 bytes. The information within the header includes a description of the extended header and image data. The column, row, and section are equivalent to the x, y, and z axes.
% 
% byte    
% Numbers  Variable Type Variable Name	  Contents
%  1 - 4	      i	     NumCol	            Number of columns. Typically, NumCol represents the number of image elements along the X axis.
%  5 - 8	      i	     NumRow	            Number of rows. Typically, NumRow represents the number of image elements along the Y axis.
% 9 - 12	      i	     NumSections	    Total number of sections. (NumZSec*NumWave*NumTimes)
% 13 - 16	      i    	PixelType	        The format of each pixel value. See the Pixel Data Types table below.
% 17 - 20	      i	     mxst	            Starting point along the X axis of sub-image (pixel number). Default is 0.
% 21 - 24	      i	     myst	            Starting point along the Y axis of sub-image (pixel number). Default is 0.
% 25 - 28	      i   	 mzst	            Starting point along the Z axis of sub-image (pixel number). Default is 0.
% 29 - 32	      i	     mx	                Sampling frequency in x; commonly set equal to one or the number of columns.
% 33 - 36	      i	     my	                Sampling frequency in y; commonly set equal to one or the number of rows.
% 37 - 40	      i	     mz	                Sampling frequency in z; commonly set equal to one or the number of z sections.
% 41 - 44	      f	     dx	                Cell dimension in x; for non-crystallographic data, set to the x sampling frequency times the x pixel spacing.
% 45 - 48	      f	     dy	                Cell dimension in y; for non-crystallographic data, set to the y sampling frequency times the y pixel spacing.
% 49 - 52	      f    	 dz	                Cell dimension in z; for non-crystallographic data, set to the z sampling frequency times the z pixel spacing.
% 53 - 56	      f	     alpha	            Cell angle (alpha) in degrees. Default is 90.
% 57 - 60	      f	     beta	            Cell angle (beta) in degrees. Default is 90.
% 61 - 64	      f	     gamma	            Cell angle (gamma) in degrees. Default is 90.
% 65 - 68	      i	     -	                Column axis. Valid values are 1,2, or 3. Default is 1.
% 69 - 72	      i	     -	                Row axis. Valid values are 1,2, or 3. Default is 2.
% 73 - 76	      i	     -	                Section axis. Valid values are 1,2, or 3. Default is 3.
% 77 - 80	      f	     min	            Minimum intensity of the 1st wavelength image.
% 81 - 84	      f	     max	            Maximum intensity of the 1st wavelength image.
% 85 - 88	      f	     mean	            Mean intensity of the first wavelength image.
% 89 - 92	      i	     nspg	            Space group number. Applies to crystallography data.
% 93 - 96	      i	     next	            Extended header size, in bytes.
% 97 - 98	      n	     dvid	            ID value. (-16224)
% 99 - 100	      n	     nblank	            Unused.
% 101 - 104	      i	     ntst	            Starting time index.
% 105 - 128	      c24	 blank	            Blank section. 24 bytes.
% 129 - 130       n	     NumIntegers	    Number of 4 byte integers stored in the extended header per section.
% 131 - 132	      n	     NumFloats	        Number of 4 byte floating-point numbers stored in the extended header per section.
% 133 - 134       n	     sub	            Number of sub-resolution data sets stored within the image. Typically, this equals 1.
% 135 - 136	      n	     zfac	            Reduction quotient for the z axis of the sub-resolution images.
% 137 - 140	      f	     min2	            Minimum intensity of the 2nd wavelength image.
% 141 - 144	      f	     max2	            Maximum intensity of the 2nd wavelength image.
% 145 - 148	      f	     min3	            Minimum intensity of the 3rd wavelength image.
% 149 - 152	      f	     max3	            Maximum intensity of the 3rd wavelength image.
% 153 - 156	      f	     min4	            Minimum intensity of the 4th wavelength image.
% 157 - 160       f	     max4	            Maximum intensity of the 4th wavelength image.
% 161 - 162	      n	     type	            Image type. See the Image Type table below.
% 163 - 164	      n	     LensNum	        Lens identification number.
% 165 - 166	      n	     n1	                Depends on the image type.
% 167 - 168	      n	     n2	                Depends on the image type.
% 169 - 170	      n	     v1	                Depends on the image type.
% 171 - 172	      n	     v2	                Depends on the image type.
% 173 - 176	      f	     min5	            Minimum intensity of the 5th wavelength image.
% 177 - 180	      f	     max5	            Maximum intensity of the 5th wavelength image.
% 181 - 182	      n	     NumTimes	        Number of time points.
% 183 - 184	      n	     ImgSequence	    Image sequence. 0=ZTW, 1=WZT, 2=ZWT.
% 185 - 188	      f	     -	                X axis tilt angle (degrees).
% 189 - 192	      f	     -	                Y axis tilt angle (degrees).
% 193 - 196	      f	     -	                Z axis tilt angle (degrees).
% 197 - 198	      n	     NumWaves	        Number of wavelengths.
% 199 - 200	      n	     wave1	            Wavelength 1, in nm.
% 201 - 202	      n	     wave2	            Wavelength 2, in nm.
% 203 - 204	      n	     wave3	            Wavelength 3, in nm.
% 205 - 206	      n	     wave4	            Wavelength 4, in nm.
% 207 - 28	      n	     wave5	            Wavelength 5, in nm.
% 209 - 212	      f	     z0	                Z origin, in um.
% 213 - 216	      f	     x0	                X origin, in um.
% 217 - 220	      f	     y0	                Y origin, in um.
% 221 - 224	      i	     NumTitles	        Number of titles. Valid numbers are between 0 and 10.
% 225 - 304	      c80	 -	                Title 1. 80 characters long.
% 305 - 384	      c80	 -	                Title 2. 80 characters long.
% 385 - 464	      c80	 -	                Title 3. 80 characters long.
% 465 - 544	      c80	 -	                Title 4. 80 characters long.
% 545 -624	      c80	 -	                Title 5. 80 characters long.
% 625-704	      c80	 -	                Title 6. 80 characters long.
% 705-784	      c80	 -	                Title 7. 80 characters long.
% 785-864	      c80	 -	                Title 8. 80 characters long.
% 865-944	      c80	 -         	        Title 9. 80 characters long.
% 945-1024	      c80	 -	                Title 10. 80 characters long.

% Pixel Data Types
% The data type of an image, stored in header bytes 13-16, is designated by one of the code numbers in the following table. 
%
% Code	C/C++ Macro	        Description
%  0	IW_BYTE	            1-byte unsigned integer
%  1	IW_SHORT	        2-byte signed integer
%  2	IW_FLOAT	        4-byte floating-point (IEEE)
%  3	IW_COMPLEX_SHORT	4-byte complex value as 2 2-byte signed integers
%  4	IW_COMPLEX	        8-byte complex value as 2 4-byte floating-point values
%  5	IW_EMTOM	        2-byte signed integer
%  6	IW_USHORT	        2-byte unsigned integer
%  7	IW_LONG	            4-byte signed integer
"""

HEADER_SIZE = 1024 # 1024 bytes


# ----------------------------------------
# Function: Correct header
# ----------------------------------------
def correct_header(header):
    if not isinstance(header, list): header = list(header)
    if header[24] % 65536 == 49312: pass
    elif header[24] // 65536 == 49312:
        for idx in [24, 32, 33, 40, 41, 42, 45, 49, 50, 51]:
            header[idx] = header[idx] % 65536 * 65536 + header[idx] // 65536
    else: raise IOError("check the mrc file header!")
    return header


# ----------------------------------------
# Class: ReadMRC
# ----------------------------------------
# Function:
#       -> read header and option
#       -> read sim raw .mrc files
#       -> read general .mrc files including otf
#       -> batch reading (deprecated for the lack of ending signal)
#       -> Format as [timepoint, (num_oritentation, num_phase, num_wave), depth, height, width] array
# API:
#       -> __init__ read header and get option
#       -> read some data and convert to [tuple, list, torch]
#       -> read next batch
# [safe]:
#    * do not deliver pointer (handle) *
#    * control pointer scope with 'with' *
# ----------------------------------------

class ReadMRC:
    # ----------------------------------------
    # __init__ method (structor function in c++)
    # ----------------------------------------
    def __init__(self, file, opt=None, is_SIM_rawdata=True, big_endian=None):

        self.file = file

        self.is_SIM_rawdata = is_SIM_rawdata

        self.header, self.big_endian = self.__read_mrc_header(big_endian)

        self.header = correct_header(self.header)

        self.big_endian_signal = '>' if self.big_endian else '<'

        self.opt = self.__get_option_from_mrc_header(opt)
        
        self.timepoint_have_been_read = 0

    # ----------------------------------------
    # read header
    # ----------------------------------------
    def __read_mrc_header(self, big_endian=None):

        file = self.file

        # auto choose big_endian / small endian type
        if big_endian is None:
            big_endian = False
            with open(file, 'rb') as f:
                header = struct.unpack('<256I', f.read(HEADER_SIZE))
            if header[3] > 7:
                big_endian = True
                with open(file, 'rb') as f:
                    header = struct.unpack('>256I', f.read(HEADER_SIZE))
        # < < < < < < < < < < < < <

        # small_endian type > > > >
        elif big_endian is True:
            with open(file, 'rb') as f:
                header = struct.unpack('>256I', f.read(HEADER_SIZE))
        # < < < < < < < < < < < < <

        # big_endian type > > > > >
        elif big_endian is False:
            with open(file, 'rb') as f:
                header = struct.unpack('<256I', f.read(HEADER_SIZE))
        # < < < < < < < < < < < < <

        # data offset (pos of data) should be 1024
        assert header[23] == 0, "data offset should be zero! update the code if necessary"

        return header, big_endian

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    def update(self):
        opt = self.opt
        self.opt.num_sections = int((os.path.getsize(self.file) - HEADER_SIZE) // (opt.num_pixel_width * opt.num_pixel_height * opt.byte_per_pixel))

        self.opt.num_timepoint = int((os.path.getsize(self.file) - HEADER_SIZE) // (opt.num_pixel_width * opt.num_pixel_height *
                                        opt.num_pixel_depth * opt.num_phase * opt.num_orientation * opt.num_channel * opt.byte_per_pixel))

    # ----------------------------------------
    # header -> imaging options
    # ----------------------------------------
    def __get_option_from_mrc_header(self, opt):

        if opt is None: opt = OptionClass()
        header = self.header

        opt.num_pixel_width = header[0]
        opt.num_pixel_height = header[1]

        # Total number of sections. (NumZSec*NumWave*NumTimes*NumOrient*NumPhase)
        opt.num_sections = header[2]
        # numbers of time points
        opt.num_timepoint = header[45] % 65536
        # number of channels (equals to colors in general) and emission wavelength (in nm)
        opt.num_channel = max(1, min(struct.unpack('2h', struct.pack('I', header[49]))))
        opt.em_wavelength = []
        for idx in range(opt.num_channel):
            opt.em_wavelength.append(struct.unpack('2h', struct.pack('I', header[49+(idx+1)//2]))[(idx+1)%2])

        if self.is_SIM_rawdata:
            # number of phases
            opt.num_phase = header[41] % 65536
            # number of orientation
            opt.num_orientation = header[41] // 65536
            # raw data have a single channel
            assert opt.num_channel == 1, "chennel of the raw data should be 1"
        else:
            opt.num_phase = 1
            opt.num_orientation = 1

        # spacing rate, i.e., pixel size in space domain (in um)
        opt.width_space_sampling = struct.unpack('f', struct.pack('I', header[10]))[0]
        opt.height_space_sampling = struct.unpack('f', struct.pack('I', header[11]))[0]
        opt.depth_space_sampling = abs(struct.unpack('f', struct.pack('I', header[12]))[0])

        opt.num_pixel_depth = int(opt.num_sections // (opt.num_phase * opt.num_orientation * opt.num_timepoint * opt.num_channel))

        # data save mode. Z: depth | W: number of wave | T: timepoints
        opt.data_save_order = {'0':'ZTW','1':'WZT','2':'ZWT'}[str(header[45] // 65536)]

        opt.is_complex, opt.byte_per_pixel, opt.dtype_symbol = {'6':[False, 2, 'H'], '4':[True, 8, 'f'], '2':[False, 4, 'f']}[str(header[3])]

        opt.pixel_per_timepoint = opt.num_pixel_width * opt.num_pixel_height * opt.num_pixel_depth * opt.num_phase * opt.num_orientation * opt.num_channel

        # additional option:
        # order
        opt.num_order = ((1 + opt.num_phase) // 2)
        # freq sampling (in 1/um)
        opt.width_freq_sampling = 0 if opt.width_space_sampling == 0 else 1 / (opt.num_pixel_width * opt.width_space_sampling)
        opt.height_freq_sampling = 0 if opt.height_space_sampling == 0 else 1 / (opt.num_pixel_height * opt.height_space_sampling)
        opt.depth_freq_sampling = 0 if opt.depth_space_sampling == 0 else 1 / (opt.num_pixel_depth * opt.depth_space_sampling)

        # dxy dkr
        opt.radial_space_sampling = max(opt.width_space_sampling, opt.height_space_sampling)
        opt.radial_freq_sampling = min(opt.width_freq_sampling, opt.height_freq_sampling)

        return opt

    # ----------------------------------------
    # get total binary data
    # ----------------------------------------
    def __get_total_data(self):

        self.update()
        
        pixel_read = self.opt.num_pixel_width * self.opt.num_pixel_height * self.opt.num_sections

        if self.opt.is_complex: pixel_read *= 2

        with open(self.file, 'rb') as f:
            raw_image = np.fromfile(f, dtype=self.big_endian_signal + self.opt.dtype_symbol, count=pixel_read, offset=HEADER_SIZE)

        if self.opt.is_complex: raw_image = raw_image[0::2] + 1j * raw_image[1::2]

        # with open(self.file, 'rb') as f:
        #     f.seek(HEADER_SIZE)
        #     raw_image = struct.unpack(self.big_endian_signal + str(pixel_read) + self.opt.dtype_symbol, f.read(pixel_read * self.opt.byte_per_pixel))

        return raw_image

    # ----------------------------------------
    # get binary data at given timepoints
    # ----------------------------------------
    def __get_timepoint_data(self, begin_timepoint, read_timepoint):

        self.update()
        
        assert begin_timepoint + read_timepoint <= self.opt.num_timepoint, 'out of mrc length'

        pixel_pass = begin_timepoint * self.opt.pixel_per_timepoint

        pixel_read = read_timepoint * self.opt.pixel_per_timepoint

        if self.opt.is_complex:
            pixel_pass *= 2
            pixel_read *= 2

        with open(self.file, 'rb') as f:
            raw_image = np.fromfile(f, dtype=self.big_endian_signal+self.opt.dtype_symbol, count=pixel_read, offset=HEADER_SIZE + pixel_pass * self.opt.byte_per_pixel)

        # with open(self.file, 'rb') as f:
        #     f.seek(HEADER_SIZE + pixel_pass * self.opt.byte_per_pixel)
        #     raw_image = struct.unpack(self.big_endian_signal + str(pixel_read) + self.opt.dtype_symbol, f.read(pixel_read * self.opt.byte_per_pixel))

        return raw_image
    
    # ----------------------------------------
    # convert to [T, O, P, C, D, H, W]
    # ----------------------------------------
    def __convert_dtype(self, x, convert_to_tensor=True, do_reshape=True):

        assert self.opt.num_channel == 1, "chennel of the raw data should be 1"

        if convert_to_tensor:
            x = torch.from_numpy(x.astype(np.complex64)) if self.opt.is_complex else torch.from_numpy(x.astype(np.float32))

        if do_reshape:
            (W, H, D, O, P, C) = (self.opt.num_pixel_width, self.opt.num_pixel_height, self.opt.num_pixel_depth,
                                  self.opt.num_orientation, self.opt.num_phase, self.opt.num_channel)
            T = int(x.squeeze().shape[0] / (W * H * D * O * P * C))

            fn = torch.Tensor.permute if convert_to_tensor else np.transpose

            if self.opt.data_save_order == 'ZTW':  # [C, T, D, O, P, H, W] -> [T, O, P, C, D, H, W]
                x = fn(x.reshape(C, T, D, O, P, H, W), (1, 3, 4, 0, 2, 5, 6))
            elif self.opt.data_save_order == 'WZT':  # [T, D, C, O, P, H, W] -> [T, O, P, C, D, H, W]
                x = fn(x.reshape(T, D, C, O, P, H, W), (0, 3, 4, 2, 1, 5, 6))
            elif self.opt.data_save_order == 'ZWT':  # [T, C, D, O, P, H, W] -> [T, O, P, C, D, H, W]
                x = fn(x.reshape(T, C, D, O, P, H, W), (0, 3, 4, 1, 2, 5, 6))
            else: raise NotImplementedError("check the code!")

        return x

    # ----------------------------------------
    # get total data as torch mat
    # ----------------------------------------
    def get_total_data_as_mat(self, convert_to_tensor=True, do_reshape=True):
        x = self.__get_total_data()
        x = self.__convert_dtype(x, convert_to_tensor=convert_to_tensor, do_reshape=do_reshape)
        return x

    # ----------------------------------------
    # get data at given timepoints as torch mat
    # ----------------------------------------
    def get_timepoint_data_as_mat(self, begin_timepoint, read_timepoint, convert_to_tensor=True, do_reshape=True):
        x = self.__get_timepoint_data(begin_timepoint, read_timepoint)
        x = self.__convert_dtype(x, convert_to_tensor=convert_to_tensor, do_reshape=do_reshape)
        return x

    # ----------------------------------------
    # get next timepoint (batch)
    # ----------------------------------------
    def get_next_timepoint_batch(self, batchsize=1, convert_to_tensor=True, do_reshape=True, strict=False):
        # self.update()
        begin_timepoint = self.timepoint_have_been_read
        read_timepoint = batchsize

        if begin_timepoint + read_timepoint <= self.opt.num_timepoint:
            self.timepoint_have_been_read += batchsize
            return self.get_timepoint_data_as_mat(begin_timepoint, read_timepoint, convert_to_tensor=convert_to_tensor, do_reshape=do_reshape)
        else:
            if strict is False:
                timepoint_can_be_read = self.opt.num_timepoint - begin_timepoint
                if timepoint_can_be_read != 0:
                    self.timepoint_have_been_read += timepoint_can_be_read
                    return self.get_timepoint_data_as_mat(begin_timepoint, timepoint_can_be_read, convert_to_tensor=convert_to_tensor, do_reshape=do_reshape)
                else:
                    return None
            else:
                return None


# ----------------------------------------
# Function: make header
# ----------------------------------------
def make_sr_header(header, opt):

    header = correct_header(header)

    header[41] = 65536 * 1 + 1  # orientation = 1, phase = 1
    header[45] = 65536 * 0 + 0  # data_save_order = 0 (ZTW), num_timepoint = 0

    header[0] = opt.num_pixel_width * opt.zoom_factor_xy
    header[1] = opt.num_pixel_height * opt.zoom_factor_xy
    header[24] = opt.num_pixel_depth * opt.zoom_factor_z * 65536 + 49312

    header[2] = opt.num_pixel_depth * opt.num_timepoint * opt.num_channel * 1 * 1 # num_sections = num_pixel_depth * num_timepoint * num_channel * orientation * phase


    header[10] = struct.unpack('I', struct.pack('f', opt.width_space_sampling / opt.zoom_factor_xy ))[0]
    header[11] = struct.unpack('I', struct.pack('f', opt.height_space_sampling / opt.zoom_factor_xy ))[0]
    header[12] = struct.unpack('I', struct.pack('f', opt.depth_space_sampling / opt.zoom_factor_z ))[0]
    header[3] = 2  # single

    return header


# ----------------------------------------
# Class: WriteMRC
# ----------------------------------------
# Method:
#       -> __init__ make opt and header for mrc file to be writed from opt
#       -> write header
#       -> [api] write data in append mode
# [safe]:
#   do not deliver pointer (handle)
#   control pointer scope with 'with'
# ----------------------------------------

class WriteMRC:
    # ----------------------------------------
    # __init__ method (structor function in c++)
    # ----------------------------------------
    def __init__(self, file, header, big_endian=False):
        self.file = file
        self.header = header
        self.header = tuple(correct_header(self.header))
        self.opt = self.get_option_from_mrc_header()
        self.big_endian = big_endian
        self.big_endian_signal = '>' if self.big_endian else '<'
        self.write_mrc_header()
        self.timepoint_have_been_written = 0

    def get_option_from_mrc_header(self):

        opt = OptionClass()
        header = self.header

        # [T, O, P, C, Z, H, W]
        opt.num_orientation = header[41] // 65536
        opt.num_phase = header[41] % 65536
        opt.num_channel = max(1, struct.unpack('2h', struct.pack('I', header[49]))[0])
        opt.num_pixel_depth = max(1, header[24] // 65536)
        opt.num_pixel_height = header[1]
        opt.num_pixel_width = header[0]

        # data saving mode, Z: depth | W: number of wave | T: timepoints
        opt.data_save_order = {'0':'ZTW','1':'WZT','2':'ZWT'}[str(header[45] // 65536)]
        opt.is_complex, opt.byte_per_pixel, opt.dtype_symbol = {'6':[False, 2, 'H'], '4':[True, 8, 'f'], '2':[False, 4, 'f']}[str(header[3])]
        opt.pixel_per_timepoint = opt.num_pixel_width * opt.num_pixel_height * opt.num_pixel_depth * opt.num_phase * opt.num_orientation * opt.num_channel

        return opt

    def write_mrc_header(self):
        with open(self.file, 'wb') as f:
            for idx in range(int(HEADER_SIZE // 4)):
                f.write(struct.pack(self.big_endian_signal + 'I', self.header[idx]))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    def update(self):
        opt = self.opt
        header = list(self.header)
        opt.num_timepoint = self.timepoint_have_been_written
        self.opt.num_sections = opt.num_timepoint * opt.num_pixel_depth * opt.num_phase * opt.num_orientation * opt.num_channel
        header[2] = opt.num_sections
        header[45] = header[45] // 65536 + opt.num_timepoint
        self.header = tuple(header)
        with open(self.file, 'rb+') as f:
            for idx in range(int(HEADER_SIZE // 4)):
                f.write(struct.pack(self.big_endian_signal + 'I', self.header[idx]))

    # ----------------------------------------
    # write binary data
    # ----------------------------------------
    def write_mrc_append(self, x):
        assert len(x.shape) == 1, "check code!"
        save_data_type = {'6': 'uint16', '2': 'float32', '4': 'complex64'}[str(self.header[3])]

        if save_data_type == 'complex64':
            assert self.opt.is_complex
            x = x.to(np.complex64)
            # This is a strange implement inapplicable to visualization. Anyway we follow it.
            x_complex = np.zeros(2 * x.shape[0], dtype=np.float32)
            x_complex[0::2] = np.real(x)
            x_complex[1::2] = np.imag(x)
            x = x_complex
            x = x.astype(self.big_endian_signal + 'f')
        # elif dtype_symbol == 'e':
        #     x = x.astype(self.big_endian_signal+'e')
        elif save_data_type == 'float32':
            x = x.astype(self.big_endian_signal + 'f')
        elif save_data_type == 'uint16':
            x = x.clip(0, 65535).round().astype(self.big_endian_signal + 'H')
        # elif dtype_symbol == 'h':
        #     x = x.clip(-32768, 32767).round().astype(self.big_endian_signal+'h')
        else:
            raise RuntimeError('the dtype of stack to be saved must be uint16, float32, or complex64')

        with open(self.file, 'ab') as f:
            temp = x.tobytes()
            # f.write(data.tobytes()) | x.tofile(f)
            b = ba.bitarray()
            b.frombytes(temp)
            b.tofile(f)

    # ----------------------------------------
    # from torch [T, O, P, C, D, H, W] convert to binary
    # ----------------------------------------
    def __convert_dtype(self, x):

        if isinstance(x, torch.Tensor):
            from_tensor=True
            fn = torch.Tensor.permute
        elif isinstance(x, np.ndarray):
            from_tensor=False
            fn = np.transpose
        else: raise NotImplementedError('check the code')

        if len(x.shape) == 1: need_reshape=False
        elif len(x.shape) == 7: need_reshape=True
        else: raise NotImplementedError('check the code')

        opt = self.opt
        (T, O, P, C, D, H, W) = x.shape

        assert (O, P, C, D, H, W) == (opt.num_orientation, opt.num_phase, opt.num_channel, opt.num_pixel_depth, opt.num_pixel_height, opt.num_pixel_width), "wrong data size"

        assert C == 1, "saving mulit-wave data is not supported now"

        if opt.data_save_order == 'ZTW': # [T, O, P, C, D, H, W] -> [C, T, D, O, P, H, W]
            x = fn(x, (3, 0, 4, 1, 2, 5, 6)).reshape(-1)
        elif opt.data_save_order == 'WZT': # [T, O, P, C, D, H, W] -> [T, D, C, O, P, H, W]
            x = fn(x, (0, 4, 3, 1, 2, 5, 6)).reshape(-1)
        elif opt.data_save_order == 'ZWT': # [T, O, P, C, D, H, W] -> [T, C, D, O, P, H, W]
            x = fn(x, (0, 3, 4, 1, 2, 5, 6)).reshape(-1)
        else: raise NotImplementedError("check the code")

        if from_tensor:
            x = x.cpu().numpy()

        return x, T

    # ----------------------------------------
    # write torch mat as binary in append mode
    # ----------------------------------------
    def write_data_append(self, x):
        x, T = self.__convert_dtype(x)
        self.write_mrc_append(x)
        self.timepoint_have_been_written += T
        self.update()


def write_mrc_image(data, path, samplint_rate=None, datatype='single'):
    """
    Fast Write [without spacing/freq sampling rate] only for debug!
    """
    header = [512, 512, 189, 6, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1119092736, 1119092736, 1119092736, 1, 2, 3, 0, 0, 0, 0, 0, 49312, 0, 0, 0, 0, 0, 0, 0,
              131072, 0, 0, 1176256512, 0, 1176256512, 0, 1176256512, 0, 196611, 7274496, 0, 1176256512, 21, 0, 0, 0, 1, 0, 0, 0, 0, 0, 10, 1819043171, 51, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    (T, O, P, C, D, H, W) = data.shape
    header[45] = 65536 * 0 + T  # data_save_order = 0 (ZTW), num_timepoint = T
    header[41] = 65536 * O + P  # orientation = O, phase = P
    header[24] = D * 65536 + 49312
    header[1] = H
    header[0] = W
    header[2] = T * O * P * C * D
    if datatype == 'single': header[3] = 2  # 2=single, 6=uint16
    elif datatype == 'uint16': header[3] = 6  # 2=single, 6=uint16
    else: raise NotImplementedError
    if samplint_rate is not None:
        header[10] = struct.unpack('I', struct.pack('f', samplint_rate[0]))[0]
        header[11] = struct.unpack('I', struct.pack('f', samplint_rate[1]))[0]
        header[12] = struct.unpack('I', struct.pack('f', samplint_rate[2]))[0]
    wm = WriteMRC(path, header)
    wm.write_data_append(data)

# ----------------------------------------
# debug/test function
# ----------------------------------------
def main():
    pass
    # ----------------------------------------
    # test reading 2d/3d stack
    # ----------------------------------------
    # file = r'K:\AIA_SIM_TestData\lifeact488\TIRF-488-10ms_cam1_step1_001.mrc'
    # # file = r'D:\不用的数据\data_PKMR_3D-SIM_training\c15-560-1.26-0.4W-0.16step-0.175cc-SLM6.1-LCde0_20201202_173932_good-\3D-560-4step_cam2_step4_001.mrc'
    # file = r'D:\不用的数据\data_2017-6-14-5Kframe-highspeed\1XmEmerald-KDEL\cell1-488-1.35-0.4ms-300mW_20170614_231340(ok-3.5k)\TIRF488_cam1_0.mrc' # 6G
    # # file = r'D:\time-lapse-imaging-data_Lifeact-cos7_mEmerald_JiangT_202105\cell30_20210520_214153\TIRF-488-1ms_cam1_step1_001.mrc' # 1.3G
    # CR = ReadMRC(file) # auto run __init__ while instantiate
    # tic = time.time()
    # arr = CR.get_total_data_as_mat()  # run 'method'
    # arr = CR.get_timepoint_data_as_mat(0, 1) # run 'method'
    # print(time.time() - tic) # 1.3G data = 8.64s | 6G = 33s
    # print(arr.shape)
    # plt.imshow(arr[0,0,0,0,0,...])
    # plt.show()
    # # if arr is not None: print(arr.shape)
    # # tic = time.time()
    # # while True:
    # #     arr = CR.get_next_timepoint_batch(1) # run 'method'
    # #     print(time.time()-tic)
    # #     # print(len(arr))
    # #     # if arr is not None: print(arr.shape)

    # ----------------------------------------
    # test reading 2d/3d stack and Queue
    # ----------------------------------------
    # from queue import Queue, LifoQueue, PriorityQueue
    # import threading
    # file = r'D:\AIASIM\TIRF-488-10ms_cam1_step1_001.mrc'
    # CR = ReadMRC(file) # auto run __init__ while instantiate
    # print(CR.opt)
    # q = Queue(maxsize=10)
    #
    # def put_queue(CR):
    #     count = 0
    #     while count < CR.opt.num_timepoint:
    #         q.put(CR.get_next_timepoint_batch(1, 'tensor'))
    #         count += 1
    #         print('in | ' + str(count))
    #
    # def get_queue(CR):
    #     count = 0
    #     while count < CR.opt.num_timepoint:
    #         if q.get() is not None:
    #             count = count + 1
    #             time.sleep(0.1)
    #             print('out | ' + str(count))
    #         q.task_done()
    #
    # p1 = threading.Thread(target=put_queue, args=(CR,))
    # p2 = threading.Thread(target=get_queue, args=(CR,))
    # p1.start()
    # p2.start()

    # ----------------------------------------
    # test reading 2d/3d otf
    # ----------------------------------------
    # file = r'D:\batfile\NS-100X1.49NA\37-New\TIRF488_OTF2d.mrc'
    # file = r'D:\batfile\NS-100X1.49NA\37-3D-191217\3D-488_cam1_step1_001_otf3d-ang0-28-29-2-1.46-Final.mrc'
    # CR = ReadMRC(file, is_SIM_rawdata=False)  # auto run __init__ while instantiate
    # print(CR.big_endian)
    # arr = CR.get_total_data_as_mat()
    # print(arr.shape)

    # ----------------------------------------
    # test reading SR stack
    # ----------------------------------------
    # file = r'D:\2021_0727_yangji\SIM_data\cell_20210727_105722\TIRF-560-5ms_N_cam2_step1_001_L.mrc'
    # CR = ReadMRC(file, is_SIM_rawdata=False) # auto run __init__ while instantiate
    # print(CR.opt)
    # # arr = CR.get_total_data_as_mat()  # run 'method'
    # # arr = CR.get_timepoint_data_as_mat(0, 1) # run 'method'
    # # if arr is not None: print(arr.shape)
    # while True:
    #     arr = CR.get_next_timepoint_batch() # run 'method'
    #     if arr is not None: print(arr.shape)

    # ----------------------------------------
    # test reading mrc saved by other lib
    # warning: other lib saving 'timepoint' as 'depth'
    # ----------------------------------------
    # file = r'D:\data_MTs_COS7_mEmerald_JiangT_202105\20210526_121151\488-1.35-1ms_cam1_step1_001_SR_nonclip.mrc'
    # CR = ReadMRC(file, is_SIM_rawdata=False) # auto run __init__ while instantiate
    # print(CR.opt)
    # arr = CR.get_total_data_as_mat()  # run 'method'
    # arr = CR.get_timepoint_data_as_mat(0, 1) # run 'method'
    # if arr is not None: print(arr.shape)
    # while True:
    #     arr = CR.get_next_timepoint_batch() # run 'method'
    #     if arr is not None: print(arr.shape)

# def main():
#     pass
#     # ----------------------------------------
#     # test writing 2d/3d stack
#     # ----------------------------------------
#     from utils.mrc_read import ReadMRC
#
#     file = r'K:\AIA_SIM_TestData\lifeact488\TIRF-488-10ms_cam1_step1_001.mrc'
#     CR = ReadMRC(file) # auto run __init__ while instantiate
#     arr = CR.get_total_data_as_mat()  # run 'method'
#
#     opt = CR.opt
#     opt.zoom_factor_xy = 2
#     opt.zoom_factor_z = 1
#     from torch.nn.functional import interpolate
#     arr = arr + torch.randn(1).to(arr.device)
#     arr = interpolate(torch.mean(arr,(1,2)).reshape(21,1,512,512), size=(1024,1024)).reshape(21,1,1,1,1,1024,1024)
#
#     file_save = 'K:\AIA_SIM_TestData\lifeact488\debug.mrc'
#
#     CW = WriteMRC(file_save, make_sr_header(CR.header, CR.opt))
#
#     # too fast to be queried if it is thread-safe
#     with timeblock('write time'):
#         CW.write_data_append(arr)
#
if __name__ == '__main__':

    main()