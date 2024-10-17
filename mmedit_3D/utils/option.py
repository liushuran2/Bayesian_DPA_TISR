import struct
from collections import OrderedDict
import json
import os
import platform
from utils.tools import get_timestamp, mkdir


# ----------------------------------------
# class option <same as 'struct' in C>
# ----------------------------------------
class OptionClass:
    pass

# ----------------------------------------
# dict2class
# ----------------------------------------
class dict2class(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

# ----------------------------------------
# class2dict
# ----------------------------------------
def class2dict(obj):
    dic = {}
    for fieldkey in dir(obj):
        fieldvaule = getattr(obj, fieldkey)
        if not fieldkey.startswith("__") and not callable(fieldvaule) and not fieldkey.startswith("_"):
            dic[fieldkey] = fieldvaule
    return dic

# ----------------------------------------
# json -> [struct-like class] opt
# ----------------------------------------
def sir_parse2class(opt_path):
    """
    :param opt_path: [str] path of option saved in json file
    :return: [struct-like class]
    """
    return dict2class(sir_parse2dict(opt_path))

# ----------------------------------------
# json -> [struct-like class] opt
# ----------------------------------------
def sir_parse2dict(opt_path):
    """
    :param opt_path: [str] path of option saved in json file
    :return: dict
    """
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    if platform.system().lower() in ['windows']:
        pass
    else:
        # only for research
        json_str = json_str.replace('\\\\', '/')
        json_str = json_str.replace('K:', '/media/zkyd/LiuJiahao')
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt = dict2nonedict(opt)
    return opt

if __name__ == '__main__':
    sir_parse2dict(r'K:\AIA_SIM_TestData\data_Lifeact_COS7_mEmerald_HSNR_bioSR_all\20211120_101810\TIRF-488-10ms_cam1_step1_001.json')

# ----------------------------------------
# json opt -> [dict] opt
# ----------------------------------------
def AI_parse(opt_path, is_train=True, newDirTag=None):

    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # initialize opt
    if platform.system().lower() in ['windows']:
        pass
    else:
        # only for research
        json_str = json_str.replace('\\\\', '/')
        json_str = json_str.replace('K:', '/media/zkyd/LiuJiahao')
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt = dict2nonedict(opt)

    opt['opt_path'] = opt_path
    opt['is_train'] = is_train

    # path -> expanduser
    opt['dataset_root'] = os.path.expanduser(opt['dataset_root'])
    if opt['pretrained_netG']: opt['pretrained_netG'] = os.path.expanduser(opt['pretrained_netG'])
    if opt['model'] in ['gan-denoise', 'gan-reconstruction', 'gan-super-resolution']:
        if opt['pretrained_netD']: opt['pretrained_netD'] = os.path.expanduser(opt['pretrained_netD'])

    # path for model / valimages / output_txt
    if "lastTrainDir" not in opt.keys():
        opt['save_path'] = os.path.join(opt['dataset_root'], '{}_{}_{}_{}_{}'.format(opt['supervise'], opt['model'], opt['net_G'], opt['G_lossfn_type'], get_timestamp()))
        if newDirTag is not None:
            opt['save_path'] = opt['save_path'] + newDirTag
        if is_train:
            mkdir(opt['save_path'])
            mkdir(os.path.join(opt['save_path'], 'model'))
            mkdir(os.path.join(opt['save_path'], 'image'))
            mkdir(os.path.join(opt['save_path'], 'plot'))
    else:
        opt['save_path'] = os.path.join(opt['dataset_root'],opt["lastTrainDir"])



    # ----------------------------------------
    # IF training GaN
    # ----------------------------------------
    if opt['model'] in ["gan-denoise", "gan-reconstruction", "gan-super-resolution", "circle-raw-reconstruction", "circle-wf-reconstruction"]: # GAN + DN, REC, SR
        opt['train_gan'] = True
    elif opt['model'] in ["denoise", "reconstruction", "super-resolution", "rdl", "rdl_n2n", "rdl_SRn2n", 'pattern-only-rdl']: # DN, REC, SR
        opt['train_gan'] = False
    else:
        raise NotImplementedError

    # ----------------------------------------
    # IF Predictor <for rdl>
    # ----------------------------------------
    opt['need_predictor'] = False
    if opt['model'] in ["rdl"]:
        opt['need_predictor'] = True

    # ----------------------------------------
    # set channel
    # ----------------------------------------
    # for plain - provide data needed
    if opt['model'] in ["denoise", "gan-denoise"]: # DN
        opt['num_channels_in'] = (1, 1, 1, 1) # TOPC
        opt['num_channels_out'] = (1, 1, 1, 1)  # TOPC
        opt['scale'] = 1
        if opt['data'] in ['linear-sim', 'single-slice-sim']:
            opt['scale_having_done'] = 2
        elif opt['data'] == 'non-linear-SIM':
            opt['scale_having_done'] = 3
        else:
            raise NotImplementedError
    elif opt['model'] in ["reconstruction", "gan-reconstruction"]: # REC
        opt['num_channels_out'] = (1, 1, 1, 1)
        if opt['data'] == 'linear-sim':
            opt['num_channels_in'] = (1, 3, 3, 1) # TOPC
            opt['scale'] = 2
        elif opt['data'] == 'single-slice-sim':
            opt['num_channels_in'] = (1, 3, 5, 1) # TOPC
            opt['scale'] = 2
        elif opt['data'] == 'non-linear-SIM':
            opt['num_channels_in'] = (1, 5, 5, 1) # TOPC
            opt['scale'] = 3
        else:
            raise NotImplementedError
    elif opt['model'] in ["super-resolution", "gan-super-resolution"]: # SR
        opt['num_channels_out'] = (1, 1, 1, 1)
        if opt['data'] in ['linear-sim', 'single-slice-sim']:
            opt['num_channels_in'] = (1, 1, 1, 1) # TOPC
            opt['scale'] = 2
        elif opt['data'] == 'non-linear-SIM':
            opt['num_channels_in'] = (1, 1, 1, 1) # TOPC
            opt['scale'] = 3
        else:
            raise NotImplementedError

    # for general reconstruction - provide all data
    elif opt['model'] in ['rdl', 'rdl_n2n', 'rdl_SRn2n', 'pattern-only-rdl', 'circle-raw-reconstruction', 'circle-wf-reconstruction']: # raw + circle
        if opt['data'] in ['linear-sim']:
            opt['num_channels_raw'] = (1, 3, 3, 1)  # TOPC
            opt['scale'] = 2
        elif opt['data'] in ['single-slice-sim']:
            opt['num_channels_raw'] = (1, 3, 5, 1)  # TOPC
            opt['scale'] = 2
        elif opt['data'] == 'non-linear-SIM':
            opt['num_channels_raw'] = (1, 5, 5, 1)  # TOPC
            opt['scale'] = 3
        else:
            raise NotImplementedError
        opt['num_channels_sim'] = (1, 1, 1, 1)

        if opt['model'] in ['rdl', 'pattern-only-rdl', 'rdl_n2n', 'rdl_SRn2n']:  # rdl
            opt['num_channels_in'] = opt['num_channels_raw']
            opt['num_channels_out'] = opt['num_channels_raw']
        elif opt['model'] in ['circle-raw-reconstruction', 'circle-wf-reconstruction']: # raw + circle
            opt['num_channels_in'] = opt['num_channels_raw']
            opt['num_channels_out'] = opt['num_channels_sim']

    else:
        raise NotImplementedError

    # ----------------------------------------
    # set dataroot
    # ----------------------------------------
    opt['train_dataroot_para'] = os.path.join(opt['dataset_root'], 'train_Para')
    opt['train_dataroot_json'] = os.path.join(opt['dataset_root'], 'train_Json')
    opt['val_dataroot_para'] = os.path.join(opt['dataset_root'], 'val_Para')
    opt['val_dataroot_json'] = os.path.join(opt['dataset_root'], 'val_Json')
    # FULL-SUP
    if opt['supervise'] in ["full-supervised"]:
        # SUP + SR
        if opt['model'] in ["super-resolution", "gan-super-resolution"]:
            opt['train_dataroot_input'] = os.path.join(opt['dataset_root'], 'train_WF')
            opt['val_dataroot_input'] = os.path.join(opt['dataset_root'], 'val_WF')
            opt['train_dataroot_target'] = os.path.join(opt['dataset_root'], 'train_SIM_HSNR')
            opt['val_dataroot_target'] = os.path.join(opt['dataset_root'], 'val_SIM_HSNR')
        # SUP + DN
        elif opt['model'] in ["denoise", "gan-denoise"]:
            opt['train_dataroot_input'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_1')
            opt['train_dataroot_target'] = os.path.join(opt['dataset_root'], 'train_SIM_HSNR')
            opt['val_dataroot_input'] = os.path.join(opt['dataset_root'], 'val_SIM_LSNR_1')
            opt['val_dataroot_target'] = os.path.join(opt['dataset_root'], 'val_SIM_HSNR')
        # SUP + REC
        elif opt['model'] in ["reconstruction", "gan-reconstruction"]:
            opt['train_dataroot_input'] = os.path.join(opt['dataset_root'], 'train_Raw_LSNR_1')
            opt['train_dataroot_target'] = os.path.join(opt['dataset_root'], 'train_SIM_HSNR')
            opt['val_dataroot_input'] = os.path.join(opt['dataset_root'], 'val_Raw_LSNR_1')
            opt['val_dataroot_target'] = os.path.join(opt['dataset_root'], 'val_SIM_HSNR')
        # SUP + RDL / circle-network
        elif opt['model'] in ['rdl','rdl_SRn2n', 'pattern-only-rdl', 'circle-raw-reconstruction', 'circle-wf-reconstruction']: # use all data
            opt['train_dataroot_raw_input'] = os.path.join(opt['dataset_root'], 'train_Raw_LSNR_1')
            opt['train_dataroot_raw_target'] = os.path.join(opt['dataset_root'], 'train_Raw_HSNR')
            opt['train_dataroot_sim_target'] = os.path.join(opt['dataset_root'], 'train_SIM_HSNR')
            opt['val_dataroot_raw_input'] = os.path.join(opt['dataset_root'], 'val_Raw_LSNR_1')
            opt['val_dataroot_raw_target'] = os.path.join(opt['dataset_root'], 'val_Raw_HSNR')
            opt['val_dataroot_sim_target'] = os.path.join(opt['dataset_root'], 'val_SIM_HSNR')
            if opt['model'] in ['rdl_SRn2n']:
                opt['train_dataroot_sim_input'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_1')
                opt['val_dataroot_sim_input'] = os.path.join(opt['dataset_root'], 'val_SIM_LSNR_1')
        else:
            raise NotImplementedError

    # Self-SUP
    elif opt['supervise'] in ["self-supervised"]:
        # Self-SUP + DN
        if opt['model'] in ["denoise"]:
            opt['train_dataroot_input'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_1')
            opt['train_dataroot_target'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_2')
            opt['val_dataroot_input'] = os.path.join(opt['dataset_root'], 'val_SIM_LSNR_1')
        # Self-SUP + REC
        elif opt['model'] in ["reconstruction"]:
            opt['train_dataroot_input'] = os.path.join(opt['dataset_root'], 'train_Raw_LSNR_1')
            opt['train_dataroot_target'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_2')
            opt['val_dataroot_input'] = os.path.join(opt['dataset_root'], 'val_Raw_LSNR_1')
        elif opt['model'] in ['rdl', 'pattern-only-rdl', 'circle-raw-reconstruction', 'circle-wf-reconstruction']:
            opt['train_dataroot_raw_input'] = os.path.join(opt['dataset_root'], 'train_Raw_LSNR_1')
            opt['train_dataroot_raw_target'] = os.path.join(opt['dataset_root'], 'train_Raw_LSNR_2')
            opt['train_dataroot_sim_target'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_2')
            opt['val_dataroot_raw_input'] = os.path.join(opt['dataset_root'], 'val_Raw_LSNR_1')
        else:
            raise NotImplementedError

    # Self-SUP with validation
    elif opt['supervise'] in ["self-supervised-val"]:
        # Self-SUP + DN
        if opt['model'] in ["denoise"]:
            opt['train_dataroot_input'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_1')
            opt['train_dataroot_target'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_2')
            opt['val_dataroot_input'] = os.path.join(opt['dataset_root'], 'val_SIM_LSNR_1')
            opt['val_dataroot_target'] = os.path.join(opt['dataset_root'], 'val_SIM_HSNR')
        # Self-SUP + REC
        elif opt['model'] in ["reconstruction"]:
            opt['train_dataroot_input'] = os.path.join(opt['dataset_root'], 'train_Raw_LSNR_1')
            opt['train_dataroot_target'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_2')
            opt['val_dataroot_input'] = os.path.join(opt['dataset_root'], 'val_Raw_LSNR_1')
            opt['val_dataroot_target'] = os.path.join(opt['dataset_root'], 'val_SIM_HSNR')
        elif opt['model'] in ['rdl', 'rdl_n2n', 'rdl_SRn2n', 'pattern-only-rdl', 'circle-raw-reconstruction', 'circle-wf-reconstruction']:
            opt['train_dataroot_raw_input'] = os.path.join(opt['dataset_root'], 'train_Raw_LSNR_1')
            opt['train_dataroot_raw_target'] = os.path.join(opt['dataset_root'], 'train_Raw_LSNR_2')
            opt['train_dataroot_sim_target'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_2')
            opt['val_dataroot_raw_input'] = os.path.join(opt['dataset_root'], 'val_Raw_LSNR_1')
            opt['val_dataroot_raw_target'] = os.path.join(opt['dataset_root'], 'val_Raw_HSNR')
            opt['val_dataroot_sim_target'] = os.path.join(opt['dataset_root'], 'val_SIM_HSNR')
            if opt['model'] in ['rdl_SRn2n']:
                opt['train_dataroot_sim_input'] = os.path.join(opt['dataset_root'], 'train_SIM_LSNR_1')
                opt['val_dataroot_sim_input'] = os.path.join(opt['dataset_root'], 'val_SIM_LSNR_1')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return opt

# --------------------------------------------
# process null in json / dict
# --------------------------------------------
def dict2nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict2nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict2nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
class NoneDict(dict):
    def __missing__(self, key):
        return None

# --------------------------------------------
# dict to string for logger by recursion
# --------------------------------------------
def dict2str(opt, indent_l=1):
    msg = ''
    for key, vaule in opt.items():
        if isinstance(vaule, dict):
            msg += ' ' * (indent_l * 2) + key + ':{\n'
            msg += dict2str(vaule, indent_l + 1)
            msg += ' ' * (indent_l * 2) + '}\n'
        else:
            msg += ' ' * (indent_l * 2) + key + ': ' + str(vaule) + '\n'
    return msg

# --------------------------------------------
# print dict structurally
# --------------------------------------------
def print_dict(opt):
    print(dict2str(opt))

# --------------------------------------------
# infer json path from name of imaged mrc
# --------------------------------------------
def infer_json_from_name(filename, imaging_device):
    json_name = imaging_device
    if '3d' in os.path.basename(filename).lower():
        json_name += '_3d'
    else:
        json_name += '_2d'
    if '405' in os.path.basename(filename):
        json_name += '_405.json'
    elif '488' in os.path.basename(filename):
        json_name += '_488.json'
    elif '560' in os.path.basename(filename) or '561' in os.path.basename(filename):
        json_name += '_560.json'
    elif '640' in os.path.basename(filename) or '642' in os.path.basename(filename) or '647' in os.path.basename(filename):
        json_name += '_642.json'
    return json_name


# --------------------------------------------
# save json
# --------------------------------------------
def save_json(opt, dump_path):
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)