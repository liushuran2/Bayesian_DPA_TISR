# import logging
# import warnings
# import datetime
# import numpy as np

# # --------------------------------------------
# # report software running logging
# # --------------------------------------------
#
# # --------------------------------------------
# # MAIN FUNCTION
# # --------------------------------------------
# def make_log(log_path='opt.log', need_time_stamp=None, print_to_termial=False):
#
#     logger_name = 'opt'
#
#     logger_info(logger_name, log_path=log_path, need_time_stamp=need_time_stamp, print_to_termial=print_to_termial)
#
#     logger_opt = logging.getLogger(logger_name, )
#
#     if need_time_stamp == 'once': logger_opt.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#
#     return logger_opt
#
#
# # --------------------------------------------
# # ASSISTED FUNCTION
# # --------------------------------------------
# def logger_info(logger_name, log_path, need_time_stamp='always', print_to_termial=False):
#
#     log = logging.getLogger(logger_name)
#
#     if log.hasHandlers():
#
#         warnings.warn('LogHandlers exist')
#
#     else:
#         level = logging.INFO
#
#         if need_time_stamp == 'always':
#             formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
#         else:
#             formatter = logging.Formatter('%(message)s', datefmt=None)
#
#         fh = logging.FileHandler(log_path, mode='a')
#         fh.setFormatter(formatter)
#         log.setLevel(level)
#         log.addHandler(fh)
#
#         if print_to_termial is True:
#             sh = logging.StreamHandler()
#             sh.setFormatter(formatter)
#             log.addHandler(sh)