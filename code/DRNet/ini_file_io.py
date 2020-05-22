#!/usr/bin/python
import ConfigParser

def load_train_ini(ini_file):
    # initialize
    cf = ConfigParser.ConfigParser()
    cf.read(ini_file)
    # dictionary list
    param_sections = []

    s = cf.sections()
    for d in range(len(s)):
        # create dictionary
        level_dict = dict(phase         = cf.get(s[d], "phase"),
                          batch_size    = cf.getint(s[d], "batch_size"),
                          inputI_width_size   = cf.getfloat(s[d], "inputI_width_size"),
                          inputI_height_size   = cf.getfloat(s[d], "inputI_height_size"),
                          inputI_chn    = cf.getint(s[d], "inputI_chn"),
                          output_chn    = cf.getint(s[d], "output_chn"),
                          trainImagePath = cf.get(s[d], "trainImagePath"),
                          trainDmapPath = cf.get(s[d], "trainDmapPath"),
                          testImagePath = cf.get(s[d], "testImagePath"),
                          testDmapPath = cf.get(s[d], "testDmapPath"),
                          chkpoint_dir  = cf.get(s[d], "chkpoint_dir"),
                          learning_rate = cf.getfloat(s[d], "learning_rate"),
                          beta1         = cf.getfloat(s[d], "beta1"),
                          epoch         = cf.getint(s[d], "epoch"),
                          model_name    = cf.get(s[d], "model_name"),
                          save_intval   = cf.getint(s[d], "save_intval"),
                          labeling_dir  = cf.get(s[d], "labeling_dir"),
                          load_model_path= cf.get(s[d], "load_model_path"))
        # add to list
        param_sections.append(level_dict)

    return param_sections