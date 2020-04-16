import torch
import sys

import datasets.factory as factory
import LOI

from scipy import misc, io

import PIL.Image as Image

print("--- LOADING TESTSET ---")
train_pairs, test_pairs = factory.load_train_test_frame_pairs('')
print("- Loaded {} pairs".format(len(test_pairs)))
print("--- DONE ---")

# Load counting model
cc_model = LOI.init_cc_model()

# Load flow estimation model
fe_model = LOI.init_fe_model()

for i,pair in enumerate(train_pairs[200:400]):

    print("{}/200".format(i))
    # Run counting model on frame A
    cc_output = LOI.run_cc_model(cc_model, pair.get_frames()[0])

    # Run flow estimation model on frame pair
    fe_output = LOI.run_fe_model(fe_model, pair)

    # Run the merger for the crowdcounting and flow estimation model
    # In code mention the papers based on
    #loi_output = LOI.pixelwise_loi(cc_output, fe_output)
    loi_output = LOI.regionwise_loi(cc_output, fe_output)

    # print(cc_output.shape)
    #
    # img = Image.fromarray(cc_output * 255.0 / cc_output.max())
    # img = img.convert("L")
    # img.save('results/csr.png')
    #
    # img = Image.open(pair.get_frames()[0].get_image_path())
    # img = img.convert("L")
    # img.save('results/original.png')
    #
    # misc.imsave('results/flow_est_.png', fe_output[1][0])