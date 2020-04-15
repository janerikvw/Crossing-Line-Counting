import torch
import sys

import datasets.factory as factory
import LOI

print("--- LOADING TESTSET ---")
train_pairs, test_pairs = factory.load_train_test_frame_pairs('')
print("- Loaded {} pairs, train {}".format(len(test_pairs), len(train_pairs)))
print("--- DONE ---")

# Load counting model
cc_model = LOI.init_cc_model()

# Load flow estimation model
fe_model = LOI.init_fe_model()

for pair in test_pairs:
    print(pair)

    # Run counting model on frame A
    cc_output = LOI.run_cc_model(cc_model, pair.get_frames()[0])

    # Run flow estimation model on frame pair
    fe_output = LOI.run_fe_model(fe_model, pair)

    # Run the merger for the crowdcounting and flow estimation model
    loi_output = LOI.pixelwise_loi(cc_output, fe_output)

    print(loi_output)

    break