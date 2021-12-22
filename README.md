# CSAW-HackML-2020

```bash
├── data 
    └── clean_valid.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test.h5
    └── sunglasses_poisoned_data.h5
    └── anon_poison.h5
    └── eyebrows_poison.h5
    └── lipstick_poison.h5
    └── sun_normal.h5
    └── sun_multi.h5
               
├── models
    └── sunglasses_bd_net.h5
    └── sunglasses_repaired.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_eyebrows_repaired.h5
    └── multi_trigger_multi_target_lipstick_repaired.h5
    └── multi_trigger_multi_target_sunglass_repaired.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_repaired.h5
    └── anonymous_1_bd_weights.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_repaired.h5
    └── anonymous_2_bd_weights.h5
├── architecture.py
└── eval.py // this is the evaluation script
└── custom_eval.py  // script to predict labels of input images. (outputs 1283 if the data is poisoned)
```
   
## I. Validation Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5. Similarly, there are other .h5 files with poisoned data that correspond to different BadNets under models directory.

## II. Evaluating the Backdoored Model
   1. The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network. This DNN is backdoored with multiple triggers. Each trigger is associated with its own target label. 
   2. To evaluate the backdoored model, execute `eval.py` by running:  
      `python3 eval.py <clean validation data directory> <model directory>`.
      
      E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.

## IV. Evaluating repaired models on new test images
   To evaluate the any repaired model, execute `custom_eval.py` by running:  
      `python3 custom_eval.py <clean validation data directory/test image path> <good model directory> <bad model directory>`.
   Will output the correct model for clean data and 1283 for poisoned data.
      

