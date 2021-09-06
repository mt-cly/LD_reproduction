# set up
1. Install the required packages including tensorflow-gpu, opencv, scipy, numpy etc.
2. As descripted  in LD github. Download the pre-trained model from this link (https://drive.google.com/open?id=1u96zu0VyNpy-1VL90EbriN74hGaBBK08). Unzip it and put them into the "Models" folder. 
3. modify the path to dataset in 'evaluate.py#91'.
4. run the 'evaluate.py' 

# tips for the baseline
* the main code is in 'IntSeg_GUI.py' without function to generate nextclick. 
* according to the paper, the 'genIntSegPairs.m' is used to generate nextclick in training phase instead of test phase, so it can be removed.
* In 'IntSeg_GUI.py', the generated segmentation and clicked information will be saved in 'res' folder.

# tips for the reproduction
* the self-defined dataloader is in the 'cly_instance_property.py'.
* The 'IntSeg_GUI.py' is not suitable for evaluation. so I realize the evaluation code in 'evaluate.py'.
* In 'evaluate.py', the function 'get_next_anno_point' will return the center of error region (like fca,fbrs), while the function 'get_next_anno_point_prob' will return sampled nextclick following the description of LD paper.
* In 'our_func_cvpr18.py', the function 'our_func' is not useful for my evaluation, so I revise it in 'cly_our_func'. This function is to predict segmentation base on the input.

tips for results:
* SBD result exists gap with paper report(mine:SBD@0.85=8.8,  paper :SBD@0.85=7.41)
* the result of COCO_Mal: (mine:COCO@0.85=7.1; paper :COCO@0.85=7.86) this gap may be caused by different COCO_MVal datasets, LD sample 800 images by themself.
