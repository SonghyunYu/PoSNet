# PoSNet
Pytorch Implementation of Songhyun Yu's algorithm
AIM 2019 Video Temporal Super-Resolution Challenge 

## Related Work


## Environment  
  python 3.7   
  pytorch 1.0.0  

## Prepare data


## Test
Download Pre-trained models: [[download]](https://drive.google.com/open?id=18-39JPIN0w7rp7oewlQf8C0ur7oa4DxY)  
and place them in the './models' folder.  
for middle frame, use 'model_middle.pth.tar'
for side frame, use 'model_side_pth.tar'

Example:  
```
python main_middle.py --cuda --eval
```

## Training
To train your own model, you should prepare dataset.

Example:  
```
python main_middle.py --cuda
```
  
## Anknowledgement


## Contact
If you have any questions about the code or paper, please contact fkdlzmtld@gmail.com
