# PoSNet
Pytorch Implementation of Songhyun Yu's algorithm

AIM 2019 Video Temporal Super-Resolution Challenge 

## Related Work
We modified [PWCNet](https://github.com/sniklaus/pytorch-pwc) ans [SpyNet](https://github.com/sniklaus/pytorch-spynet) of [Simon Niklaus](https://github.com/sniklaus)'s implementation. Thanks for open the code.

## Environment  
  python 3.7   
  pytorch 1.0.0  

## Prepare data


## Test
Download Pre-trained models: [[download]](https://drive.google.com/open?id=18-39JPIN0w7rp7oewlQf8C0ur7oa4DxY)  
and place them in the './models' folder.  

Example:  
```
python main_middle.py --cuda --eval  # to interpolate middle frame
python main_side.py --cuda --eval  # to interpolate side frames
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
