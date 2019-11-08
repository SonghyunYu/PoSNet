# PoSNet
Pytorch Implementation of "[PoSNet: 4x video frame interpolation using position-specific flow]()"

AIM 2019 Video Temporal Super-Resolution Challenge 

## Related Work
In our work, we used the code of [PWCNet](https://github.com/sniklaus/pytorch-pwc) and [SpyNet](https://github.com/sniklaus/pytorch-spynet). Thanks to [Simon Niklaus](https://github.com/sniklaus) for releasing the implementations.

## Environment  
  python 3.7   
  pytorch 1.0.0  

## Prepare data


## Test
Download Pre-trained models: [[download]](https://drive.google.com/folderview?id=18-39JPIN0w7rp7oewlQf8C0ur7oa4DxY) and place them in the './models' folder.  

Example:  
```
python main_middle.py --cuda --eval  # to interpolate middle frame
python main_side.py --cuda --eval  # to interpolate side frames
```


## Training
To train your own model, you should prepare dataset using prepare_data.py.

Example:  
```
python main_middle.py --cuda
```
  
## Anknowledgement


## Contact

