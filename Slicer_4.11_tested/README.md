# COVID19 CT Segmentation in 3DSlicer

An extension module in 3DSlicer for COVID19 CT segmentation using Convolutional Neural Networks.

We trained a 2D ConvNet to segment lung and possible infected regions from patient CT scans. We then incorporated this ConvNet segmentation model into 3DSlicer.

This module requires `Keras/Tensorflow`, `scikit-image`, `scipy`, and `numpy` (all installed under 3DSlicer).

Location of the model can be changed in `COVID19_CT_SEGMENTATION/ConvNetCovid/ConvNetCovid.py`:

    line 55: segmentation_model = load_model('xxx_Model.h5')
    
## Example segmentation by our ConvNet model:
![](https://github.com/junyuchen245/COVID19_CT_Segmentation_3DSlicer/blob/master/pics/Screen%20Shot%202020-05-03%20at%2011.33.15%20PM.png)

## Screenshot of the extension module in 3DSlicer:
![](https://github.com/junyuchen245/COVID19_CT_Segmentation_3DSlicer/blob/master/pics/Screen%20Shot%202020-05-04%20at%205.04.11%20PM.png)

### <a href="https://junyuchen245.github.io"> About Myself</a>
