<h2 align="center">CoralSMIA: Physically-Grounded Scattering Modeling and
Illumination Adaptation for Generalized Underwater Coral
Segmentation</h2>


---------------------

## Environments:
- pytorch >= 1.11.0
- python >=3.6

```bash
pip install --upgrade git+https://github.com/Z-Zheng/ever.git
pip install git+https://github.com/qubvel/segmentation_models.pytorch
pip install mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```
The Swin-Transformer pretrained weights can be prepared following [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin).

### Train model
```bash 
bash ./scripts/train_loveda.sh
```
