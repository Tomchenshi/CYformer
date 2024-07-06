# CYformer
The pytorch code of hyperspectral and multispectral image fusion

## Requirements
* Python 3.6.13
* Pytorch 1.10.0

## Training
python main.py train --batch_size 32 --n_scale 4 --dataset_name "Cave"

## Testing
python main.py test --n_scale 4 --dataset_name "Cave"
