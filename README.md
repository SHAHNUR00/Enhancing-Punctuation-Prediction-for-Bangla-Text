# Punctuation Predict & Restore

Punctuation-Prediction-in-Bangla-language-using-Transformer-Models.

## Data

In order to train, store 'train.txt', 'test.txt', 'val.txt' in  `data/bn_inhouse/` directory.


## Dependencies
In order to install required dependencies, do the following commands. 
(Please, do it inside a `conda` or `venv` environment, otherwise all the packages will be installed globally.)

```bash
pip install -r requirements.txt
```

## Training

To train for Bangla the corresponding command is

```
! python src/train.py --cuda=True --pretrained-model=bert-base-multilingual-cased --freeze-bert=False --lstm-dim=-1 --language=bangla --seed=1 --lr=5e-6 --epoch=3 --use-crf=False --augment-type=none --data-path=data --save-path=out --batch-size=64
```

You can also run the `src/run.sh` instead of the above command.
```
sh src/run.sh
```

#### Supported params:

```
--pretrained-model=
    bert-base-multilingual-cased /
    bert-base-multilingual-uncased /
    xlm-mlm-100-1280 /
    distilbert-base-multilingual-cased
    xlm-roberta-base /
    xlm-roberta-large 

--cuda=
    True, if gpu available else False

--augment-type=
    none: augment_none /
    substitute: augment_substitute /
    insert: augment_insert /
    delete: augment_delete /
    all: augment_all /
```

## Trained Models

Trained model can be found [here](https://drive.google.com/drive/folders/1w8Qsj1dyjMOiaQacS7byhl-YzOO9wn-4?usp=sharing).

Put it in the `out/` directory in order to infer.

## Inference

To infer on data (text without punctuation), put text in `data/test_bn.txt` file. Then run the below script. (Make sure that `out/` directory contains the trained model (`weights.pt`) already.)

Example script:

```bash
! python src/inference.py --pretrained-model=bert-base-multilingual-cased --weight-path=out/weights.pt --language=bn  --in-file=data/test_bn.txt --out-file=data/test_bn_out.txt
```

Please provide same `pretrained-model` argument that were used during training.

Infered result (text with punctuation) can be found in `data/test_bn_out.txt` file.

