# UFNRec

## Reference Code：
https://github.com/pmixer/SASRec.pytorch
https://github.com/fadel/pytorch_ema/tree/master/torch_ema

## Commands：
To train our model on the default Beauty data with default parameters:
```
python main.py --name=UFNRec_Beauty
```

## Options:
The training of the UFNRec model is handled by the main.py script that provides the following command line arguments.
```
--dataset            STR           Name of dataset.               Default is "Beauty".
--name               STR           Train directory.               Required.
--batch_size         INT           Batch size.                    Default is 128.    
--lr                 FLOAT         Learning rate.                 Default is 0.001.
--maxlen             INT           Maxmum length of sequence.     Default is 50.
--hidden_units       INT           Number of hidden units.        Default is 50.
--num_blocks         INT           Number of blocks.              Default is 2.
--num_epochs         INT           Number of epochs to run.       Default is 201.
--num_heads          INT           Number of heads.               Default is 1.
--dropout_rate       FLOAT         Dropout rate value.            Default is 0.5.
--device             STR           Device for training.           Default is 'cuda'.
--l2_emb             FLOAT         L2 regularization value.       Default is 0.0.
--gpu                STR           Name of GPU to use.            Default is "0".
--reverse            INT           m in the paper.                Default is 5.
--lbd                FLOAT         alpha in the paper.            Default is 0.3.
--decay              FLOAT         d in the paper.                Default is 0.999.
--neg_nums           INT           number of negative samples.    Default is 100.
```
