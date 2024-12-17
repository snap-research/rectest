# Data Schema
Currently files provided should be in jsonl format (similar to the Amazon Reviews 2018 & 2023 versions). Each line should represent an interaction of item/user.

## Initial Dataset
All the data must be provided in the following format.
- user_id (str): id of the user interacting with the product
- item_id (str): id of the product
- timestamp (int): time of the interaction in Unix time
- review_text (optional, str): review left by the user
- score (optional, bool|float): rating/score of item left by the user, binary or float

*Note:* The name of the input columns can be set on the yaml file.

Additional metadata for items and users can optionally be provided.

## Metadata-Items
- item_id: id of the product
- feature_1
- feature_2
- feature_3
- ...

*Metadata-Item* example based on the [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/):
- item_id: id of the product
- title: name of the product
- description: description of the product
- brand: brand name
- categories: list of categories the product belongs to

## Output dataset
The data provided will be passed through a custom (based on each dataset) preprocess step. The resulting dataset folder will be in the format of:

### Example directory structure
- Name_of_Dataset
    - interactions: dask.dataframe
    - train (optional) dask.dataframe
    - val (optional) dask.dataframe
    - test (optional) dask.dataframe
    - metadata (optional) dask.dataframe:
    - encoder_items.json (optional): dictionary in the form {item_id: item_id_encoded}
    - encoder_users.json (optional): dictionary in the form {user_id: user_id_encoded}
    - stats.json (optional)

- interactions
    - user_id str: User ID as in the input dataset.
    - user_id_encoded (optional) int: Numeric user ID assinged during preprocessing.
    - item_id str: Item ID as in the input dataset. This item is the "target" of the sequence.
    - item_id_list list[str]: ordered list of the original item ids ordered chronologically from older to most recent.
    - item_id_encoded (optional) int: Numeric item ID assinged during preprocessing.
    - item_id_encoded_list (optional) list[int]: ordered list of encoded (numeric) item ids ordered chronologically from older to most recent.
    - features (optional) str: item metadata concatenated. as provided in *Metadata-Item*
    - item_features_list (optional) list[str]: items metadata as provided in *Metadata-Item* ordered chronologically from older to most recent.
    - item_score (optional): score/rating assigned to target item
    - item_score_list (optional): list of scores/ratings assinged to each item ordered chronologically from older to most recent.
    - len_seq int: length of interaction

- train: same schema as `interactions`
- val:  same schema as `interactions`
- test: same schema as `interactions`

- metatdata
    - item_id str: Item ID as in the input dataset
    - features (optional) str: item metadata concatenated. as provided in *Metadata-Item*s

- encoder_items.json: dict object in the form {user_id: user_id_encoded}
- encoder_users.json: dict object in the form {item_id: item_id_encoded}
- stats.json: Information about the number of unique users (nusers), number of total unique items (nitems) and total number of interactions (ninteractions) in the form: {"num_users": 13, "num_items": 3, "num_interactions": 390}


## Guide on how to create a dataset: [notebooks/create_dataset.ipynb](notebooks/create_dataset.ipynb).

# Training pipeline
The user can utilise the existing training script `baselines/train_ce.py` to run a specific model on the selected dataset.

To run the script use:
`accelerate launch baselines/train_ce.py experiment=sasrec.yaml`

Where `sasrec.yaml` is the config file of the experiment and should be placed under the `configs_hydra/experiment` directory (example: [sasrec.yaml](configs_hydra/experiment/sasrec.yaml)).

We utilise hydra to insantiate objects so the user should use the `_target_` parameter to point to the object's Class and pass any necessary parameters.

## The yaml file should include the following arguments:

`model`: (hydra instantiate) Model to use.

`optimizer`: (hydra instantiate) Optimizer to use.

`collator_train`: (hydra instantiate) Collator to use for training.

`collator_val`: (hydra instantiate) Collator to use for evaluation.

*NOTE:* we utilise two different collators for cases where different number of candidates is used for training/evaluation.

`dataset`: (hydra instantiate) Dataset class to use.

`seed`: (int) Random seed to use for reproducability.

`epochs`: (int) Number of training epocchs.

`batch_size_train`: (int) Size of training batch.

`batch_size_val`: (int) Size of validation/testing batch.

`early_stopping`: (int) number of steps without improvement before triggering early stopping.

`gradient_accumulation_steps`: (int) Number of step to accumulate gradients.

`accelerate_checkpoint`: (int) Path to accelerate checkpoint to load model/continue training.

`eval_at_start`: (bool) Whether to evaluate the model before training or not. `true` or `false`.

`eval_steps`: (int) How ofter (in steps) to evaluate the model.

`metric_ks`: (int) At wich level should we calculate the metrics (e.g. top 10, top 20, top 30)

`save_metric`: (str) which metric to consider when selecting the best model. Available metrics 'NDCG', 'MRR', 'RECALL'

`device`: (str) device to use for training. Options: `cuda` or `cpu`.

`number_of_devices`: (int) Number of GPUs to use.

`fp16`: (bool) Options `true` or `false`.

`output_dir`: (str) Path to save model/checkpoints/results.

# Existing resources and configs
## Models
- [SASRec](https://arxiv.org/abs/1808.09781)
```
model:
  _target_: recommender.rec_model.SASRec
  hidden_size: 64
  item_num: null # updated in code
  state_size: 10 # Max Sequence Length
  dropout: 0.1
  num_heads: 1
  no_id: false
  device: 'cuda'
  extra_embeddings: null
```

- **hidden_size** (int): The size of the hidden layers.
- **item_num** (int): The number of unique items.
- **state_size** (int): The size of the state (max sequence length).
- **dropout** (float): Dropout rate for regularization.
- **device** (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
- **num_heads** (int, optional): The number of attention heads. Default is 1.
- **extra_embeddings** (str, optional): Path to extra embeddings file. Default is None.
- **no_id** (bool, optional): If True, do not use item IDs in the input. Default is False.

- [GRU4Rec](https://arxiv.org/pdf/1606.08117)
```
  _target_: recommender.rec_model.GRU
  hidden_size: 64
  item_num: null # updated in code
  state_size: 10 # Max Sequence Length
  gru_layers: 1
  extra_embeddings: null
  no_id: false
```

- **hidden_size** (int): The number of features in the hidden state.
- **item_num** (int): The number of unique items in the dataset. dynamically set during runtime.
- **state_size** (int): The size of the state. (should be equal to the max sequence length)
- **gru_layers** (int, optional): Number of recurrent layers. Default is 1.
- **extra_embeddings** (str, optional): Path to extra embeddings file. Default is None.
- **no_id** (bool, optional): If True, do not use item IDs in the input. Default is False.

- [Caser](https://arxiv.org/pdf/1809.07426)
Sample config:
```
model:
  _target_: recommender.rec_model.Caser
  hidden_size: 64
  item_num: null # updated in code
  state_size: 10 # Max Sequence Length
  num_filters: 16
  filter_sizes: [2, 3, 4]
  dropout: 0.1
  no_id: false
  extra_embeddings: null
```

- **hidden_size**: The size of the hidden layer.
- **item_num**: Total number of unique items; dynamically set during runtime.
- **state_size**:  The size of the state. (should be equal to the max sequence length)
- **num_filters**: The number of filters for the convolutional layers.
- **filter_sizes**: The sizes of the filters for the convolutional layers.
- **dropout**: The dropout rate.
- **no_id**: Flag to indicate if ID should be used. Defaults to False.
- **extra_embeddings**: Path to extra embeddings file. Defaults to None.


- [Recformer](https://arxiv.org/pdf/2305.13731)
```
model:
  _target_: recommender.rec_model.RecformerForSeqRec
  pretrain_ckpt: longformer_ckpt/custom_longformer-base-4096.bin
  item_num: null # updated in code
  attention_window: "[64] * 12"
  model_name: allenai/longformer-base-4096
  max_attr_num: 3
  max_attr_length: 32
  max_item_embeddings: 51
  max_token_num: 1024
  extra_embeddings: null
```

- **model_name** : The name of the huggingface backbone
- **max_attr_num**: The maximum number of attributes (e.g. title, brand, category)
- **max_attr_length**: The maximum length of attributes.
- **max_item_embeddings** (int): Size of item embeddings
- **attention_window** (str): The attention window size.
- **max_token_num** (int): The maximum number of tokens.
- **item_num** (int): The number of items. # dynamically updated on runtime
- **pretrain_ckpt^** (str): Path to the pretrained checkpoint.
- **extra_embeddings** (Optional[str], optional): Path to extra embeddings. Defaults to None

***^NOTE:***The pretrained checkpoint should be the output of `recommender/RecFormer/convert_pretrain_ckpt.py` script.

## Datasets
Currently we implement two datasets classes `SequentialIterableDataset` and `RecformerDataset`.

Both dataset classes expect a path argument indicating the path to the output of the data pipeline. `RecformerDataset` is used specifically for the recformer model as it includes text features.
`SequentialIterableDataset` can be used for SASRec, GRU, Caser and any other ID based models.


## Collators
Every dataset implemented should be accompanied with the appropriate collator. Currently we implement `RecFormerCollateFn` and `IDOnlyCollateFn`.

`RecFormerCollateFn` is used for the `Recformer` and `RecformerDataset` tokenizes on the fly the text features used and produces candidates base on the items present in the batch.

`IDOnlyCollateFn` can be used for any models/datasets that are only IDs to creatte candidates base on the items present in the batch.


# Adding new resources
Add new models under `recommender/rec_model.py`.

Add new datasets under `baselines/dataloading.py`.

Add new collators under `baselines/collators.py`.
