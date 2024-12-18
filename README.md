# RecTest Library: Data Schema and Benchmarking Guide

## Overview
The RecTest library allows you to benchmark your datasets for recommendation systems. This guide provides the necessary data schema and steps to get started.

## Data Schema

### Initial Dataset
Your dataset should be in JSONL format, similar to the Amazon Reviews 2018 & 2023 versions. Each line should represent an interaction between a user and an item.

#### Required Fields:
- `user_id` (str): ID of the user interacting with the product.
- `item_id` (str): ID of the product.
- `timestamp` (int): Time of the interaction in Unix time.

#### Optional Fields:
- `review_text` (str): Review left by the user.
- `score` (bool|float): Rating/score of the item left by the user, binary or float.

*Note:* The names of the input columns can be set in the YAML configuration file.

### Metadata-Items
You can optionally provide additional metadata for items.

#### Required Field:
- `item_id`: ID of the product.

#### Optional Fields:
- `feature_1`
- `feature_2`
- `feature_3`
- ...

#### Example (based on [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)):
- `item_id`: ID of the product.
- `title`: Name of the product.
- `description`: Description of the product.
- `brand`: Brand name.
- `categories`: List of categories the product belongs to.

## Output Dataset
The provided data will be processed through a custom preprocessing step. The resulting dataset folder will have the following structure:

### Example Directory Structure
- `Name_of_Dataset`
  - `interactions`: Dask DataFrame
  - `train` (optional): Dask DataFrame
  - `val` (optional): Dask DataFrame
  - `test` (optional): Dask DataFrame
  - `metadata` (optional): Dask DataFrame
  - `encoder_items.json` (optional): Dictionary in the form `{item_id: item_id_encoded}`
  - `encoder_users.json` (optional): Dictionary in the form `{user_id: user_id_encoded}`
  - `stats.json` (optional): Information about the number of unique users, items, and interactions.

## Benchmarking Your Dataset
To benchmark your dataset using the RecTest library, follow these steps:

1. **Prepare Your Data**: Ensure your data is in the required JSONL format with the necessary fields.
2. **Create a YAML Configuration File**: Define the column names and other settings in a YAML file.
3. **Run the Preprocessing Script**: Use the provided script to preprocess your data.
4. **Train and Evaluate Models**: Use the training scripts to benchmark your dataset.

For detailed instructions and examples, refer to the [notebooks/create_dataset.ipynb](notebooks/create_dataset.ipynb).

