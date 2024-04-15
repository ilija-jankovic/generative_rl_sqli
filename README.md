**IMPORTANT**

Use Python `3.7.4`.

To do with TensorFlow but reason forgotten. `TODO:` experiment with newer Python versions.

**XLA**

XLA is uesd for this project, which requires CUDA.

You may need to set the XLA flag to point to CUDA before running this program. An example of how to achieve this is as follows:

```sh
set XLA_FLAGS=--xla_gpu_cuda_data_dir="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
```

Based on solution by [user14653986](https://stackoverflow.com/a/64872998).

**Useful Commands**

Run from root of project with minimal options for debugging (requires generated embeddings):

```sh
python lib\main.py --no-run-sqlmap --no-demonstrations --from-cache --no-double-requests
```

Enable profiling with the `--profile` flag. For example:

```sh
python lib\main.py --profile
```

Open TensorBoard for profiling from root of project:

```sh
tensorboard --logdir tensorboard_log
```

Make sure to open the TensorBoard `localhost` URL through Google Chrome. [Known issue](https://github.com/tensorflow/tensorboard/issues/2874).

**SQL List**

This list contains a number of entries which the agent can use to build queries which which to view data with. 

This list does *NOT* contain entries which allows the agent to manipulate the target database.

Many word entries by [Charlie Custer](https://www.dataquest.io/blog/sql-commands/).

SQL injection list (SQLiV3.csv) by [SYED SAQLAIN HUSSAIN SHAH](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset).

Reserved SQL keywords by [W3 Schools](https://www.w3schools.com/sql/sql_ref_keywords.asp).

Reserved SQL operators by [W3 Schools](https://www.w3schools.com/sql/sql_operators.asp).

10GB Test DB for test site from [StackOverflow](https://www.brentozar.com/archive/2015/10/how-to-download-the-stack-overflow-database-via-bittorrent/)

**Extra Information**

Using old version of 2.8.0 of `tensorboard-plugin-profile` due to error outlined by Anatolist in [this issue](https://github.com/tensorflow/profiler/issues/12).

Solution by [`GoldenGoldy`](https://github.com/tensorflow/profiler/issues/12#issuecomment-1806793838).