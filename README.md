**IMPORTANT**

Recommended Python `3.11.9` (no higher given TensorFlow [installation instructions](https://www.tensorflow.org/install)).

If running on Windows, use WSL 2 as TensorFlow version used is higher than 2.10 - the maximum supporting
GPU support on Windows.

**Running on WSL2**

Following [this guide](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-march-2023.ipynb)
by Jeff Heaton, the Anaconda environment can be set up to detect GPU as follows:

```sh
pip install cudatoolkit, cudnn
```

```sh
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

**XLA**

XLA is uesd for this project, which requires CUDA.

You may need to set the XLA flag to point to CUDA before running this program. An example of how to achieve this is as follows:

```sh
set XLA_FLAGS=--xla_gpu_cuda_data_dir="/usr/local/cuda-12.6/"
```
Based on solution by [user14653986](https://stackoverflow.com/a/64872998).

**ADDITIONAL TENSORFLOW ERROR CONSIDERATION**
Potential required activation script (assuming use of Anaconda) if CuDNN path not found.
Solution by Leigh (Gwyki) and Shayan Shahrokhi (sh-shahrokhi) from:
https://github.com/tensorflow/tensorflow/issues/63362#issuecomment-1988827544

```sh
NVIDIA_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")))
for dir in $NVIDIA_DIR/*; do
    if [ -d "$dir/lib" ]; then
        export LD_LIBRARY_PATH="$dir/lib:$LD_LIBRARY_PATH"
    fi
done
```

**Useful Commands**

Run from root of project with minimal options for debugging (requires generated embeddings):

```sh
python -m lib --no-run-sqlmap --from-cache --no-double-requests
```

Enable profiling with the `--profile` flag. For example:

```sh
python lib/main.py --profile
```

Open TensorBoard for profiling from root of project:

```sh
tensorboard --logdir tensorboard_log
```

Make sure to open the TensorBoard `localhost` URL through Google Chrome. [Known issue](https://github.com/tensorflow/tensorboard/issues/2874).

Run reinforcement learning environment tests:

```sh
python -m unittest test/environment.py
```

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