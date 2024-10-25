**IMPORTANT**

Recommended Python `3.11.9` (no higher given TensorFlow [installation instructions](https://www.tensorflow.org/install)).

If running on Windows use WSL 2 for latest TensorFlow version.

**Useful Commands**

Run from root of project with minimal options for debugging (requires generated embeddings):

```sh
python -m lib --no-run-sqlmap --from-cache
```

Open TensorBoard for profiling from root of project:

```sh
tensorboard --logdir tensorboard_log
```

Make sure to open the TensorBoard `localhost` URL through Google Chrome. [Known issue](https://github.com/tensorflow/tensorboard/issues/2874).

Run tests:

```sh
python -m unittest test/[FILENAME].py
```

**SQL List**

[WikiSQL](https://github.com/salesforce/WikiSQL).

SQL injection list (SQLiV3.csv) by [SYED SAQLAIN HUSSAIN SHAH](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset).

Reserved SQL keywords by [W3 Schools](https://www.w3schools.com/sql/sql_ref_keywords.asp).

Reserved SQL operators by [W3 Schools](https://www.w3schools.com/sql/sql_operators.asp).
