**Useful Commands**

Run from root of project with minimal options for debugging (requires generated embeddings):

```sh
python lib\main.py --no-run-sqlmap --no-demonstrations --from-cache --no-double-requests
```

Open TensorBoard for profiling from root of project:

```sh
tensorboard --logdir tensorboard_log
```

**SQL List**

This list contains a number of entries which the agent can use to build queries which which to view data with. 

This list does *NOT* contain entries which allows the agent to manipulate the target database.

Many word entries by [Charlie Custer](https://www.dataquest.io/blog/sql-commands/).

SQL injection list (SQLiV3.csv) by [SYED SAQLAIN HUSSAIN SHAH](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset).

Reserved SQL keywords by [W3 Schools](https://www.w3schools.com/sql/sql_ref_keywords.asp).

Reserved SQL operators by [W3 Schools](https://www.w3schools.com/sql/sql_operators.asp).

10GB Test DB for test site from [StackOverflow](https://www.brentozar.com/archive/2015/10/how-to-download-the-stack-overflow-database-via-bittorrent/)