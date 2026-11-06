# Data sources

All datasets are fetched from OpenML with `sklearn.datasets.fetch_openml` and cached under `data/raw/openml/`. Retrieval date is the date `run_all.py` first populated the cache.

## electricity (OpenML data_id 151, version 1)

- URL: https://www.openml.org/d/151
- Original name: electricity
- License (OpenML `licence` field): Public
- Retrieved: 2026-11-04
- Features used (numeric): nswprice, nswdemand, vicprice, vicdemand, transfer, period
- Features used (categorical): day
- Cached files (sha256):
  - `raw/openml/openml.org/api/v1/json/data/151.gz`: 53284206e86e9a5f441e7fe9509f3223cf431e6a4ddc84c280883516d0be6fae
  - `raw/openml/openml.org/api/v1/json/data/features/151.gz`: 1a1a4133830b7cd488e3ec03c0be1b7c420bf2b7f9084b0bc739994537559c84
  - `raw/openml/openml.org/api/v1/json/data/qualities/151.gz`: 962b7035621144988f6f9805a6e6b7578d6b2f922c4122ff8b74484b8d3e51d4
  - `raw/openml/openml.org/data/v1/download/2419/electricity.arff.gz`: 60df61719ff2065ae144ae7788a2a1fca603e6c5bc83c5bd82bb3b63b0d28c74

## adult (OpenML data_id 1590, version 2)

- URL: https://www.openml.org/d/1590
- Original name: adult
- License (OpenML `licence` field): Public
- Retrieved: 2026-11-04
- Features used (numeric): age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- Features used (categorical): workclass, occupation, relationship, race
- Cached files (sha256):
  - `raw/openml/openml.org/api/v1/json/data/1590.gz`: ad0dfaa0ac02096f602b961d84b47d537e1ad79debb005a6eb18c4b622354a8c
  - `raw/openml/openml.org/api/v1/json/data/features/1590.gz`: 5f0dfb39521ba4167125bd83ee66a678f0cb90f5ed21fbf26b112c883a133663
  - `raw/openml/openml.org/api/v1/json/data/qualities/1590.gz`: b4db125ecb6aab8ba788d1b159eec90ce9adc620c468c1ecb3b0b564132ed115
  - `raw/openml/openml.org/data/v1/download/1595261/adult.arff.gz`: bd3946bc53d0fd3274f5bf6a0ffb014bbb4cde99243affc196734fc99a3bb0e5

## bank-marketing (OpenML data_id 1461, version 1)

- URL: https://www.openml.org/d/1461
- Original name: bank-marketing
- License (OpenML `licence` field): Public
- Retrieved: 2026-11-04
- Features used (numeric): V1, V6, V12, V13, V14, V15
- Features used (categorical): V2, V3, V11, V16
- Column meaning: UCI column names: V1 age, V6 balance, V12 duration, V13 campaign, V14 pdays, V15 previous; V2 job, V3 marital, V11 month, V16 poutcome
- Cached files (sha256):
  - `raw/openml/openml.org/api/v1/json/data/1461.gz`: ffc845c6205b991c64bc3f6ae71f70f580a582e15f2d69100cf543c4f6493c60
  - `raw/openml/openml.org/api/v1/json/data/features/1461.gz`: 4a2d784a26c1967222eddb3ab996083a1a1f939f7a96e529ce6717698be8043f
  - `raw/openml/openml.org/api/v1/json/data/qualities/1461.gz`: 5b1b7477efbbaf46abc0798c0c6aa60fda6ba785b2b50a27595ed7cbc033c98a
  - `raw/openml/openml.org/data/v1/download/1586218/bank-marketing.arff.gz`: 9337b4ba9cfdb99707a95f2af8508c102add2a12bde0b70f76187252d6091d2d
