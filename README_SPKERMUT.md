# SP Kermut

This is a wrapper repository on top of [Kermut](https://github.com/petergroth/kermut) to predict protein fitness using a Gaussian Process. Predictions are enhanced by replacing the ESM2 module with the top-scoring VenusREM model embeddings and zero shot scores.

New scripts in this repository are the following:
  - `venusrem_fitness.py`
  - `venusrem_zero_shot.py`
  - `run_venusrem_embeddings.py`
  - `run_venusrem_zero_shot.py`
  - `configs/proteingym_venusrem.yaml`

---

## Commands to run
To reproduce the results of SP Kermut, the embeddings and zero shot scores need to be reproduced from VenusREM.
```
# Generate embeddings
python run_venusrem_embeddings.py

# Generate zero shot scores
python run_venusrem_zero_shot.py

# Run benchmark on proteingym
python run_venusrem_benchmark.py --config-name proteingym_venusrem
```

To reproduce the results from the original Kermut github, run:
```
python proteingym_benchmark.py --multirun \
    dataset=benchmark \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    kernel=kermut
```

Finally, the last script will calculate spearman coefficients accessing the `results/` directory:
```
python calculate_spearman.py
```

