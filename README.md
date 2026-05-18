usage: main.py [-h\] [--phase {1,2,all}\] [--runs RUNS\] [--seed SEED\] [--workers WORKERS\]
               [--algorithms {Algsp,SPONGE,CAbABC,SN-MOGA,KST}\]

signed community detection experiment.

options:
  -h, --help            show this help message and exit
  --phase {1,2,all}
  --runs RUNS           Repetitions for stochastic algorithms (default: 30)
  --seed SEED
  --workers WORKERS     Parallel worker processes (default: os.cpu_count())
  --algorithms {Algsp,SPONGE,CAbABC,SN-MOGA,KST
