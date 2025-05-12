# torchrun_demos

Demos for torch DDP

Install with (python3.10+ recommended)

```shell
pip install -r requirements.txt
```

Run a standard elastic task (8 GPUs, 1 node, no crash ratio)

```shell
torchrun --nproc_per_node=8 ddp_demo_elastic.py --workdir runs/elastic_demo_1 --fuck-ratio 0.0
```

Run an elastic task with 1% ratio of crash in any processed at the end of epochs (8 GPUs, 1 node)

```shell
torchrun --nproc_per_node=8 --max-restarts=1000 ddp_demo_elastic.py --workdir runs/elastic_demo_1x
```

Run a standard elastic task (2 nodes, 2 GPUs/node, no crash ratio)

```shell
torchrun \
  --nnodes=${MLP_WORKER_NUM} \
  --node_rank=${MLP_ROLE_INDEX} \
  --nproc_per_node=${MLP_WORKER_GPU} \
  --master_addr=${MLP_WORKER_0_PRIMARY_HOST} \
  --master_port=${MLP_WORKER_0_PORT} \
  --max_restarts=1000 \
  --rdzv_id=${MLP_TASK_ID} \
  ddp_demo_elastic.py \
  --workdir runs/elastic_demo_2nodes --fuck-ratio 0.0
```

Run an elastic task with 1% ratio of crash in any processed at the end of epochs (2 nodes, 2 GPUs/node, no crash ratio)

```shell
torchrun \
  --nnodes=1:${MLP_WORKER_NUM} \
  --node_rank=${MLP_ROLE_INDEX} \
  --nproc_per_node=${MLP_WORKER_GPU} \
  --master_addr=${MLP_WORKER_0_PRIMARY_HOST} \
  --master_port=${MLP_WORKER_0_PORT} \
  --max_restarts=1000 \
  --rdzv_id=${MLP_TASK_ID} \
  ddp_demo_elastic.py \
  --workdir runs/elastic_demo_2nodes_x
```
