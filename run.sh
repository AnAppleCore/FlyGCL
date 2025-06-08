# argument 1: gpu id
# argument 2: seed
# argument 3: dataset

screen -dmS fly1 bash ./scripts/run_baselines_fly.sh 1 1 cifar100

screen -dmS fly2 bash ./scripts/run_baselines_fly.sh 2 2 cifar100

screen -dmS fly3 bash ./scripts/run_baselines_fly.sh 3 3 cifar100

screen -dmS fly4 bash ./scripts/run_baselines_fly.sh 4 4 cifar100

screen -dmS fly5 bash ./scripts/run_baselines_fly.sh 5 5 cifar100


screen -dmS l2p1 bash ./scripts/run_baselines_l2p.sh 1 1 imagenet-r

screen -dmS l2p2 bash ./scripts/run_baselines_l2p.sh 2 2 imagenet-r

screen -dmS l2p3 bash ./scripts/run_baselines_l2p.sh 3 3 imagenet-r

screen -dmS l2p4 bash ./scripts/run_baselines_l2p.sh 4 4 imagenet-r

screen -dmS l2p5 bash ./scripts/run_baselines_l2p.sh 5 5 imagenet-r


screen -dmS l2p1 bash ./scripts/run_baselines_l2p.sh 1 1 cub200

screen -dmS l2p2 bash ./scripts/run_baselines_l2p.sh 2 2 cub200

screen -dmS l2p3 bash ./scripts/run_baselines_l2p.sh 3 3 cub200

screen -dmS l2p4 bash ./scripts/run_baselines_l2p.sh 4 4 cub200

screen -dmS l2p5 bash ./scripts/run_baselines_l2p.sh 5 5 cub200