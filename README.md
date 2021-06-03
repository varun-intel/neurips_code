# ECNet

We provide instructions to reproduce the experiments in the paper `inimizing Communication while MaximizingPerformance in Multi-Agent Reinforcement Learning`. These commands use the tarmac network architecture. To use maddpg, commnet, or tarmac-sigmoid, replace tarmac.json with the respective config file.

## Secret
Baseline:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/secret.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json`

IC3Net:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/secret.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/reinforce_global_gate.json --gin_binding='default_teams.final_comm_penalty=0.0'`

ECNet-REINFORCE:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/secret.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/reinforce_global_gate.json --gin_binding='default_teams.final_comm_penalty=0.1'`

ECNet-GS:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/secret.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/gumbel_global_gate.json --gin_binding='default_teams.final_comm_penalty=0.01'`

In order to remove multitask training, add `--gin_binding='default_teams.use_comm_penalty_prob=1.0'`; to remove message forwarding, add `--gin_binding='default_teams.forward_messages=False'`

## Predator-Prey
Baseline:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/meet.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json`

IC3Net:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/meet.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/reinforce_global_gate.json --gin_binding='default_teams.final_comm_penalty=0.0'`

ECNet-REINFORCE Low Penalty:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/meet.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/reinforce_global_gate.json --gin_binding='default_teams.final_comm_penalty=0.005'`

ECNet-REINFORCE High Penalty:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/meet.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/reinforce_global_gate.json --gin_binding='default_teams.final_comm_penalty=0.1'`

ECNet-GS Low Penalty:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/meet.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/gumbel_global_gate.json --gin_binding='default_teams.final_comm_penalty=0.001'`

ECNet-GS High Penalty:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/meet.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/gumbel_global_gate.json --gin_binding='default_teams.final_comm_penalty=0.01'`

## Secret Pairs
ECNet-REINFORCE:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/secret_pairs.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/reinforce_p2p_gate.json --gin_binding='default_teams.final_comm_penalty=0.1' --gin_binding='default_teams.p2p_gen_type="p2p_obs"' --gin_binding='default_teams.use_select_comm_one=True'`

ECNet-GS:
`python3.7 scripts/reinforce/run.py --gin_file scripts/reinforce/configs/single_team/secret_pairs.json --gin_file scripts/reinforce/configs/comm_monitor.json --gin_file scripts/reinforce/configs/algos/tarmac.json --gin_file scripts/reinforce/configs/algos/gumbel_p2p_gate.json --gin_binding='default_teams.final_comm_penalty=0.01' --gin_binding='default_teams.p2p_gen_type="p2p_obs"' --gin_binding='default_teams.use_select_comm_one=True' --gin_binding='default_teams.temperature=0.5'`
