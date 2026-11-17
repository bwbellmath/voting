#!/bin/bash
# Held-out comparison across dimensions / type counts / split seeds. Logs to outputs/compass/logs/.
cd "$(dirname "$0")/.." || exit 1
R=".venv/bin/python scripts/run_compass.py --offices PSH --M 64 --steps 1200"
L=outputs/compass/logs; mkdir -p $L
$R --split random --d 1 --K 3 > $L/sw_d1K3.log 2>&1
$R --split random --d 3 --K 6 > $L/sw_d3K6.log 2>&1
$R --split random --seed 1 > $L/sw_s1.log 2>&1
$R --split random --seed 2 > $L/sw_s2.log 2>&1
$R --split random > $L/sw_s0.log 2>&1
$R --split temporal > $L/sw_temporal.log 2>&1
