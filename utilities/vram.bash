#!/usr/bin/env bash
a=0
while true; do 
b=$(nvidia-smi --query-gpu=memory.used --format=csv|grep -v memory|awk '{print $1}')
clear
echo "Current: ${b}MiB"
[ $b -gt $a ] && a=$b
echo "Max: ${a}MiB" 
sleep 1.0
done