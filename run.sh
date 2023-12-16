#!/bin/bash

#for i in `seq 0 1`; do
echo "Starting node 5000"
python3 main.py localhost:5000 localhost:5001 localhost:5002 localhost:5003

echo "Starting node 5001"
python3 main.py localhost:5001 localhost:5000 localhost:5002 localhost:5003

echo "Starting node 5002"
python3 main.py localhost:5002 localhost:5000 localhost:5001 localhost:5003

echo "Starting node 5003"
python3 main.py localhost:5003 localhost:5000 localhost:5001 localhost:5002
#done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait