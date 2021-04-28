#!/bin/bash
while :
do
  var_run=$(python gd_start.py)
  if [ "$var_run" == "no_experiments" ]; then
    echo "No experiments left to run"
    break
  fi
  echo "RUN: $var_run"

  IFS=' ' read -ra VALUES <<< "$var_run"
  name=$VALUES

  echo $name
  cd ../
  python main.py $var_run
  cd gd_runner/
  python gd_end.py
done