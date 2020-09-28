#!/bin/bash
while :
do
  var_run=$(python run_spreadsheet.py)
  if [ "$var_run" == "no_experiments" ]; then
    echo "No experiments left to run"
    break
  fi
  echo "RUN: $var_run"

  IFS=' ' read -ra VALUES <<< "$var_run"
  name=$VALUES

  echo $name
  cd full_on_pwc
  python main.py $var_run
  cd ../
  python finishing_spreadsheet.py
done