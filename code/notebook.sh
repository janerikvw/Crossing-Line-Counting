#!/bin/bash
conda init bash
conda activate LOI
jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''