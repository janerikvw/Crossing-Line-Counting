## Google Drive Runner
This is a set of files to simply run multiple experiments from a spreadsheet in Google Drive.
Run run.sh to read all the files from the experiments.

The file runs gd_start.py to get the arguments and gd_end.py to update the experiments file.
The arguments can be tweaked inside the gd_start.py. The experiments spreadsheet has to have the arguments columns on the top row and each row functions as a experiment.

The Google Gspread token can be found in your Google Account and the spreadsheet name on top of your spreadsheet URL. Both have to be edited in both files to function properly.