#!/bin/bash
ssh-add
sudo -E ssh -A -L 9000:localhost:9000 -L 443:localhost:443 eagle@185.211.184.60