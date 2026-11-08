#!/bin/bash
# Run on n05: wait for farm.sh on GPU $1 to finish, then run steer_matrix.sh with the remaining args.
G=$1; shift
while pgrep -f "farm.sh $G " > /dev/null; do sleep 30; done
exec bash /data2/xiangding/companion/scripts/steer_matrix.sh $G "$@"
