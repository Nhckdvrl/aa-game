#!/bin/bash
# Run on n04: wait until no farm is using GPU $1, then run farm.sh with the remaining args (env passes through).
G=$1; shift
while pgrep -f "farm.sh $G " > /dev/null; do sleep 30; done
exec bash /data2/xiangding/companion/logs/snap/farm.sh $G "$@"
