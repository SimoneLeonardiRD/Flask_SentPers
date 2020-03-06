#!/bin/bash
#grep zombies from preavious execution of the script and kill them
pidn=$(ps | grep "python3" | awk '{print $1}')
readarray -t strarr <<< "$pidn"
for (( n=0; n < ${#strarr[*]}; n++))
do kill -KILL "${strarr[n]}"
done