#!/bin/bash
#grep zombies from preavious execution of the script and kill them
pidn=$(ps | grep "python3" | awk '{print $1}')
readarray -t strarr <<< "$pidn"
for (( n=0; n < ${#strarr[*]}; n++))
do kill -KILL "${strarr[n]}"
done
#prepare the flask environment and folder
export FLASK_APP=flaskr
export FLASK_ENV=development
flask run --host=0.0.0.0
