#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $DIR

docker volume create --name pgdata
docker-compose --env-file conf/env.conf up