#!/bin/bash
docker volume create --name pgdata
docker-compose --env-file conf/env.conf -f docker-compose.yml -f docker-compose-with-planner.yml up