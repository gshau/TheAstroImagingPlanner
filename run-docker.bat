docker pull gshau/astroimaging-planner:latest
docker volume create --name pgdata 
docker-compose --env-file conf/env.conf up