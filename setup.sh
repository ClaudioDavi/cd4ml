# This should run on a ECS Instance
docker build -t Dockerfile_dvc_Agent gocd-dvc-agent

docker run -d -p8153:8153 gocd/gocd-server:v21.1.0

docker run -d -e GO_SERVER_URL=http://$(hostname -I):$(docker inspect --format='{{(index (index .NetworkSettings.Ports "8153/tcp") 0).HostPort}}' boring_sanderson)/go  gocd-dvc-agent