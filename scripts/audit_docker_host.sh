#!/usr/bin/env bash

# T9 deployment inventory. This script only reads host and Docker metadata.
# It deliberately avoids printing container environment variables or secrets.

set -u

section() {
  printf '\n===== %s =====\n' "$1"
}

run_readonly() {
  "$@" 2>&1 || true
}

docker_snapshot() {
  docker ps --no-trunc \
    --format '{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}' 2>/dev/null \
    | sort
}

inspect_running_containers() {
  local container
  while IFS= read -r container; do
    [ -n "$container" ] || continue
    docker inspect --format \
      '{{.Name}}|image={{.Config.Image}}|memory={{.HostConfig.Memory}}|nano_cpus={{.HostConfig.NanoCpus}}|restart={{.HostConfig.RestartPolicy.Name}}|network={{.HostConfig.NetworkMode}}' \
      "$container" 2>&1 || true
  done < <(docker ps --format '{{.Names}}' 2>/dev/null | sort)
}

container_versions() {
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'mongo'; then
    printf '%s\n' '--- mongo ---'
    docker exec mongo mongosh --quiet --eval 'db.version()' 2>&1 \
      || docker exec mongo mongo --quiet --eval 'db.version()' 2>&1 \
      || true
  fi

  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'milvus-minio'; then
    printf '%s\n' '--- milvus-minio ---'
    docker exec milvus-minio minio --version 2>&1 | head -n 2 || true
  fi

  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'milvus-etcd'; then
    printf '%s\n' '--- milvus-etcd ---'
    docker exec milvus-etcd etcd --version 2>&1 | head -n 2 || true
  fi

  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'minio'; then
    printf '%s\n' '--- minio ---'
    docker exec minio minio --version 2>&1 | head -n 2 || true
  fi

  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'semikb-redis'; then
    printf '%s\n' '--- semikb-redis ---'
    docker exec semikb-redis redis-server --version 2>&1 || true
  fi
}

before_file="$(mktemp)"
after_file="$(mktemp)"
trap 'rm -f "$before_file" "$after_file"' EXIT

docker_snapshot > "$before_file"

section "AUDIT METADATA"
printf 'audit_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
printf 'hostname=%s\n' "$(hostname 2>/dev/null || true)"

section "OPERATING SYSTEM"
run_readonly cat /etc/os-release
run_readonly uname -a
run_readonly getenforce
run_readonly systemctl is-active firewalld

section "CPU AND MEMORY"
run_readonly nproc
run_readonly lscpu
run_readonly free -b

section "DISK"
run_readonly df -B1 /
# CentOS 7 lsblk requires the singular mount-point column.
run_readonly lsblk -b -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT

section "DOCKER RUNTIME"
run_readonly docker version
run_readonly docker compose version
run_readonly docker info --format 'driver={{.Driver}} cgroup_driver={{.CgroupDriver}} cgroup_version={{.CgroupVersion}} containers={{.Containers}} images={{.Images}}'
run_readonly docker system df

section "RUNNING CONTAINERS"
cat "$before_file"

section "CONTAINER RESOURCE POLICIES"
inspect_running_containers

section "CONTAINER RESOURCE SNAPSHOT"
run_readonly docker stats --no-stream --format '{{.Name}}|cpu={{.CPUPerc}}|memory={{.MemUsage}}|memory_percent={{.MemPerc}}'

section "SERVICE VERSIONS"
container_versions

section "DOCKER NETWORKS AND VOLUMES"
run_readonly docker network ls
run_readonly docker volume ls

section "LISTENING PORTS"
run_readonly ss -lntup

docker_snapshot > "$after_file"
section "READ-ONLY GUARD"
if cmp -s "$before_file" "$after_file"; then
  printf '%s\n' 'containers_unchanged=true'
  exit 0
fi

printf '%s\n' 'containers_unchanged=false'
printf '%s\n' 'The running-container snapshot changed during the audit. Review external activity.'
diff -u "$before_file" "$after_file" || true
exit 3
