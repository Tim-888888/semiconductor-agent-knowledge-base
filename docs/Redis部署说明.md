# Redis 独立部署说明

Redis 是一期 Celery 的 broker/result backend，不保存入库任务的业务真相；
`ingestion_jobs` 及其事件仍以 MongoDB 为准。

## 部署边界

部署清单为 `deploy/redis/docker-compose.redis.yml`。它只会创建本项目专属资源：

- 容器：`semikb-redis`
- 网络：`semikb-network`
- 持久化卷：`semikb-redis-data`
- 宿主机端口：`${REDIS_BIND_ADDRESS}:6379`

不会加入、重建或修改 MongoDB、Milvus、MinIO、Attu 的容器、网络、卷或配置。

## 运行方式

在 CentOS 服务器的 `/opt/semiconductor-agent-knowledge-base/` 放置部署清单和真实 `.env`：

```dotenv
REDIS_BIND_ADDRESS=192.168.10.100
REDIS_PASSWORD=<dedicated-random-password>
```

启动与检查：

```bash
cd /opt/semiconductor-agent-knowledge-base
chmod 600 .env
docker compose --env-file .env -f docker-compose.redis.yml up -d
docker compose --env-file .env -f docker-compose.redis.yml ps
```

Redis 7.4 使用密码认证、AOF、`appendfsync everysec`、`protected-mode yes` 和
`unless-stopped` 重启策略。客户端 `.env` 必须使用：

```dotenv
REDIS_URL=redis://:<same-password>@192.168.10.100:6379/0
```
