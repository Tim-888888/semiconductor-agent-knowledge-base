#!/usr/bin/env bash
set -euo pipefail

mongosh --quiet \
  --username "$MONGO_INITDB_ROOT_USERNAME" \
  --password "$MONGO_INITDB_ROOT_PASSWORD" \
  --authenticationDatabase admin \
  "$MONGO_INITDB_DATABASE" \
  --eval '
    db.createUser({
      user: process.env.MONGO_APP_USERNAME,
      pwd: process.env.MONGO_APP_PASSWORD,
      roles: [
        { role: "readWrite", db: process.env.MONGO_INITDB_DATABASE },
        { role: "dbAdmin", db: process.env.MONGO_INITDB_DATABASE }
      ]
    })
  '
