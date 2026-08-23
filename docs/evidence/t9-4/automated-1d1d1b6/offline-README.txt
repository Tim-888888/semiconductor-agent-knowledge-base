SemiAtlas T9-4.8 offline migration bundle

1. Copy this entire directory to the target host.
2. Extract source.tar.gz into /opt/semiconductor-agent-knowledge-base/.
3. Create the target .env separately; this bundle contains no credentials or business data.
4. Run: python3 scripts/verify_t948_offline_bundle.py --bundle-dir <bundle> --load --verify-docker
5. Run: ./scripts/deployment/deploy.sh --offline

Restore business data from a separately governed cold backup after verifying the target paths.
