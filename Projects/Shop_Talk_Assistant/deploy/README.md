# Deploy (`deploy/`)

Operational scripts for EC2 setup and service deployment.

## Files

- `deploy/setup-ec2.sh`: one-time host bootstrap (Ubuntu 22.04)
  - installs Docker + compose plugin
  - optionally configures NVIDIA container runtime if GPU detected
  - prepares `~/shoptalk/data`

- `deploy/deploy.sh`: day-2 operations
  - `deploy`: build + start + health check
  - `restart`: restart services
  - `logs`: tail compose logs
  - `stop`: stop and remove services
  - `status`: compose status + backend health output

## Typical EC2 Flow

1. Run setup once:
   - `chmod +x deploy/setup-ec2.sh && ./deploy/setup-ec2.sh`
2. Upload notebook artifacts to `data/`.
3. Deploy:
   - `./deploy/deploy.sh`

## Notes

- `deploy.sh` requires `data/rag_text_index.npy` before deploy.
- For CPU-only machines, use compose override:
  - `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d`
