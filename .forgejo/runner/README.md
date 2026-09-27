# Forgejo runner

Builds the root `Dockerfile` and pushes the image to a registry whenever a
`v<semver>` tag is pushed, via [`../workflows/docker.yml`](../workflows/docker.yml).
The workflow needs a runner that gives its jobs a Docker daemon; this directory
is one: a Forgejo runner plus an isolated Docker-in-Docker daemon, following
Forgejo's own Docker installation guide.

## 1. Start the runner

1. In Forgejo, open **Settings → Actions → Runners** on the repository (or the
   organisation, or the site admin panel) and **Create new runner**. Keep the
   UUID and token it shows.
2. On the build host, from this directory:

   ```bash
   mkdir -p data/.cache
   cp runner-config.example.yml data/runner-config.yml
   # edit data/runner-config.yml: server.connections.forgejo.url, uuid, token
   sudo chown -R 1001:1001 data
   sudo chmod 775 data/.cache && sudo chmod g+s data/.cache
   docker compose up -d
   ```

   The runner should then show as online in Forgejo with the `docker` label.
   `data/` holds the runner token and is git-ignored.

Already running a runner? Anything works that has the `docker` label and gives
job containers a Docker daemon: `DOCKER_HOST` in `runner.envs` (as here) or
`container.docker_host: automount`, which mounts the host's socket into jobs.

## 2. Configure the repository

Enable Actions for the repository (**Settings → Units**), then under
**Settings → Actions** add:

| Name | Kind | Required | Purpose |
|---|---|---|---|
| `REGISTRY_USERNAME` | secret | yes | Registry login. |
| `REGISTRY_PASSWORD` | secret | yes | Access token allowed to push packages/images. |
| `REGISTRY` | variable | no | Registry host, e.g. `docker.io` for Docker Hub. Defaults to this Forgejo instance's own container registry. |
| `IMAGE` | variable | no | Image path without the registry, e.g. `mauriceboe/trek`. Defaults to `<owner>/<repo>`, lower-cased. |
| `PLATFORMS` | variable | no | e.g. `linux/amd64,linux/arm64`. Foreign platforms are emulated with QEMU (slow). Defaults to the runner's native platform. |

## 3. Release

```bash
git tag v4.3.1 && git push origin v4.3.1
```

| Tag | Image tags pushed |
|---|---|
| `v4.3.1` | `4.3.1`, `4`, `latest` |
| `v4.4.0-pre.1` | `4.4.0-pre.1`, `4-pre`, `latest-pre` |

The version (without the `v`) is baked in as `APP_VERSION`.

Once `.forgejo/workflows/` exists, Forgejo runs only the workflows in it and
ignores `.github/workflows/`, so the GitHub-only release, test and housekeeping
workflows do not run on a Forgejo mirror.
