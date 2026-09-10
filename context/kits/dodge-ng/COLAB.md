# Colab jobs through colabctl

`dodge-ng-colab` submits the NG DDQN/HPO trainer through `colabctl`'s durable
job API. The adapter packages the current Python/native source, embeds that
payload in a remote bootstrap, and verifies returned artifacts. `colabctl`
handles allocation, detached execution, status, logs, cancellation, and
session reattachment over the sanctioned Google `colab` CLI transport.

The previous custom controller remains available as
`dodge-ng-colab-legacy` while this path is exercised.

## Install and inspect

The project keeps `colabctl` in a Python 3.12+ optional extra because the main
project still supports Python 3.11:

```bash
uv sync --extra native --extra colab
just dodge-colabctl doctor
just dodge-colabctl job backends
```

The Colab extra pins the Google Colab kernel-client fork commit required by the
installed `google-colab-cli`; keep that lock in place when refreshing
dependencies.

No login is needed for local planning, packaging tests, or the commands above
that only inspect local configuration. Login is needed immediately before the
first real submission or remote status/log/download operation:

```bash
just dodge-colabctl auth login
```

That command is intentionally not run by the project tooling.

## Plan and submit

First validate the trainer arguments without building a wheel or contacting
Colab:

```bash
just dodge-ng-colab plan \
  --run-dir history/dodge/ng/waypoint-hpo-colab \
  --gpu-tiers T4,L4,A100,H100 \
  -- --trials 8 --budgets 20000,60000,120000 --native-lanes 32 --device cuda
```

When ready to allocate a runtime, submit a detached job. The command returns
the `colabctl` job ID and exits; it does not run a local supervisor:

```bash
just dodge-ng-colab submit \
  --run-dir history/dodge/ng/waypoint-hpo-colab \
  --gpu-tiers T4,L4,A100,H100 \
  -- --trials 8 --budgets 20000,60000,120000 --native-lanes 32 --device cuda
```

GPU attempts are T4-first and advance only when `colabctl` reports an
accelerator-unavailable error. Quota, authentication, parser, and transport
failures stop the submission instead of silently escalating.

## Observe, retrieve, and stop

Use the adapter commands for lifecycle operations. They reattach through the
same `colabctl` state store and official CLI transport:

```bash
just dodge-ng-colab status <colab-job-id>
just dodge-ng-colab logs <colab-job-id> --tail 200
just dodge-ng-colab watch <colab-job-id>
just dodge-ng-colab retrieve <colab-job-id>
just dodge-ng-colab stop <colab-job-id>
```

`watch` downloads and verifies the artifact archive when the job reaches a
terminal state, then releases the runtime. If transfer or verification fails,
it leaves the runtime allocated so the artifacts can be recovered. Pass
`--keep-session` to `watch` or `stop` when the runtime must remain available.
`retrieve` is idempotent for an unchanged verified archive and refuses to
overwrite an unrelated existing run directory.

Local records are intentionally small and live under
`history/dodge/ng/colab-jobs/<colab-job-id>/`: `job.json`, `status.json`, a
bounded `colabctl.log`, and the downloaded artifact archive/receipt. The
durable `colabctl` job record lives in `~/.colabctl`.

The detached backend leaves the remote runtime allocated until `watch` or
`stop` releases it. This is deliberate: an explicit release avoids conflating
job completion with artifact transfer completion.

## Leave-running monitor

For the current unattended HPO run, the host user timer checks the durable job
every ten minutes. On terminal completion it runs `watch`, verifies and
retrieves the artifacts, releases the runtime, and writes a handoff marker
under the local job directory. The timer cannot choose new experiments; the
post-report diagnosis and iteration remain evidence-driven.
