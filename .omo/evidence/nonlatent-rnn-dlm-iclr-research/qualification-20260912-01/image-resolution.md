# Image resolution — qualification nonlatent-h100-qualification-20260912-01

Date: 2026-09-12. Read-only pass, 7 qz API calls (bound <=8). No docker pull/build, no registry credentials, no job create/dry-run/stop.

## Commands (exit 0 unless noted)
1. `qz schema image.ListImages`
2. `qz schema image.GetImageById`
3. `qz image ListImages --data '{"filter":{"keyword":"relay2"},...}'` — API error `InvalidParameter: unknown field "keyword"` (proto-strict filter); not retried with same shape
4. `qz train GetTrainScheduleConfig --data '{"workspace_id":"ws-9dcc0e1f-..."}'` — spec refresh
5. `qz image ListImages --data '{"filter":{"name":"relay2"},"page_size":20}'` — OK
6. `qz schema workspace.GetWorkspaceQuota`
7. `qz workspace GetWorkspaceQuota --data '{"workspace_id":"ws-9dcc0e1f-..."}'` — **AccessForbidden: not workspace admin** (bounded single attempt, no loop)

## Image identity (scheduler-managed catalog, call 5)
- Target tag `docker.sii.shaipower.online/inspire-studio/relay2:v2` found:
  - **image_id `image-7330d118-df9d-4e2e-82b1-c2543e831eb4`** (stable scheduler identity)
  - name `relay2:v2`, version `v2`, status `SUCCESS`, visibility `VISIBILITY_PRIVATE`, source `SOURCE_PUBLIC`
  - registry_id `qbHarbor`, region 七宝机房, size 50.32 GB, add_method `Notebook`, publish_status `NOT_PUBLISHED`
  - project binding: registered under current user's catalog scope; same address+tag the reference job job-83690070 ran
- Second match was unrelated `relay2-cpu:v2` (image-39004fab..., status FAILED) — NOT the target; do not confuse.

## Content digest: UNKNOWN — cannot be observed via read-only qz APIs
- `GetImageById` schema response fields: image_id, name, source, visibility, address, version, size, status, add_method, description, preheat_*, support_brand_info_list, registry_id, region, publish_status, project_id — **no digest field**
- Catalog listing payload contains zero occurrences of `digest`/`sha256`
- `train.CreateJob` `framework_config` accepts only `image` (address:tag string) + `image_type` — **no image_id and no digest parameter exists**; the scheduler pins whatever the registry currently serves for the tag at submit time
- Therefore: **immutable content digest UNKNOWN** (not a hash of the URL, not the image_id — image_id is a scheduler row id, not content identity). Distinguish: catalog metadata at API-request time ≠ digest actually observed in the pod. Per authorization, a missing digest is NOT to be invented; QUALIFICATION_IMAGE_DIGEST cannot be produced from qz reads alone. If a digest is required, it needs an external authoritative source (e.g., the relay2:v2 image owner/registry UI) or a later in-job record of the pulled digest as observed evidence — subject to authorization constraints.

## Spec / pool refresh (call 4, this pass)
- spec `7166bd2e-6cbe-4bd9-be38-762d11003e7f` ("8卡160核", gpu=8, cpu=160) still listed in `predef_train_spec` and still scoped to `lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e` (H100 pool). Node type is pool-bound; spec is shared across H200/H100 pools → always pin LCG 71b971a7.

## Quota
- Read path blocked by RBAC: workspace-quota API requires workspace admin. No quota numbers obtained; submit-time quota exhaustion remains an unverified risk (single fail-and-stop per authorization).

## Conclusions for submission design
- Approved tuple remains expressible: image `relay2:v2` by tag + image_type SOURCE_PRIVATE? — NOTE: catalog says source `SOURCE_PUBLIC` for this image record while reference job used `image_type: SOURCE_PRIVATE` in framework_config; keep `image_type` consistent with the reference job's proven-good value (`SOURCE_PRIVATE`) since CreateJob takes it as a free string; flag as minor open item for the submission pass.
- Tag-pinned only: registry-side retag of v2 between now and submit cannot be detected from here. Record the catalog image_id and this evidence as the best available identity at authorization time.
- Raw API outputs held in temp only and deleted; creator PII excluded; no credentials touched.
