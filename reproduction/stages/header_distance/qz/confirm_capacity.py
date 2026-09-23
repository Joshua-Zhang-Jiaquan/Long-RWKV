"""User-authorized confirmation allocation: original project32 plus extra project16."""
import campaign as C

PRIMARY = 'project-160ccb20-98ab-4538-a847-01d1f83d5b0f'
EXTRA = 'project-632c8db8-4530-413a-ada5-df91774a7e09'
PROJECT_CAPS = {PRIMARY: 32, EXTRA: 16}
CAP = sum(PROJECT_CAPS.values())


class CapacityUnavailable(SystemExit):
    """Pre-submission capacity refusal; no scheduler mutation has occurred."""


def live_usage(jobs):
    rows = []
    usage = {project: 0 for project in PROJECT_CAPS}
    for job in jobs:
        if not (job.get('name') or '').startswith('lrwkv-') or job.get('status') not in C.LIVE_STATUSES:
            continue
        project = job.get('project_id')
        if project not in usage:
            raise ValueError('Live campaign job has unknown budget project: ' + str(project))
        gpus = max(8, int(job.get('gpu_count') or 0))
        usage[project] += gpus
        rows.append(dict(job_id=job.get('job_id'), name=job.get('name'), status=job['status'],
                         project_id=project, reserved_gpus=gpus))
    return dict(cap=CAP, project_caps=PROJECT_CAPS.copy(), project_reserved_gpus=usage,
                live_reserved_gpus=sum(usage.values()), jobs=rows)


def route_body(body, census, preferred_project=None):
    if census['live_reserved_gpus'] + 8 > CAP:
        raise CapacityUnavailable(f'{CAP} GPU cap')
    # The live resource-price API confirms the same eight-H100/1800GiB spec in
    # both projects; evidence: results/train04confirm/extra_project_resource_specs.json.
    shapes = body['framework_config']
    if len(shapes) != 1 or shapes[0]['instance_count'] != 1 or shapes[0]['spec_id'] != '7166bd2e-6cbe-4bd9-be38-762d11003e7f' or shapes[0]['shm_gi'] != 1800:
        raise ValueError('Routing requires the verified eight-H100 shape')
    if preferred_project is not None and preferred_project not in PROJECT_CAPS:
        raise ValueError('Unknown preferred budget project')
    order=(PRIMARY,EXTRA) if preferred_project==PRIMARY else (EXTRA,PRIMARY)
    for project in order:
        if census['project_reserved_gpus'][project] + 8 <= PROJECT_CAPS[project]:
            body['project_id'] = project
            return project
    raise CapacityUnavailable(f'{CAP} GPU cap: no authorized project has eight free slots')
