"""Calibrate the requested ProfilingDiT fast tier by speed; report quality separately."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def select_by_speed(rows, targets, tolerance):
    """Match ordered, distinct tiers without consulting any quality score."""
    if len(targets) != 1 or not math.isfinite(targets[0]) or targets[0] <= 1:
        raise ValueError('One finite speed target above 1x is required')
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('Invalid tolerance')
    if len({row['mode'] for row in rows}) != len(rows):
        raise ValueError('Duplicate candidate')
    for row in rows:
        if not math.isfinite(row['speedup']) or row['speedup'] <= 0:
            raise ValueError('Invalid timing')
    if not rows:
        raise ValueError('No candidates')
    eligible = [row for row in rows if abs(row['speedup'] / targets[0] - 1) <= tolerance]
    chosen = [min(eligible or rows, key=lambda row: (
        math.log(row['speedup'] / targets[0]) ** 2, row['mode']))]
    tiers = {}
    for name, row, target in zip(('profiling_speed_fast',), chosen, targets):
        tiers[name] = {'candidate': row['mode'], 'target_speedup': target,
                       'measured_speedup': row['speedup'],
                       'relative_target_error': abs(row['speedup'] / target - 1),
                       'target_met': abs(row['speedup'] / target - 1) <= tolerance}
    return {'tiers': tiers, 'all_targets_met': all(t['target_met'] for t in tiers.values()),
            'selection_uses_quality': False, 'quality_gate': None}


def validate_protocol(protocol):
    if protocol['quality_gate'] is not None or protocol['selection_uses_quality']:
        raise ValueError('This experiment must select by speed only')
    targets = protocol['speed_targets']
    if len(targets) != 1 or not math.isfinite(targets[0]) or targets[0] <= 1:
        raise ValueError('Invalid speed targets')
    if not 0 < protocol['speed_tolerance'] < 1:
        raise ValueError('Invalid speed tolerance')
    if protocol['generation']['steps'] != 50 or protocol['generation']['frames'] != 121:
        raise ValueError('The original frame count and step count must be preserved')
    validation, holdout = protocol['validation'], protocol['holdout']
    if len(validation) != 8 or len(holdout) != 8:
        raise ValueError('Expected eight validation and eight held-out cases')
    if {c['prompt'] for c in validation} & {c['prompt'] for c in holdout}:
        raise ValueError('Validation and holdout prompts overlap')
    ids = [c['id'] for c in validation + holdout]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate case IDs')
    ranking = protocol['ranking']
    if sorted(ranking) != list(range(28)):
        raise ValueError('Invalid 28-layer ranking')
    modes = [c['mode'] for c in protocol['candidates']]
    if len(set(modes)) != len(modes):
        raise ValueError('Duplicate candidate modes')
    for candidate in protocol['candidates']:
        k = len(candidate['background_blocks'])
        if not 0 < k < 28 or candidate['background_blocks'] != sorted(ranking[:k]):
            raise ValueError('Candidate does not follow the frozen profiling ranking')
        if not 1 <= candidate['min_interval'] <= candidate['max_interval']:
            raise ValueError('Invalid refresh intervals')


def verify_package(package, protocol):
    manifest = package / 'package_manifest.json'
    if sha(manifest) != protocol['package_manifest_sha256']:
        raise ValueError('Use the original verified release package')
    for name, meta in read(manifest)['files'].items():
        path = (package / name).resolve()
        if not path.is_relative_to(package.resolve()):
            raise ValueError('Package manifest path escapes package')
        if path.stat().st_size != meta['bytes'] or sha(path) != meta['sha256']:
            raise ValueError('Package changed: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('protocol.json'))
    parser.add_argument('--package', type=Path)
    parser.add_argument('--paths', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    protocol = read(args.protocol)
    validate_protocol(protocol)
    count = (len(protocol['candidates']) + 1) * len(protocol['validation'])
    count += (1 + len(protocol['speed_targets'])) * len(protocol['holdout'])
    if args.action == 'plan':
        print(json.dumps({'targets': protocol['speed_targets'], 'quality_gate': None,
                          'candidates': len(protocol['candidates']), 'generation_jobs': count,
                          'validation_cases': 8, 'holdout_cases': 8,
                          'protocol_sha256': sha(args.protocol), 'gpu_run_completed': False}, indent=2))
        return
    if not args.package or not args.output or sys.platform != 'linux':
        raise ValueError('run requires Linux CUDA, --package and a new --output directory')
    package, out = args.package.resolve(), args.output.resolve()
    verify_package(package, protocol)
    if read(package / 'configs/generation.json') != protocol['generation']:
        raise ValueError('Generation configuration changed')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'sweep.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'protocol_sha256': sha(args.protocol), 'sweep_code_sha256': sha(__file__)}
        identity_path = out / 'experiment_identity.json'
        if identity_path.exists() and read(identity_path) != identity:
            raise ValueError('Existing output belongs to a different experiment; use a new directory')
        save(identity_path, identity)
        sys.path.insert(0, str(package))
        from runtime import Runner, TimingGuard, GPUContended
        runner = Runner(args.paths)
        hardware_identity = {'gpu_uuid': runner.gpu_uuid}
        hardware_path = out / 'hardware_identity.json'
        if hardware_path.exists() and read(hardware_path) != hardware_identity:
            raise ValueError('Resume cannot mix physical GPUs across experiment cases')
        save(hardware_path, hardware_identity)
        # Settings are added only in memory; the published package is immutable.
        candidates = {c['mode']: {k: v for k, v in c.items() if k != 'mode'}
                      for c in protocol['candidates']}
        runner.methods.update(candidates)
        metric = None

        def status(stage, **fields):
            save(out / 'status.json', {'stage': stage, 'updated': time.time(), **fields})

        def execute(split, cases, methods):
            nonlocal metric
            for case in cases:
                for method in ['native'] + list(methods):
                    status('generation', split=split, case=case['id'], method=method)
                    runner.generate(case['prompt'], case['seed'], method,
                                    out / split / case['id'] / method, keep_latents=False)
            if metric is None:
                from score_pairs import CommonMetrics
                metric = CommonMetrics()
            rows = []
            for case in cases:
                native = out / split / case['id'] / 'native'
                left = read(native / 'run.json')
                for method in methods:
                    prediction = out / split / case['id'] / method
                    right = read(prediction / 'run.json')
                    if any(left['request'][k] != right['request'][k]
                           for k in ['prompt', 'seed', 'generation', 'identity']):
                        raise ValueError('Mismatched native reference')
                    if left['sigmas'] != right['sigmas']:
                        raise ValueError('Sampling schedule differs from native')
                    if left['hardware_gpu_uuid'] != right['hardware_gpu_uuid']:
                        raise ValueError('Resume cannot mix timings from different GPUs')
                    if not left['exclusive_gpu_timing_verified'] or not right['exclusive_gpu_timing_verified']:
                        raise ValueError('GPU timing was not exclusive')
                    destination = out / 'metrics' / split / case['id'] / (method + '.json')
                    status('quality_reporting', split=split, case=case['id'], method=method)
                    if destination.exists():
                        row = read(destination)
                        if row['reference_sha256'] != sha(native / 'video.mp4') or row['prediction_sha256'] != sha(prediction / 'video.mp4'):
                            raise ValueError('Cached score does not match video')
                    else:
                        guard = TimingGuard(runner.gpu_uuid)
                        with guard:
                            score = metric.compare(native / 'video.mp4', prediction / 'video.mp4')
                        guard.verify()
                        if score['frames'] != 121:
                            raise ValueError('Incomplete full-video metric')
                        if not all(math.isfinite(score[k]) for k in ['ssim', 'lpips']):
                            raise ValueError('Non-finite quality metric; this is invalid output, not a quality threshold')
                        row = {'id': case['id'], 'mode': method,
                               'native_seconds': left['online_seconds'],
                               'seconds': right['online_seconds'], **score}
                        save(destination, row)
                    rows.append(row)
            summaries = []
            for method in methods:
                group = [r for r in rows if r['mode'] == method]
                if len(group) != len(cases):
                    raise ValueError('Incomplete candidate')
                total_native, total = sum(r['native_seconds'] for r in group), sum(r['seconds'] for r in group)
                if not all(math.isfinite(r['seconds']) and r['seconds'] > 0 and
                           math.isfinite(r['native_seconds']) and r['native_seconds'] > 0 and
                           math.isfinite(r['ssim']) and math.isfinite(r['lpips']) for r in group):
                    raise ValueError('Invalid online timing')
                summaries.append({'mode': method, 'cases': len(group), 'mean_seconds': total / len(group),
                                  'speedup': total_native / total,
                                  'mean_ssim': sum(r['ssim'] for r in group) / len(group),
                                  'mean_lpips': sum(r['lpips'] for r in group) / len(group),
                                  'min_case_ssim': min(r['ssim'] for r in group),
                                  'max_case_lpips': max(r['lpips'] for r in group)})
            save(out / (split + '_summary.json'), {'complete': True, 'profiles': summaries})
            return summaries

        try:
            validation = execute('validation', protocol['validation'], candidates)
            selection = select_by_speed(validation, protocol['speed_targets'], protocol['speed_tolerance'])
            selection.update(identity)
            if not selection['all_targets_met']:
                save(out / 'closest_candidates.json', selection)
                status('needs_speed_grid_expansion', speed_targets_met=False,
                       reason='No candidate within the frozen speed tolerance; holdout not evaluated')
                raise SystemExit(2)
            presets = {}
            for name, tier in selection['tiers'].items():
                presets[name] = {**candidates[tier['candidate']],
                                 'target_speedup': tier['target_speedup'],
                                 'validation_speedup': tier['measured_speedup'],
                                 'speed_target_met': tier['target_met'],
                                 'tier_basis': 'closest measured validation speed; no quality threshold'}
            selection['presets'] = presets
            frozen = out / 'selection.json'
            if frozen.exists() and read(frozen) != selection:
                raise ValueError('Frozen selection changed')
            save(frozen, selection)
            runner.methods.update(presets)
            heldout = execute('holdout', protocol['holdout'], presets)
            save(out / 'summary.json', {'stage': 'complete', 'selection': selection,
                                       'validation': validation, 'holdout': heldout,
                                       'full_vbench_completed': False,
                                       'scope': 'Speed-selected LTX adaptation; not official WAN settings'})
            # Keep all existing methods; the requested fast tier has an explicit new name.
            save(out / 'methods.with_speed_tiers.json', read(package / 'configs/methods.json') | presets)
            status('complete', speed_targets_met=selection['all_targets_met'],
                   selection_sha256=sha(frozen), final_summary=str(out / 'summary.json'))
        except GPUContended as error:
            status('waiting_for_exclusive_gpu', error=str(error), resumable=True)
            raise SystemExit(75)
        except Exception as error:
            status('failed', error=str(error), resumable=True)
            raise


if __name__ == '__main__':
    main()
