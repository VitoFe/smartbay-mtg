"""Isolated new bay-twin campaign transport. Existing deploy credentials unchanged."""
from pathlib import Path
import argparse, hashlib, io, json, shlex, tarfile, sys
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scientific_campaign import connect

ROOT=Path(__file__).resolve().parents[1]

def execute(client,command):
    stdin,stdout,stderr=client.exec_command(command,timeout=60)
    try:
        out=stdout.read().decode('utf8',errors='replace')
        error=stderr.read().decode('utf8',errors='replace')
        return stdout.channel.recv_exit_status(),out+error
    finally:
        stdin.close(); stdout.close(); stderr.close(); stdout.channel.close()

def launch(client,base,smoke,campaign='thermal'):
    suffix=('_thermal_smoke' if smoke else '_thermal_discovery') if campaign=='thermal' else ('_oxygen_smoke' if smoke else '_oxygen_pilot')
    if campaign=='warning': suffix='_warning_smoke' if smoke else '_warning_pilot'
    if campaign=='transfer': suffix='_transfer_smoke' if smoke else '_transfer_pilot'
    if campaign=='field': suffix='_field_smoke' if smoke else '_field_pilot'
    if campaign=='kernel': suffix='_kernel_smoke' if smoke else '_kernel_pilot'
    if campaign=='boundary': suffix='_boundary_smoke' if smoke else '_boundary_pilot'
    if campaign=='pretrain': suffix='_pretrain_smoke' if smoke else '_pretrain_pilot'
    if campaign=='dynamics': suffix='_dynamics_smoke' if smoke else '_dynamics_pilot'
    if campaign=='reactive': suffix='_reactive_smoke' if smoke else '_reactive_pilot'
    if campaign=='dynamics_tree': suffix='_dynamics_tree_smoke' if smoke else '_dynamics_tree_pilot'
    if campaign=='graph_geometry': suffix='_graph_geometry_smoke' if smoke else '_graph_geometry_pilot'
    if campaign=='oxygen_memory': suffix='_oxygen_memory_pilot'
    if campaign=='thermal_memory': suffix='_thermal_memory_pilot'
    if campaign=='transfer_calibration': suffix='_transfer_calibration_pilot'
    if campaign=='split_calibration': suffix='_split_calibration_pilot'
    if campaign=='resilience': suffix='_resilience_pilot'
    if campaign=='state_update': suffix='_state_update_pilot'
    if campaign=='observation_stress': suffix='_observation_stress_pilot'
    if campaign=='field_export': suffix='_field_export_pilot'
    if campaign=='robust': suffix='_robust_pilot'
    if campaign=='seasonal': suffix='_seasonal_pilot'
    if campaign=='decision_calibration': suffix='_decision_calibration_pilot'
    if campaign=='forced_thermal': suffix='_forced_thermal_smoke' if smoke else '_forced_thermal_pilot'
    if campaign=='mesh_features': suffix='_mesh_features_smoke' if smoke else '_mesh_features_pilot'
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+suffix
    local=ROOT/'reports/bay_twin/runs'/run; local.mkdir(parents=True)
    files=[ROOT/p for p in ['src/experiments/bay_twin_sweep.py','src/data/bay_observations.py',
        'config/bay_twin_campaign.json','scripts/build_bay_geometry.py','scripts/fetch_bay_landcover.py',
        'tests/test_bay_twin_contracts.py']]
    files+=list((ROOT/'reports/bay_twin').glob('*.json'))
    files+=[ROOT/'reports/bay_twin'/p for p in ['sensor_inventory.csv','sensor_geometry.csv','hourly_observations.csv.gz','geometry.npz']]
    audit=json.loads((ROOT/'reports/bay_twin/data_audit.json').read_text())
    files += [ROOT/p for p in audit['sources_sha256']]
    files += list((ROOT/'data/external/CoastalZones2018').glob('*'))
    if campaign in ('field','kernel','boundary','pretrain','field_export','observation_stress','state_update','resilience','split_calibration','transfer_calibration','thermal_memory','oxygen_memory'):
        files += [ROOT/p for p in ['src/data/bay_profile_context.py','src/models/bay_query_field.py',
            'src/experiments/bay_profile_field.py','config/bay_profile_field.json','tests/test_bay_query_field.py',
            'src/data/thermodynamics.py','reports/bay_twin/gen2_profile_inventory.csv','data/processed/ctd_all_casts.csv.gz']]
        import csv
        with (ROOT/'reports/bay_twin/gen2_profile_inventory.csv').open() as stream:
            files += [ROOT/'data/raw'/(row['cast_id']+'.txt') for row in csv.DictReader(stream)]
        if campaign in ('kernel','boundary'): files.append(ROOT/'src/experiments/bay_kernel_field.py')
        if campaign in ('boundary','field_export','observation_stress','state_update','resilience','split_calibration','transfer_calibration','thermal_memory','oxygen_memory'):
            files += [ROOT/p for p in ['scripts/build_bay_boundary_context.py','reports/bay_twin/profile_boundary_context.csv.gz']]
            provenance=json.loads((ROOT/'reports/bay_twin/profile_boundary_provenance.json').read_text())
            files += [ROOT/p for p in provenance['input_sha256']]
        if campaign=='pretrain':
            files += [ROOT/p for p in ['src/data/bay_thermal_supervision.py','src/experiments/bay_thermal_pretraining.py',
                'config/bay_thermal_pretraining.json','tests/test_bay_thermal_supervision.py','scripts/build_bay_thermal_supervision.py']]
            files += list((ROOT/'reports/bay_twin/thermal_supervision').glob('*'))
            for task in ['temporal','held_site']:
                files.append(ROOT/'reports/bay_twin/runs/20260910T063618Z_field_pilot/results'/('field_'+task)/'selection_summary.json')
    if campaign=='field_export':
        files.append(ROOT/'src/experiments/bay_field_export.py')
        for task in ['temporal','held_site']:
            source=ROOT/'reports/bay_twin/runs/20260910T065505Z_boundary_pilot/results'/('kernel_'+task)
            selection=json.loads((source/'selection_summary.json').read_text())
            files += [source/'selection_summary.json',source/('trial_%03d_linear_residual_predictions.csv.gz'%selection['trial'])]
    if campaign in ('observation_stress','state_update','resilience','split_calibration','transfer_calibration','thermal_memory','oxygen_memory'):
        files += [ROOT/'src/experiments/bay_field_export.py',ROOT/'src/experiments/bay_observation_stress.py']
        source=ROOT/'reports/bay_twin/runs/20260917T011757Z_field_export_pilot/results'
        files += list(source.rglob('*.json'))+list(source.rglob('*.csv.gz'))
    if campaign in ('state_update','resilience','split_calibration','transfer_calibration','thermal_memory','oxygen_memory'):
        files += [ROOT/'src/experiments/bay_state_update.py',ROOT/'src/models/bay_observation_filter.py',ROOT/'tests/test_bay_observation_filter.py']
    if campaign=='resilience':
        files += [ROOT/'src/experiments/bay_resilience.py',ROOT/'src/models/bay_localized_update.py']
        files += list((ROOT/'reports/bay_twin/runs/20260917T080417Z_state_update_pilot/results').rglob('*.json'))
    if campaign in ('split_calibration','transfer_calibration','thermal_memory','oxygen_memory'):
        files += [ROOT/'src/experiments/bay_split_calibration.py',ROOT/'src/data/bay_thermal_supervision.py']
        files += list((ROOT/'reports/bay_twin/thermal_supervision').glob('*'))
    if campaign in ('transfer_calibration','thermal_memory','oxygen_memory'):
        files.append(ROOT/'src/experiments/bay_transfer_calibration.py')
    if campaign in ('thermal_memory','oxygen_memory'):
        files += [ROOT/'src/experiments/bay_thermal_memory.py',ROOT/'tests/test_bay_thermal_memory.py']
    if campaign=='oxygen_memory':
        files.append(ROOT/'src/experiments/bay_oxygen_memory.py')
    if campaign in ('oxygen','warning','transfer','decision_calibration'):
        files += [ROOT/p for p in ['src/experiments/oxygen_trend_pilot.py','src/protocols/oxygen_trends.py',
            'src/data/thermodynamics.py','src/models/bay_physics.py','config/oxygen_trend_pilot.json',
            'data/raw/T2-Meteo.txt','tests/test_bay_physics_trends.py']]
        if campaign=='warning': files.append(ROOT/'src/experiments/oxygen_warning_head.py')
        if campaign=='transfer':
            files += [ROOT/p for p in ['src/experiments/oxygen_transfer_pilot.py',
                'config/oxygen_transfer_campaign.json','tests/test_oxygen_transfer.py']]
    if campaign in ('dynamics','reactive','dynamics_tree','graph_geometry','mesh_features','forced_thermal','seasonal'):
        files += [ROOT/p for p in ['src/data/bay_profile_context.py','src/models/bay_graph_dynamics.py','src/experiments/bay_graph_dynamics.py','tests/test_bay_graph_dynamics.py','tests/test_bay_dynamics_data.py','src/models/bay_query_field.py','src/experiments/bay_dynamics_tree.py']]
    if campaign=='graph_geometry':
        files += [ROOT/'src/experiments/bay_graph_geometry.py',ROOT/'scripts/build_bay_exchange_geometry.py']
        for variable in ['temperature','oxygen']:files.append(ROOT/'reports/bay_twin/runs/20260910T115509Z_dynamics_pilot/results'/f'dynamics_{variable}/selection_summary.json')
    if campaign=='mesh_features':
        files += [ROOT/'src/experiments/bay_mesh_features.py',ROOT/'src/models/bay_finite_volume.py']
        files += [ROOT/'reports/bay_twin/layered_mesh'/name for name in ['columns.csv','sensor_bathymetry_audit.csv','mesh_shallow.npz','mesh_central.npz','mesh_deep.npz','provenance.json']]
        files.append(ROOT/'tests/test_bay_finite_volume.py')
        for variable in ['temperature','oxygen']:files.append(ROOT/'reports/bay_twin/runs/20260910T120301Z_dynamics_tree_pilot/results'/f'dynamics_{variable}/selection_summary.json')
    if campaign=='forced_thermal':
        files += [ROOT/p for p in ['src/data/bay_forcing.py','src/models/bay_forced_thermal.py','src/experiments/bay_forced_thermal.py','data/raw/T2-Meteo.txt','tests/test_bay_forced_thermal.py']]
        files += [ROOT/'reports/bay_twin/layered_mesh'/name for name in ['columns.csv','sensor_bathymetry_audit.csv','mesh_central.npz','boundary_context.nc','boundary_audit.json','provenance.json']]
        files.append(ROOT/'reports/bay_twin/runs/20260910T120301Z_dynamics_tree_pilot/results/dynamics_temperature/selection_summary.json')
    if campaign=='decision_calibration':
        files += [ROOT/p for p in ['src/experiments/bay_decision_calibration.py','src/protocols/bay_calibration.py','src/protocols/oxygen_episodes.py','tests/test_bay_calibration.py','tests/test_oxygen_episodes.py']]
        source=ROOT/'reports/bay_twin/runs/20260916T022938Z_conditional_oxygen_pilot/results/conditional_oxygen'
        files += [source/p for p in ['calibration.csv.gz','tree_predictions.csv.gz','weather_tree_predictions.csv.gz']]
        for family in ['oxygen_history','thermal_weather_geo','physics_features']:
            source=ROOT/'reports/bay_twin/runs/20260910T000056Z_warning_pilot/results'/('warning_6h_'+family)
            trial=json.loads((source/'selection_summary.json').read_text())['trial']
            files += [source/'selection_summary.json',source/f'trial_{trial:03d}_predictions.csv.gz']
            files += [source/f'trial_{trial:03d}_fold{fold}_calibration.csv.gz' for fold in [0,1]]
    if campaign=='seasonal':
        files += [ROOT/p for p in ['src/experiments/bay_seasonal_validation.py','src/data/bay_forcing.py','src/protocols/bay_calibration.py','data/raw/T2-Meteo.txt','tests/test_bay_calibration.py']]
        for variable in ['temperature','oxygen']:files.append(ROOT/f'reports/bay_twin/runs/20260910T120301Z_dynamics_tree_pilot/results/dynamics_{variable}/selection_summary.json')
    if campaign=='robust':
        files += [ROOT/'src/experiments/bay_robust_forecast.py',ROOT/'src/protocols/bay_calibration.py']
        for variable in ['temperature','oxygen']:
            source=ROOT/'reports/bay_twin/runs/20260916T194019Z_seasonal_pilot/results'/f'seasonal_{variable}'
            files += [source/'predictions.csv.gz',source/'calibration.csv.gz']
    files=list(dict.fromkeys(files))
    manifest={'run_id':run,'files_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'created_utc':datetime.now(timezone.utc).isoformat(),'smoke_only':smoke,
              'evaluation':'Validation-only discovery. All 2026 values excluded by experiment code.'}
    (local/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=local/'snapshot.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files: tar.add(p,arcname=p.relative_to(ROOT).as_posix())
        content=json.dumps(manifest).encode(); info=tarfile.TarInfo('snapshot_manifest.json'); info.size=len(content)
        tar.addfile(info,io.BytesIO(content))
    remote=base+'/bay_twin/'+run
    code,out=execute(client,'mkdir -p '+shlex.quote(remote))
    if code: raise RuntimeError(out)
    sftp=client.open_sftp()
    try: sftp.put(str(archive),remote+'/snapshot.tar.gz')
    finally: sftp.close()
    probe="""import tarfile,pathlib,hashlib,json,torch,importlib.metadata,subprocess
p=pathlib.Path.cwd()
with tarfile.open('snapshot.tar.gz') as t: t.extractall(p,filter='data')
m=json.loads((p/'snapshot_manifest.json').read_text())
for name,want in m['files_sha256'].items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==want,name
runtime={'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','optuna','lightgbm','scikit-learn']},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader'],text=True)}
(p/'runtime.json').write_text(json.dumps(runtime,indent=2))
print(json.dumps(runtime))
"""
    code,out=execute(client,f'cd {shlex.quote(remote)} && {shlex.quote(base+"/.venv/bin/python")} -c {shlex.quote(probe)}')
    if code: raise RuntimeError(out)
    print(out)
    cfg=json.loads((ROOT/'config/bay_twin_campaign.json').read_text())
    families=(['local_ridge','geo_tree','network_mlp','sensor_graph'] if smoke else cfg['families']) if campaign=='thermal' else (['oxygen_3h'] if smoke else ['oxygen_1h','oxygen_3h','oxygen_6h'])
    if campaign=='warning': families=['warning_3h'] if smoke else ['warning_1h','warning_3h','warning_6h']
    if campaign=='transfer': families=['transfer_temporal','transfer_held_site']
    if campaign=='field': families=['field_temporal','field_held_site']
    if campaign=='kernel': families=['kernel_temporal','kernel_held_site']
    if campaign=='boundary': families=['kernel_temporal','kernel_held_site']
    if campaign=='pretrain': families=['pretrain_temporal','pretrain_held_site']
    if campaign=='dynamics': families=['dynamics_temperature','dynamics_oxygen']
    if campaign=='reactive': families=['dynamics_oxygen']
    if campaign=='dynamics_tree': families=['dynamics_temperature','dynamics_oxygen']
    if campaign=='graph_geometry': families=['dynamics_temperature','dynamics_oxygen']
    if campaign=='mesh_features':families=['dynamics_temperature','dynamics_oxygen']
    if campaign=='forced_thermal':families=['dynamics_temperature']
    if campaign=='decision_calibration':families=['decision_calibration']
    if campaign=='seasonal':families=['dynamics_temperature','dynamics_oxygen']
    if campaign=='oxygen_memory':families=['oxygen_memory']
    if campaign=='thermal_memory':families=['thermal_memory']
    if campaign=='transfer_calibration':families=['transfer_calibration_dense','transfer_calibration_blocked']
    if campaign=='split_calibration':families=['split_calibration_temporal','split_calibration_held_site']
    if campaign=='resilience':families=['resilience_temporal','resilience_held_site']
    if campaign=='state_update':families=['state_update_temporal','state_update_held_site']
    if campaign=='observation_stress':families=['observation_stress_temporal','observation_stress_held_site']
    if campaign=='field_export':families=['field_export_temporal','field_export_held_site']
    if campaign=='robust':families=['robust_forecast']
    jobs=[]
    for family in families:
        if campaign=='oxygen_memory':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_oxygen_memory']
        elif campaign=='thermal_memory':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_thermal_memory']
        elif campaign=='transfer_calibration':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_transfer_calibration','--task',family.removeprefix('transfer_calibration_')]
        elif campaign=='split_calibration':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_split_calibration','--task',family.removeprefix('split_calibration_')]
        elif campaign=='resilience':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_resilience','--task',family.removeprefix('resilience_')]
        elif campaign=='state_update':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_state_update','--task',family.removeprefix('state_update_')]
        elif campaign=='observation_stress':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_observation_stress','--task',family.removeprefix('observation_stress_')]
        elif campaign=='field_export':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_field_export','--task',family.removeprefix('field_export_')]
        elif campaign=='robust':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_robust_forecast']
        elif campaign=='decision_calibration':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_decision_calibration']
        elif campaign=='thermal':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_twin_sweep','--family',family]
            if smoke: args+=['--quick','--task','site_outage','--hours','6']
        elif campaign in ('dynamics','reactive','dynamics_tree','graph_geometry','mesh_features','forced_thermal','seasonal'):
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_graph_dynamics','--variable',family.removeprefix('dynamics_')]
            if campaign=='seasonal':args[3]='src.experiments.bay_seasonal_validation'
            if campaign=='forced_thermal':args[3]='src.experiments.bay_forced_thermal'
            if campaign=='mesh_features':args[3]='src.experiments.bay_mesh_features'
            if campaign=='graph_geometry':args[3]='src.experiments.bay_graph_geometry'
            if campaign=='dynamics_tree':args[3]='src.experiments.bay_dynamics_tree'
            if campaign=='reactive':args+=['--reactive']
            if smoke:args+=['--quick']
        elif campaign=='transfer':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.oxygen_transfer_pilot','--task',family.removeprefix('transfer_')]
            if smoke: args+=['--quick']
        elif campaign=='field':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_profile_field','--task',family.removeprefix('field_')]
            if smoke: args+=['--quick']
        elif campaign in ('kernel','boundary'):
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_kernel_field','--task',family.removeprefix('kernel_')]
            if smoke: args+=['--quick']
            if campaign=='boundary': args+=['--boundary']
        elif campaign=='pretrain':
            args=[base+'/.venv/bin/python','-u','-m','src.experiments.bay_thermal_pretraining','--task',family.removeprefix('pretrain_')]
            if smoke:args+=['--quick']
        else:
            module='src.experiments.oxygen_warning_head' if campaign=='warning' else 'src.experiments.oxygen_trend_pilot'
            args=[base+'/.venv/bin/python','-u','-m',module,'--horizon',family.split('_')[1].rstrip('h')]
            if smoke: args+=['--quick']
        if '-m' in args:
            module=args[args.index('-m')+1]
            if '/' in module or '\\' in module:raise ValueError('Malformed module name: '+module)
            if ROOT/(module.replace('.','/')+'.py') not in files:raise ValueError('Module absent from snapshot: '+module)
        # One GPU worker per neural family; CPU families explicitly have no CUDA visibility.
        gpu={'dynamics_temperature':'0','dynamics_oxygen':'1','network_mlp':'0','sensor_graph':'1','field_temporal':'0','field_held_site':'1','pretrain_temporal':'0','pretrain_held_site':'1'}.get(family,'')
        if campaign in ('dynamics_tree','mesh_features','forced_thermal','seasonal'):gpu=''
        script=("import os,subprocess; env=os.environ.copy(); "
                f"env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES={gpu!r}); "
                f"log=open({family+'.log'!r},'ab'); p=subprocess.Popen({args!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)")
        code,out=execute(client,f'cd {shlex.quote(remote)} && {shlex.quote(base+"/.venv/bin/python")} -c {shlex.quote(script)}')
        if code: raise RuntimeError(out)
        jobs.append({'family':family,'pid':int(out.strip()),'gpu':gpu})
    record={'run_id':run,'remote_dir':remote,'jobs':jobs,'smoke_only':smoke,'campaign':campaign}
    (local/'transport.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record,indent=2))


def status(client,base,run,fetch):
    import re
    if not re.fullmatch(r'\d{8}T\d{6}Z_(thermal_(smoke|discovery)|(oxygen|warning|transfer|field|kernel|boundary|pretrain|dynamics|reactive|dynamics_tree|graph_geometry|mesh_features|forced_thermal|decision_calibration|seasonal|robust|field_export|observation_stress|state_update|resilience|split_calibration|transfer_calibration|thermal_memory|oxygen_memory)_(smoke|pilot))',run): raise ValueError('Bad run id')
    remote=base+'/bay_twin/'+run
    probe="""import pathlib,json
p=pathlib.Path('results')
print(json.dumps({'status':[json.loads(f.read_text()) for f in p.glob('status_*.json')],
'completed_studies':[json.loads(f.read_text())|{'study':f.parent.name} for f in p.glob('*/selection_summary.json')],
'trials_written':len(list(p.glob('*/trial_*.json')))},indent=2))
"""
    code,out=execute(client,f'cd {shlex.quote(remote)} && {shlex.quote(base+"/.venv/bin/python")} -c {shlex.quote(probe)}')
    if code: raise RuntimeError(out)
    snapshot=json.loads(out)
    print(json.dumps({'workers':[{k:v for k,v in s.items() if k!='result_sha256'} for s in snapshot['status']],
                      'completed_studies':len(snapshot['completed_studies']),
                      'trials_written':snapshot['trials_written']},indent=2))
    if fetch:
        local=ROOT/'reports/bay_twin/runs'/run; local.mkdir(parents=True,exist_ok=True)
        (local/'status_snapshot.json').write_text(out)
        bundle="""import pathlib,tarfile,hashlib,json
p=pathlib.Path.cwd()
paths=[p/'runtime.json',p/'snapshot_manifest.json']+[f for f in (p/'results').rglob('*') if f.is_file() and f.name.endswith(('.json','.csv.gz','.pt'))]
with tarfile.open('results_download.tar.gz','w:gz',compresslevel=1) as archive:
 for f in paths: archive.add(f,arcname=f.relative_to(p).as_posix())
f=p/'results_download.tar.gz'
print(json.dumps({'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'files':len(paths)}))
"""
        code,bundle_out=execute(client,f'cd {shlex.quote(remote)} && {shlex.quote(base+"/.venv/bin/python")} -c {shlex.quote(bundle)}')
        if code: raise RuntimeError(bundle_out)
        expected=json.loads(bundle_out)
        sftp=client.open_sftp()
        try:
            download=local/'results_download.tar.gz'
            sftp.get(remote+'/results_download.tar.gz',str(download),max_concurrent_prefetch_requests=64)
        finally: sftp.close()
        if hashlib.sha256(download.read_bytes()).hexdigest()!=expected['sha256']: raise ValueError('Download checksum mismatch')
        with tarfile.open(download) as archive: archive.extractall(local,filter='data')
        (local/'download_manifest.json').write_text(json.dumps(expected,indent=2))
        print('Downloaded and hash-verified',expected['files'],'files;',expected['bytes'],'bytes.')


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['launch','status','fetch']); p.add_argument('--smoke',action='store_true'); p.add_argument('--run-id'); p.add_argument('--campaign',choices=['thermal','oxygen','warning','transfer','field','kernel','boundary','pretrain','dynamics','reactive','dynamics_tree','graph_geometry','mesh_features','forced_thermal','decision_calibration','seasonal','robust','field_export','observation_stress','state_update','resilience','split_calibration','transfer_calibration','thermal_memory','oxygen_memory'],default='thermal'); a=p.parse_args()
    c,b=connect()
    try:
        if a.action=='launch': launch(c,b,a.smoke,a.campaign)
        else: status(c,b,a.run_id,a.action=='fetch')
    finally: c.close()

if __name__=='__main__': main()
