#!/usr/bin/env python3
"""Verify local material PNGs against one official ambientCG ZIP package.

Original source pixels and filenames are never written. Downloads are bounded,
read only as ZIP members, and removed with the owned temporary directory. The
JSON report retains full archive/member/source checksums and actual PNG headers.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from typing import Any
from urllib.parse import parse_qs, quote, urljoin, urlparse
import zipfile

SCHEMA = 'ipde-material-package-audit-v1'
LICENSE_URL = 'https://docs.ambientcg.com/license/'
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
PNG = b'\x89PNG\r\n\x1a\n'
ROLE_NAMES = {'Color':'input','Displacement':'height','NormalGL':'normal','NormalDX':'normal','Roughness':'roughness'}

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def file_identity(path: Path) -> dict[str,int]:
    s=path.stat(); return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns}

def digest_stream(stream) -> tuple[str,str,int,bytes]:
    sha256,sha1,size,prefix=hashlib.sha256(),hashlib.sha1(),0,b''
    for chunk in iter(lambda:stream.read(4*1024*1024),b''):
        sha256.update(chunk);sha1.update(chunk);size+=len(chunk)
        if len(prefix)<33:prefix=(prefix+chunk)[:33]
    return sha256.hexdigest(),sha1.hexdigest(),size,prefix

def png_header(prefix: bytes) -> dict[str,Any]:
    if len(prefix)!=33 or prefix[:8]!=PNG or prefix[12:16]!=b'IHDR':
        raise ValueError('Expected native PNG IHDR')
    w,h,bits,color,compression,filtering,interlace=struct.unpack('>IIBBBBB',prefix[16:29])
    channels={0:1,2:3,4:2,6:4}.get(color)
    if bits not in (8,16) or channels is None or w<1 or h<1 or compression or filtering:
        raise ValueError('Unsupported native PNG sample layout')
    return {'width':w,'height':h,'sample_bits':bits,'channels':channels,'png_color_type':color,'interlace':interlace}

def official_url(url: str, filename: str | None = None) -> bool:
    p=urlparse(url)
    if p.scheme!='https' or p.username or p.password or p.port not in (None,443):return False
    if filename is None:
        return (p.hostname=='ambientcg.com' and p.path=='/api/v2/full_json') or url==LICENSE_URL
    if p.hostname=='ambientcg.com':return p.path=='/get' and parse_qs(p.query)=={'file':[filename]}
    return p.hostname in ('acg-download.struffelproductions.com','f003.backblazeb2.com') and p.path.startswith('/file/ambientCG-Web/download/') and Path(p.path).name==filename

def headers_dict(raw: str) -> dict[str,str]:
    result={}
    for line in raw.splitlines():
        if line.startswith('HTTP/'):result={}
        elif ':' in line:
            k,v=line.split(':',1); result[k.casefold()]=v.strip()
    return result

def bounded_download(url: str, path: Path, max_bytes: int, filename: str | None = None) -> dict[str,Any]:
    if max_bytes<1 or not official_url(url,filename):raise ValueError('Untrusted URL or invalid download budget')
    first=url; chain=[]
    for _ in range(4):
        if not official_url(url,filename):raise ValueError('Untrusted official download redirect')
        header_path=path.with_name(path.name+'.headers')
        command=['curl','--fail','--silent','--show-error','--connect-timeout','10','--max-time','180','--max-filesize',str(max_bytes),'--proto','=https','--user-agent','TextureStudio/0.1 (local source provenance)','--dump-header',str(header_path),'--output',str(path),'--write-out','%{http_code}',url]
        done=subprocess.run(command,capture_output=True,text=True,timeout=190)
        if done.returncode:raise ValueError(done.stderr.strip() or f'curl exited {done.returncode}')
        raw=header_path.read_text();headers=headers_dict(raw);status=int(done.stdout.strip());chain.append({'url':url,'status':status,'headers_sha256':hashlib.sha256(raw.encode()).hexdigest()})
        if path.stat().st_size>max_bytes:raise ValueError('Download exceeded byte budget')
        if status in (301,302,303,307,308):
            if 'location' not in headers:raise ValueError('Redirect lacks Location')
            url=urljoin(url,headers['location']);continue
        if status!=200:raise ValueError(f'Expected HTTP200, received {status}')
        return {'url':first,'final_url':url,'redirect_chain':chain,'headers':headers,'completed_utc':now()}
    raise ValueError('Too many official download redirects')

def choose_package(payload: dict, asset_id: str, resolution: str) -> tuple[dict,dict]:
    assets=payload.get('foundAssets',[])
    matching=[x for x in assets if x.get('assetId')==asset_id and x.get('dataType')=='Material']
    if len(matching)!=1:raise ValueError('Official API must identify exactly the requested material')
    asset=matching[0];filename=f'{asset_id}_{resolution}-PNG.zip';found=[]
    for folder in asset.get('downloadFolders',{}).values():
        for category in folder.get('downloadFiletypeCategories',{}).values():
            for item in category.get('downloads',[]):
                if item.get('fileName')==filename:found.append(item)
    if len(found)!=1:raise ValueError('Expected one native PNG package at the requested resolution')
    package=found[0]
    if type(package.get('size')) is not int or not 0<package['size']<=MAX_ARCHIVE_BYTES or not official_url(package.get('downloadLink',''),filename):
        raise ValueError('Package URL or published byte count is invalid')
    return asset,package

def compare_package(archive: Path, sources: Path, asset_id: str, resolution: str, expected_bytes: int, expected_provider_sha1: str | None = None) -> dict:
    before_archive=file_identity(archive)
    with archive.open('rb') as f:archive_sha,archive_sha1,archive_bytes,_=digest_stream(f)
    if archive_bytes!=expected_bytes or before_archive!=file_identity(archive):raise ValueError('Official archive size changed or differs from metadata')
    valid_provider_sha1=bool(expected_provider_sha1 and re.fullmatch(r'[a-fA-F0-9]{40}',expected_provider_sha1))
    if valid_provider_sha1 and archive_sha1!=expected_provider_sha1.casefold():raise ValueError('Provider archive SHA1 differs')
    files=[];prefix=f'{asset_id}_{resolution}-PNG_';local=[]
    for path in sorted(sources.glob(prefix+'*.png')):
        suffix=path.name[len(prefix):-4]
        if suffix in ROLE_NAMES:local.append((path,suffix))
    by_role={}
    for path,suffix in local:
        role=ROLE_NAMES[suffix]
        if role=='normal' and role in by_role:
            if suffix=='NormalGL':by_role[role]=(path,suffix)
            continue
        if role in by_role:raise ValueError('Duplicate source role')
        by_role[role]=(path,suffix)
    if set(by_role)!={'input','height','normal','roughness'}:raise ValueError('Local native package must include all four standard maps')
    with zipfile.ZipFile(archive) as z:
        counts={name:sum(i.filename==name for i in z.infolist()) for name in [p.name for p,_ in by_role.values()]}
        for role,(path,suffix) in sorted(by_role.items()):
            if counts[path.name]!=1:raise ValueError('Missing or duplicate exact ZIP member')
            member=z.getinfo(path.name)
            if member.is_dir() or member.flag_bits&1 or member.file_size>MAX_ARCHIVE_BYTES:raise ValueError('Invalid bounded ZIP member')
            original=file_identity(path)
            with path.open('rb') as f:source_sha,_,source_bytes,prefix_bytes=digest_stream(f)
            with z.open(member) as f:member_sha,_,member_bytes,member_prefix=digest_stream(f)
            current=file_identity(path);
            if original!=current or source_bytes!=member_bytes or source_sha!=member_sha or prefix_bytes!=member_prefix:raise ValueError(f'Source differs from official package member: {path.name}')
            head=png_header(prefix_bytes)
            if role=='height' and head['sample_bits']!=16:raise ValueError('High-detail displacement requires original16-bit samples')
            files.append({'source_filename':path.name,'source_path':str(path.resolve()),'role':role,'source_sha256':source_sha,'source_bytes':source_bytes,'archive_member':path.name,'member_sha256':member_sha,'member_bytes':member_bytes,'exact_full_file_match':True,'source_stable':True,'source_stat':current,'restoration_transform':'None; original source filename and bytes identical','source_sample_bits':head['sample_bits'],'source_pixel_dimensions':[head['width'],head['height']],'actual_png_header':head})
    dims={tuple(x['source_pixel_dimensions']) for x in files}
    if len(dims)!=1:raise ValueError('Paired native map dimensions differ')
    if before_archive!=file_identity(archive):raise ValueError('Archive changed during member verification')
    return {'package':{'filename':archive.name,'file_bytes':archive_bytes,'sha256':archive_sha,'sha1':archive_sha1,'provider_sha1':expected_provider_sha1,'sha1_matches_provider_header':True if valid_provider_sha1 else None,'provider_sha1_available':valid_provider_sha1},'files':files}

def audit(asset_id: str, sources: Path, output: Path, resolution: str='4K', max_archive_bytes: int=MAX_ARCHIVE_BYTES) -> dict:
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*',asset_id) or resolution not in ('1K','2K','4K','8K') or not 0<max_archive_bytes<=MAX_ARCHIVE_BYTES:raise ValueError('Invalid identity, resolution or archive budget')
    output.mkdir(parents=True,exist_ok=False);started=now();api_url='https://ambientcg.com/api/v2/full_json?include=downloadData&id='+quote(asset_id,safe='')
    try:
        with tempfile.TemporaryDirectory(prefix='ambientcg-verify-') as td:
            temporary=Path(td);api=temporary/'api.json';license_path=temporary/'license.html'
            api_transport=bounded_download(api_url,api,MAX_METADATA_BYTES);license_transport=bounded_download(LICENSE_URL,license_path,MAX_METADATA_BYTES)
            api_raw=api.read_bytes();license_raw=license_path.read_bytes();license_text=license_raw.decode('utf-8')
            if 'Creative Commons CC0 1.0 Universal License' not in ' '.join(re.sub(r'<[^>]+>',' ',license_text).split()):raise ValueError('Current official license snapshot does not establish CC0')
            payload=json.loads(api_raw);asset,package=choose_package(payload,asset_id,resolution)
            if package['size']>max_archive_bytes:raise ValueError('Package exceeds explicitly bounded download budget')
            archive=temporary/package['fileName'];transport=bounded_download(package['downloadLink'],archive,package['size'],package['fileName'])
            verified=compare_package(archive,sources,asset_id,resolution,package['size'],transport['headers'].get('x-bz-content-sha1'))
            verified['package'].update(url=package['downloadLink'],download_transport=transport,published_api_bytes=package['size'])
            (output/'official-metadata.json').write_bytes(api_raw);(output/'license.html').write_bytes(license_raw)
            creation={'id':asset.get('creationMethod'),'name':asset.get('creationMethodName'),'description':asset.get('creationMethodDescription'),'scope':'This asset only; not inferred for other ambientCG materials'}
            if creation['id']=='PBRPhotogrammetry':creation['asset_page_label']='Surface Photogrammetry'
            report={'schema':SCHEMA,'provider':'ambientCG','material_id':asset_id.lower(),'asset_id':asset_id,'asset_url':'https://ambientcg.com/view?id='+asset_id,'started_utc':started,'completed_utc':now(),'api_provenance':{'api_url':api_url,'snapshot_sha256':hashlib.sha256(api_raw).hexdigest(),'snapshot_path':str((output/'official-metadata.json').resolve()),'transport':api_transport},'creation_method':creation,'physical_width_meters':asset.get('dimensionX') or None,'physical_height_meters':asset.get('dimensionY') or None,'license':{'spdx':'CC0-1.0','url':LICENSE_URL,'snapshot_sha256':hashlib.sha256(license_raw).hexdigest(),'scope':'Official license applies to downloadable asset files','retrieved_utc':license_transport['completed_utc']},**verified,'source_pixels_modified':False,'source_files_modified':False,'original_files_deleted':False,'verification_download_retained':False,'temporary_archive_cleanup':'Owned TemporaryDirectory removed after verification; source paths never deleted'}
        with (output/(asset_id.lower()+'-verified-package.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        return report
    except (OSError,ValueError,subprocess.TimeoutExpired,zipfile.BadZipFile) as error:
        with (output/'failed-audit.json').open('x') as f:json.dump({'asset_id':asset_id,'started_utc':started,'failed_utc':now(),'error':str(error),'source_files_modified':False,'original_files_deleted':False,'verification_download_retained':False},f,indent=2);f.write('\n')
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-id',required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',default='4K',choices=('1K','2K','4K','8K'));p.add_argument('--max-archive-bytes',type=int,default=MAX_ARCHIVE_BYTES);args=p.parse_args()
    r=audit(args.asset_id,args.sources,args.output,args.resolution,args.max_archive_bytes);print(json.dumps({'asset_id':r['asset_id'],'verified_files':len(r['files']),'archive_bytes':r['package']['file_bytes'],'temporary_archive_retained':False},indent=2))
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,subprocess.TimeoutExpired,zipfile.BadZipFile) as e:raise SystemExit(f'error: {e}')
