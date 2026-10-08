"""CPU archive/source identity and bounded official-URL contracts."""
from pathlib import Path
import hashlib,json,struct,sys,zipfile
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_ambientcg_package as audit
from material_dataset import write_png

def fixture(tmp_path,height_bits=16):
    source=tmp_path/'sources';source.mkdir();files=[]
    for name,bits,channels in [('Color',8,4),('Displacement',height_bits,1),('NormalGL',16,4),('Roughness',8,1)]:
        p=source/f'Snow013_4K-PNG_{name}.png';data=np.arange(8*8*channels,dtype=np.uint16 if bits==16 else np.uint8).reshape(8,8,channels);write_png(p,data);files.append(p)
    archive=tmp_path/'Snow013_4K-PNG.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for p in files:z.write(p,p.name)
    return source,archive

def metadata():
    return {'foundAssets':[{'assetId':'Snow013','dataType':'Material','creationMethod':'PBRPhotogrammetry','downloadFolders':{'default':{'downloadFiletypeCategories':{'zip':{'downloads':[{'fileName':'Snow013_4K-PNG.zip','downloadLink':'https://ambientcg.com/get?file=Snow013_4K-PNG.zip','size':123}]}}}}}]}

def test_full_original_hashes_headers_and_source_bytes_preserved(tmp_path):
    source,archive=fixture(tmp_path);before={p.name:p.read_bytes() for p in source.iterdir()}
    r=audit.compare_package(archive,source,'Snow013','4K',archive.stat().st_size)
    assert len(r['files'])==4 and all(x['exact_full_file_match'] for x in r['files'])
    assert r['package']['sha256']==hashlib.sha256(archive.read_bytes()).hexdigest()
    assert r['package']['sha1_matches_provider_header'] is None
    assert {x['role']:x['source_sample_bits'] for x in r['files']}=={'input':8,'height':16,'normal':16,'roughness':8}
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}

def test_actual_provider_digest_optional_and_checked_if_published(tmp_path):
    source,archive=fixture(tmp_path);expected=hashlib.sha1(archive.read_bytes()).hexdigest()
    assert audit.compare_package(archive,source,'Snow013','4K',archive.stat().st_size,expected)['package']['sha1_matches_provider_header']
    assert audit.compare_package(archive,source,'Snow013','4K',archive.stat().st_size,'none')['package']['provider_sha1_available'] is False
    with pytest.raises(ValueError,match='SHA1 differs'):audit.compare_package(archive,source,'Snow013','4K',archive.stat().st_size,'0'*40)

def test_changed_source_and_wrong_archive_bytes_rejected(tmp_path):
    source,archive=fixture(tmp_path)
    with pytest.raises(ValueError,match='size'):audit.compare_package(archive,source,'Snow013','4K',archive.stat().st_size+1)
    (source/'Snow013_4K-PNG_Roughness.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs'):audit.compare_package(archive,source,'Snow013','4K',archive.stat().st_size)

def test_missing_duplicate_and_8bit_height_rejected(tmp_path):
    source,archive=fixture(tmp_path);(source/'Snow013_4K-PNG_Roughness.png').unlink()
    with pytest.raises(ValueError,match='all four'):audit.compare_package(archive,source,'Snow013','4K',archive.stat().st_size)
    (tmp_path/'next').mkdir()
    source2,archive2=fixture(tmp_path/'next')
    with pytest.warns(UserWarning,match='Duplicate name'):
        with zipfile.ZipFile(archive2,'a') as z:z.write(source2/'Snow013_4K-PNG_Color.png','Snow013_4K-PNG_Color.png')
    with pytest.raises(ValueError,match='duplicate'):audit.compare_package(archive2,source2,'Snow013','4K',archive2.stat().st_size)

def test_official_archive_identity_url_size_and_method(tmp_path):
    asset,p=audit.choose_package(metadata(),'Snow013','4K');assert asset['creationMethod']=='PBRPhotogrammetry' and p['size']==123
    for bad in ('Snow014','snow013'):
        with pytest.raises(ValueError,match='requested material'):audit.choose_package(metadata(),bad,'4K')
    changed=metadata();changed['foundAssets'][0]['downloadFolders']['default']['downloadFiletypeCategories']['zip']['downloads'][0]['downloadLink']='https://evil.example/get?file=Snow013_4K-PNG.zip'
    with pytest.raises(ValueError,match='URL'):audit.choose_package(changed,'Snow013','4K')

@pytest.mark.parametrize('url',('http://ambientcg.com/get?file=Snow013_4K-PNG.zip','https://ambientcg.com.evil.test/get?file=Snow013_4K-PNG.zip','https://ambientcg.com/get?file=Snow014_4K-PNG.zip','https://acg-download.struffelproductions.com/other/Snow013_4K-PNG.zip'))
def test_untrusted_download_redirect_not_permitted(url):
    assert not audit.official_url(url,'Snow013_4K-PNG.zip')

def test_allowed_official_api_archive_and_storage_urls():
    assert audit.official_url('https://ambientcg.com/api/v2/full_json?include=downloadData&id=Snow013')
    assert audit.official_url(audit.LICENSE_URL)
    assert audit.official_url('https://ambientcg.com/get?file=Snow013_4K-PNG.zip','Snow013_4K-PNG.zip')
    assert audit.official_url('https://acg-download.struffelproductions.com/file/ambientCG-Web/download/Snow013_code/Snow013_4K-PNG.zip','Snow013_4K-PNG.zip')


def test_audit_temp_download_cleanup_and_original_names(tmp_path,monkeypatch):
    source,archive=fixture(tmp_path);before={p.name:p.read_bytes() for p in source.iterdir()};temp_paths=[]
    raw=json.dumps(metadata()).encode();r=json.loads(raw);r['foundAssets'][0]['downloadFolders']['default']['downloadFiletypeCategories']['zip']['downloads'][0]['size']=archive.stat().st_size;raw=json.dumps(r).encode()
    def download(url,path,max_bytes,filename=None):
        temp_paths.append(path)
        path.write_bytes(archive.read_bytes() if filename else (b'Creative Commons CC0 1.0 Universal License' if url==audit.LICENSE_URL else raw))
        return {'headers':{},'completed_utc':'test'}
    monkeypatch.setattr(audit,'bounded_download',download)
    report=audit.audit('Snow013',source,tmp_path/'report')
    assert report['creation_method']['id']=='PBRPhotogrammetry'
    assert not report['verification_download_retained'] and all(not p.exists() for p in temp_paths)
    assert all(x['source_filename']==x['archive_member'] for x in report['files'])
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}


def test_8bit_displacement_rejected_without_upconversion(tmp_path):
    source,archive=fixture(tmp_path,height_bits=8)
    with pytest.raises(ValueError,match='original16-bit'):
        audit.compare_package(archive,source,'Snow013','4K',archive.stat().st_size)
