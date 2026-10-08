"""Fixed adapted-encoder features, real CPU head updates, resume and packages."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'verification'))
import material_training_cycle as cycle
import material_workbench as workbench
from material_fixed_adapters import POLICY, adapter_sha256, apply_fixed_adapters
from frozen_dino_height import MODEL_SHA256, CODE_REVISION, state_sha256
from train_material_height import digest
from test_material_training_cycle import arguments, dataset_with_two_corners

class TinyContractEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared base tensors keep this CPU fixture small while every exact
        # official attention child and adapter pair remains independently used.
        qkv, projection = nn.Linear(768, 2304), nn.Linear(768, 768)
        self.blocks = nn.ModuleList()
        for _index in range(12):
            attention = nn.Module()
            attention.qkv, attention.proj = qkv, projection
            block = nn.Module()
            block.attn = attention
            self.blocks.append(block)
        self.requires_grad_(False).eval()

    def features(self, source):
        grid = torch.nn.functional.avg_pool2d(source.mean(1, keepdim=True), 16).repeat(1,768,1,1)
        tokens = grid.flatten(2).transpose(1,2)
        for block in self.blocks:
            tokens = tokens + 0.01*block.attn.proj(block.attn.qkv(tokens)[...,:768])
        return tokens.transpose(1,2).reshape_as(grid)


def adapted_checkpoint(tmp_path):
    head = cycle.MaterialMapHead(4,768,4,'height')
    with torch.no_grad():
        head.head.weight.fill_(0.02)
    state = {f'blocks.{i}.attn.{part}': {'lora_A':torch.full((8,768),0.001), 'lora_B':torch.full((out,8),0.002)}
        for i in range(12) for part,out in (('qkv',2304),('proj',768))}
    path=tmp_path/'adapted.pt'
    torch.save({'schema':cycle.ADAPTATION_SCHEMA,'variant':'lora','target':'height',
        'head_config':{'base_channels':4,'feature_channels':768,'projection_channels':4},
        'head_state':head.state_dict(),'adapter_state':state,'step':300,'encoder_size':28,
        'encoder':{'checkpoint_sha256':MODEL_SHA256,'official_meta_code_revision':CODE_REVISION}},path)
    return path,state


def test_actual_fixed_adapters_change_features_but_not_base_or_adapter_tensors(tmp_path):
    _path,state=adapted_checkpoint(tmp_path)
    torch.manual_seed(12)
    encoder=TinyContractEncoder()
    base=[(parameter,parameter.detach().clone()) for parameter in encoder.parameters()]
    source=torch.full((1,3,32,32),0.4)
    with torch.no_grad():
        original=encoder.features(source)
    modules=apply_fixed_adapters(encoder,state)
    with torch.no_grad():
        adapted=encoder.features(source)
    assert not torch.equal(original,adapted)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
    assert all(torch.equal(parameter,copy) for parameter,copy in base)
    actual={name:{field:getattr(module,field).detach() for field in ('lora_A','lora_B')} for name,module in modules.items()}
    assert adapter_sha256(actual)==adapter_sha256(state)


def test_real_head_refinement_retains_adapters_through_resume_and_package(tmp_path):
    path,state=adapted_checkpoint(tmp_path)
    expected=digest(path)
    encoders=[]
    def loader(_model,_code,device):
        torch.manual_seed(11)
        encoder=TinyContractEncoder().to(device)
        encoders.append(encoder)
        return encoder,{'checkpoint_sha256':MODEL_SHA256,'official_meta_code_revision':CODE_REVISION}
    @torch.no_grad()
    def features(encoder,source,_size):
        return encoder.features(source).detach()
    settings=arguments(tmp_path,warm_start=path,warm_start_sha256=expected,prediction_limit=0)
    report=cycle.run_train(settings,encoder_loader=loader,feature_extractor=features)
    assert report['completed_steps']==4 and report['encoder_refinement_policy']==POLICY
    assert report['fixed_adapters_verified_unchanged'] and not report['encoder_adapters_trained']
    final=torch.load(settings.output/'checkpoint.final.pt',weights_only=True)
    assert final['variant']=='lora' and adapter_sha256(final['adapter_state'])==adapter_sha256(state)
    assert report['initial_head_state_sha256']!=state_sha256(final['head_state'])
    assert final['identity']['fixed_encoder_adapters_sha256']==adapter_sha256(state)
    assert all(not parameter.requires_grad and parameter.grad is None for encoder in encoders for parameter in encoder.parameters())
    extended=arguments(tmp_path,output=tmp_path/'resumed',resume_checkpoint=settings.output/'checkpoint.latest.pt',updates_per_crop=3,prediction_limit=0)
    resumed=cycle.run_train(extended,encoder_loader=loader,feature_extractor=features)
    assert resumed['started_from_step']==4 and resumed['completed_steps']==6
    package=workbench.package(SimpleNamespace(checkpoint=extended.output/'checkpoint.final.pt',expected_sha256=None,output=tmp_path/'package'))
    packed,_=workbench.checkpoint_snapshot(Path(package['checkpoint_path']))
    assert packed['variant']=='lora' and adapter_sha256(packed['adapter_state'])==adapter_sha256(state)
    info=workbench.checkpoint_info(SimpleNamespace(checkpoint=Path(package['checkpoint_path']),expected_sha256=package['checkpoint_sha256']))
    assert not info['supports_training_warm_start'] and info['refinement_policy']==POLICY
    assert info['retired'] and not info['production_eligible']
    assert digest(path)==expected


def test_malformed_adapters_and_changed_resume_identity_are_refused(tmp_path):
    path,state=adapted_checkpoint(tmp_path)
    payload=torch.load(path,weights_only=True)
    payload['adapter_state']['blocks.0.attn.qkv']['lora_B']=torch.zeros(1)
    torch.save(payload,path)
    with pytest.raises(ValueError,match='LoRA tensor'):
        cycle.load_cycle_head(path,torch.device('cpu'))
    identity={'training_sample_ids':['one'],'fixed_encoder_adapters_sha256':adapter_sha256(state)}
    checkpoint={'schema':cycle.SCHEMA,'identity':copy.deepcopy(identity),'head_config':{},'schedule':[0],'step':0,'per_crop_update_counts':[0],'optimizer_state':{}}
    identity['fixed_encoder_adapters_sha256']='changed'
    with pytest.raises(ValueError,match='Resume identity'):
        cycle.validate_resume(checkpoint,identity,[0],{})


def test_unreviewed_unselected_material_does_not_block_approved_selection(tmp_path):
    settings=arguments(tmp_path,allow_unreviewed=False)
    dataset=settings.dataset
    index=json.loads((dataset/'dataset.json').read_text())
    for entry in index['samples']:
        if entry['material_id']!='white_stucco_02':
            entry['status']='unreviewed'
            path=dataset/entry['path']/'sample.json'
            metadata=json.loads(path.read_text());metadata['status']='unreviewed';path.write_text(json.dumps(metadata))
        else:
            entry['status']='approved'
            path=dataset/entry['path']/'sample.json'
            metadata=json.loads(path.read_text());metadata['status']='approved';path.write_text(json.dumps(metadata))
    (dataset/'dataset.json').write_text(json.dumps(index))
    train,validation,_bindings,_manifests=cycle.select_samples(settings)
    assert len(train)==2 and len(validation)==1
    settings.materials=['broken_brick_wall']
    with pytest.raises(ValueError,match='Selected samples'):
        cycle.select_samples(settings)

def test_frozen_warm_start_and_resume_keep_nondefault_contextual_size(tmp_path):
    from test_material_training_cycle import tiny_encoder_loader, tiny_features
    settings=arguments(tmp_path,prediction_limit=0)
    source=tmp_path/'frozen280.pt'
    head=cycle.MaterialMapHead(4,768,4,'height')
    torch.save({'schema':cycle.SCHEMA,'variant':'frozen','head_config':{'base_channels':4,'feature_channels':768,'projection_channels':4,'target':'height'},
        'head_state':head.state_dict(),'target':'height','step':8,'encoder_size':280,
        'encoder':{'checkpoint_sha256':MODEL_SHA256,'official_meta_code_revision':CODE_REVISION}},source)
    sizes=[]
    @torch.no_grad()
    def capture(encoder,photo,size):
        sizes.append(size)
        return tiny_features(encoder,photo,size)
    settings.warm_start=source
    settings.encoder_size=518
    report=cycle.run_train(settings,encoder_loader=tiny_encoder_loader,feature_extractor=capture)
    assert set(sizes)=={280} and report['identity']['encoder_size']==280
    assert report['warm_start']['encoder_size_policy']=='selected_checkpoint_size_preserved'
    sizes.clear()
    extended=arguments(tmp_path,output=tmp_path/'extended280',resume_checkpoint=settings.output/'checkpoint.latest.pt',updates_per_crop=3,prediction_limit=0,encoder_size=518)
    resumed=cycle.run_train(extended,encoder_loader=tiny_encoder_loader,feature_extractor=capture)
    assert set(sizes)=={280} and resumed['started_from_step']==4 and resumed['completed_steps']==6
    assert torch.load(extended.output/'checkpoint.final.pt',weights_only=True)['encoder_size']==280
