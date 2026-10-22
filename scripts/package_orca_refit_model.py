"""Relocate mesh assets without changing compiled physics or historical artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    tree=ET.parse(args.model)
    assert not list(tree.iter('include'))
    args.output.mkdir(parents=True,exist_ok=False)
    assets=args.output/'assets';assets.mkdir()
    hashes={}
    for element in tree.iter():
        if 'file' not in element.attrib:
            continue
        assert element.tag=='mesh', 'Only explicit mesh assets are supported'
        source=Path(element.get('file'))
        if not source.is_absolute():
            source=args.model.parent/source
        relative=Path('assets')/(element.get('name')+source.suffix)
        target=args.output/relative
        shutil.copyfile(source,target)
        hashes[str(relative)]=digest(target)
        element.set('file',str(relative))
    xml=args.output/'cube.xml'
    ET.indent(tree);tree.write(xml,encoding='unicode')
    original=mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    packed=mujoco.MjModel.from_xml_path(str(xml.resolve()))
    compared=[]
    for key in dir(original):
        # Private buffer-size tables also include variable-length asset paths.
        if key.startswith('_') or 'path' in key:
            continue
        value=getattr(original,key)
        if isinstance(value,np.ndarray):
            np.testing.assert_array_equal(value,getattr(packed,key),err_msg=key)
            compared.append(key)
    for key in ('nq','nv','nu','nbody','ngeom','nmesh','njnt','nmocap'):
        assert getattr(original,key)==getattr(packed,key),key
    options=[]
    for key in dir(original.opt):
        if key.startswith('_'):
            continue
        value=getattr(original.opt,key)
        if isinstance(value,(np.ndarray,int,float)):
            np.testing.assert_array_equal(value,getattr(packed.opt,key),err_msg='opt.'+key)
            options.append(key)
    result=dict(source_model=str(args.model.resolve()),source_model_sha256=digest(args.model),
                packed_model_sha256=digest(xml),assets=hashes,physics_arrays_match=True,
                compared_model_arrays=compared,compared_options=options,
                excluded_metadata='Private buffers and asset file path/address arrays',
                note='Asset relocation only. No new training or fabricated validation results. '
                     'Historical experiments retain their original model and gate paths.')
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(assets=len(hashes),matched_arrays=len(compared),matched_options=len(options),
                         model=str(xml),physics_arrays_match=True)))


if __name__=='__main__':
    main()
