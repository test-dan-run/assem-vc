'''
Function to write from nemo json file into assem filelist.txt
'''
import os
import json
import random
from typing import Dict, List

random.seed(42)

def load_nemo_manifest(path: str) -> List[Dict]:
    '''
    Loads NeMo manifest format and outputs a list of dictionaries
    each dictionary denoting a row of data
    '''
    manifest = []
    with open(path, 'r', encoding='utf8') as input_file:
        lines = input_file.readlines()
        for line in lines:
            manifest.append(json.loads(line))
    return manifest

def convert_nemo_to_assem(manifest: List[Dict]) -> List[str]:
    '''
    Takes in a loaded NeMo manifest (list) and outputs in assem format
    [{audio_filepath: xxx, text: yyy, speaker_id: zzz},] -> ['xxx|yyy|zzz',]
    '''
    filelist = []
    for entry in manifest:
        filelist.append(
            f"{entry['audio_filepath']}|{entry['text']}|{entry['speaker_id']}"
        )
    return filelist

def extract_speakers(manifest: List[Dict], output_path: str) -> None:
    '''
    extracts the list of speakers in the given manifest
    '''
    speaker_list = list(set([x['speaker_id'] for x in manifest]))
    with open(output_path, 'w', encoding='utf8') as output_file:
        output_file.write(json.dumps(speaker_list))

def write_assem(filelist: List[str], output_dir: str, split: float = 0.0) -> None:
    '''
    Writes list of assem entries into metadata file
    '''
    if split:
        train_split_path = os.path.join(output_dir, 'vc_train_filelist.txt')
        dev_split_path = os.path.join(output_dir, 'vc_dev_filelist.txt')
        with open(train_split_path, 'w', encoding='utf8') as train_fw, \
            open(dev_split_path, 'w', encoding='utf8') as dev_fw:
            for entry in filelist:
                if random.random() < split:
                    train_fw.write(entry+'\n')
                else:
                    dev_fw.write(entry+'\n')
    else:
        path = os.path.join(output_dir, 'filelist.txt')
        with open(path, 'w', encoding='utf8') as output_path:
            for entry in filelist:
                output_path.write(entry+'\n')


if __name__ == '__main__':

    NEMO_MANIFEST_PATH = '/datasets/nsc/part1/test/test_manifest.json'
    OUTPUT_DIR = '/datasets/nsc/part1/test'
    SPEAKER_LIST_PATH = '/datasets/nsc/part1/test/speaker_list.txt'

    nemo_list = load_nemo_manifest(NEMO_MANIFEST_PATH)
    assem_list = convert_nemo_to_assem(nemo_list)
    extract_speakers(nemo_list, SPEAKER_LIST_PATH)
    write_assem(assem_list, OUTPUT_DIR, split=0.8)
