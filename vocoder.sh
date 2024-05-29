#!/bin/bash
cp ../EEG-AudioTransformer/test_mel_files/* ./test_mel_files/

python inference_e2e.py --checkpoint_file ./cp_hifigan/g_00185000