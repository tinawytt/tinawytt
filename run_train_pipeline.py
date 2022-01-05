#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os.path import exists
from datasets.example_dataset.create_splits import create_splits
from datasets.example_dataset.preprocessing import preprocess_data
# from experiments.UNetExperiment import UNetExperiment


if __name__ == "__main__":
    # c = get_config()

    # print("Executing: EPOCHS = {} / LEARNING RATE = {}".format(c.n_epochs, c.learning_rate))

    data_root_dir='/home/jovyan/main/'
    split_dir='/home/jovyan/main/'

    if not exists(os.path.join(os.path.join(data_root_dir,'BraTS2020_TrainingData'), 'preprocessed')):
        print('Preprocessing data. [STARTED]')
        preprocess_data(root_dir=os.path.join(data_root_dir, 'BraTS2020_TrainingData'), y_shape=64, z_shape=64)
        create_splits(output_dir=split_dir, image_dir=data_dir)
        print('Preprocessing data. [DONE]')
    else:
        print('The data has already been preprocessed. It will not be preprocessed again. Delete the folder to enforce it.')

#     exp = UNetExperiment(config=c, name=c.name, n_epochs=c.n_epochs,
#                          seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals(),
#                          # visdomlogger_kwargs={"auto_start": c.start_visdom},
#                          loggers={
#                              "visdom": ("visdom", {"auto_start": c.start_visdom})
#                          }
#                          )

#     exp.run()
#     exp.run_test(setup=False)
