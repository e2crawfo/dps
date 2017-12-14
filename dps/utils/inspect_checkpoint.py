# Original work Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modified work Copyright 2017 Eric Crawford.
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
# ==============================================================================
from tensorflow.python import pywrap_tensorflow


def get_tensors_from_checkpoint_file(file_name):
    """ Based on code in tensorflow's inspect_checkpoint.py script. """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return {key: reader.get_tensor(key) for key in var_to_shape_map}

    except Exception as e:    # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.    Try removing the '.' and extension.    Try:
inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))
