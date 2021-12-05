# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os

task_conf_mapping = dict()
exp_dir = "lite_sac"

if __name__ == "__main__":
    method, task, cluster = sys.argv[1:4]
    repeats = 3
    if len(sys.argv) > 4:
        repeats = sys.argv[4]
    print("Method: %s, task: %s, cluster: %s, repeats: %s" %
          (method, task, cluster, repeats))
    for i in range(int(repeats)):
        print("Launching repeat %d" % i)
        if task in task_conf_mapping:
            task_conf = task_conf_mapping[task]
        else:
            task_conf = task
        os.system(
            f"python -m cluster_train --job_name {method}_{task}"
            f" --conf {exp_dir}/{method}/{method}_{task_conf}_conf.py"
            f" --search_config {exp_dir}/{task}.json --cluster {cluster}")
