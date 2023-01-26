# Copyright (c) Ramy Mounir.
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

from torch.utils.tensorboard import SummaryWriter

def checkdir(path, reset = True):
    import os, shutil
    if os.path.exists(path):
        if reset:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def get_writer(path):

    checkdir(path)

    writer = SummaryWriter(path)
    writer.flush()

    return writer


class TBWriter(object):

    def __init__(self, writer, data_type, tag, mul = 1, add = 0, fps = 4):

        self.step = 0
        self.mul = mul
        self.add = add
        self.fps = fps

        self.writer = writer
        self.type = data_type
        self.tag = tag

    def __call__(self, data, step = None, flush = False, metadata=None, label_img=None):

        counter = step if step != None else self.step*self.mul+self.add

        if self.type == 'scalar':
            self.writer.add_scalar(self.tag, data, global_step = counter)
        elif self.type == 'scalars':
            self.writer.add_scalars(self.tag, data, global_step = counter)
        elif self.type == 'image':
            self.writer.add_image(self.tag, data, global_step = counter)
        elif self.type == 'video':
            self.writer.add_video(self.tag, data, global_step = counter, fps = self.fps)
        elif self.type == 'figure':
            self.writer.add_figure(self.tag, data, global_step = counter)
        elif self.type == 'text':
            self.writer.add_text(self.tag, data, global_step = counter)
        elif self.type == 'histogram':
            self.writer.add_histogram(self.tag, data, global_step = counter)
        elif self.type == 'embedding':
            self.writer.add_embedding(mat=data, metadata=metadata, label_img=label_img, global_step=counter, tag=self.tag)

        self.step += 1

        if flush:
            self.writer.flush()
