# taken from Github, yjkimnada/ns_decoding
# Original Author: Young Joon Kim

# Original License
'''
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import torch.optim as torch_optim
import matplotlib.pyplot as plt


class ThinDatasetWrapper(torch_data.Dataset):

    def __init__(self, images: np.ndarray, spikes: np.ndarray):
        self.images = images
        self.spikes = spikes

    def __getitem__(self, index):
        return self.images[index, ...], self.spikes[index, ...]

    def __len__(self):
        return self.images.shape[0]


def plot_examples(batch_ground_truth: np.ndarray,
                  batch_decoder_output: np.ndarray):
    '''
    For every image in the example batch,
    :param forward_intermediates:
    :param linear_model:
    :param batched_observed_spikes:
    :return:
    '''

    batch_size = batch_ground_truth.shape[0]

    fig, axes = plt.subplots(batch_size, 2, figsize=(2 * 5, 5 * batch_size))
    for row in range(batch_size):
        ax = axes[row, 0]
        ax.imshow(batch_ground_truth[row, ...], vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

        ax = axes[row, 1]
        ax.imshow(batch_decoder_output[row, ...], vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

    return fig


def eval_test_loss_decoder(parallel_decoder: 'Parallel_NN_Decoder',
                           hpf_test_dataloader: torch_data.DataLoader,
                           loss_callable,
                           device: torch.device) -> float:

    loss_acc = []
    with torch.no_grad():
        for it, (hpf_np, spikes_np) in enumerate(hpf_test_dataloader):

            # shape (batch, height, width)
            hpf_torch= torch.tensor(hpf_np, dtype=torch.float32, device=device)
            batch, height, width = hpf_torch.shape

            # shape (batch, n_cells, n_timebins)
            spikes_torch = torch.tensor(spikes_np, dtype=torch.float32, device=device)

            output_flat = parallel_decoder(spikes_torch).reshape(batch, height, width)
            loss = loss_callable(output_flat, hpf_torch).detach().cpu().numpy()

            loss_acc.append(np.mean(loss))
    return np.mean(loss_acc)


def train_parallel_NN_decoder(parallel_decoder: 'Parallel_NN_Decoder',
                              hpf_dataloader: torch_data.DataLoader,
                              test_hpf_dataloader: torch_data.DataLoader,
                              loss_callable,
                              device: torch.device,
                              summary_writer,
                              learning_rate: float = 1e-1,
                              weight_decay: float = 1e-7,
                              momentum: float = 0.9,
                              n_epochs=16) -> 'Parallel_NN_Decoder':

    '''
    optimizer = torch_optim.SGD(parallel_decoder.parameters(),
                                momentum=momentum,
                                lr=learning_rate,
                                weight_decay=weight_decay)
                                '''
    optimizer = torch_optim.Adam(parallel_decoder.parameters(),
                                lr=1e-4,
                                weight_decay=weight_decay)
    n_steps_per_epoch = len(hpf_dataloader)
    for ep in range(n_epochs):

        for it, (images_np, spikes_np) in enumerate(hpf_dataloader):

            # shape (batch, height, width)
            hpf_torch= torch.tensor(images_np, dtype=torch.float32, device=device)
            batch, height, width = hpf_torch.shape

            # shape (batch, n_cells, n_timebins)
            spikes_torch = torch.tensor(spikes_np, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            output_flat = parallel_decoder(spikes_torch).reshape(batch, height, width)
            loss = loss_callable(output_flat, hpf_torch)

            loss.backward()
            optimizer.step()

            # log stuff out to Tensorboard
            # loss is updated every step
            summary_writer.add_scalar('training loss', loss.item(), ep * n_steps_per_epoch + it)

            if it % 16 == 0:
                ex_fig = plot_examples(images_np, output_flat.detach().cpu().numpy())
                summary_writer.add_figure('training example images',
                                          ex_fig,
                                          global_step=ep*n_steps_per_epoch + it)
            del hpf_torch, spikes_torch, output_flat, loss

        test_loss = eval_test_loss_decoder(parallel_decoder,
                                           test_hpf_dataloader,
                                           loss_callable,
                                           device)

        # log stuff out to Tensorboard
        # loss is updated every step
        summary_writer.add_scalar('test loss ', test_loss, (ep + 1) * n_steps_per_epoch)

    return parallel_decoder


class Parallel_NN_Decoder(nn.Module):
    '''
    Massively parallel implementation of NN_Decoder by Eric Wu (wueric)
        using grouped 1D convolutions and fancy
        Pytorch indexing operations

    In our implementation, we only use known real cells
        so the terms "unit" and "cell" can be used
        interchangeably
    '''

    def __init__(self,
                 pix_cell_sel: np.ndarray,
                 cell_unit_count: int,
                 t_dim: int,
                 k_dim: int,
                 h_dim: int,
                 p_dim: int,
                 f_dim: int):
        '''

        :param pix_cell_sel: shape (p_dim, k_dim)
        :param cell_unit_count:
        :param t_dim: int, number of time bins
        :param k_dim: int, number of cells to select for each pixel
        :param h_dim: int, width of hidden layer
        :param p_dim: int, number of pixels
        :param f_dim: int, number of features per cell
        '''

        super().__init__()

        self.cell_unit_count = cell_unit_count
        self.t_dim = t_dim
        self.h_dim = h_dim
        self.p_dim = p_dim
        self.k_dim = k_dim
        self.f_dim = f_dim

        if pix_cell_sel.shape != (self.p_dim, self.k_dim):
            raise ValueError(f"pix_cell_sel must have shape {(self.p_dim, self.k_dim)}")

        # shape (p_dim, k_dim)
        self.register_buffer('pix_cell_sel', torch.tensor(pix_cell_sel, dtype=torch.long))

        # self.cell_unit_count parallel Linear layers,
        # with self.t_dim inputs, and self.f_dim outputs
        self.featurize = nn.Conv1d(self.cell_unit_count,
                                   self.f_dim * self.cell_unit_count,
                                   kernel_size=self.t_dim,
                                   groups=self.cell_unit_count,
                                   stride=1,
                                   padding=0,
                                   bias=True)

        # self.p_dim parallel Linear layers
        # with self.k_dim * self.f_dim inputs, and self.h_dim outputs
        self.hidden1 = nn.Conv1d(self.p_dim,
                                 self.h_dim * self.p_dim,
                                 kernel_size=self.k_dim * self.f_dim,
                                 groups=self.p_dim,
                                 stride=1,
                                 padding=0,
                                 bias=True)

        self.nl = nn.PReLU()

        # self.p_dim parallel Linear layers
        # with self.h_dim inputs, and 1 output
        self.output_layer = nn.Conv1d(self.p_dim,
                                      self.p_dim,
                                      kernel_size=self.h_dim,
                                      stride=1,
                                      padding=0,
                                      groups=self.p_dim,
                                      bias=True)

    def forward(self, time_binned_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param time_binned_spikes: shape (batch, n_cells, n_timebins)
            aka (batch, cell_unit_count, t_dim)
        :return:
        '''

        batch, n_cells, n_timebins = time_binned_spikes.shape
        if n_cells != self.cell_unit_count:
            raise ValueError(
                f'time_binned_spikes wrong number of cells, had {n_cells}, expected {self.cell_unit_count}')
        if n_timebins != self.t_dim:
            raise ValueError(f'time_binned_spikes wrong number of time bins, had {n_timebins}, expected {self.t_dim}')

        # shape (batch, n_cells, n_timebins)

        # shape (batch, n_cells, n_timebins) -> (batch, self.cell_unit_count * self.f_dim, 1)
        # -> (batch, self.cell_unit_count, self.f_dim)
        time_featurized_outputs_unshape = self.featurize(time_binned_spikes)
        #print('time_featurized_outputs_unshape', time_featurized_outputs_unshape.shape)
        time_featurized_outputs = time_featurized_outputs_unshape.reshape(batch, self.cell_unit_count, self.f_dim)
        #print('time_featurized_outputs', time_featurized_outputs.shape)

        # the input to the next layer, self.hidden1, must have shape (self.k_dim * self.f_dim)
        # in the conv1d "time-bin" dimension
        # the overall input should have shape (batch, self.p_dim, self.k_dim * self.f_dim)
        # We get this by collecting all of the features of the cell and concatenating them with gather

        # now we have to select across the cells, which is
        # (self.p_dim, self.k_dim) -> (batch, self.p_dim, self.k_dim, self.f_dim)
        selection_indices_repeated = self.pix_cell_sel[None, :, :, None].expand(batch, -1, -1, self.f_dim)

        # shape (batch, self.p_dim, self.cell_unit_count, self.f_dim)
        time_featurized_outputs_exp = time_featurized_outputs[:, None, :, :].expand(-1, self.p_dim, -1, -1)
        #print('selection_indices_repeated', selection_indices_repeated.shape)
        #print('time_featurized_outputs_exp', time_featurized_outputs_exp.shape)

        # -> (batch, self.p_dim, self.k_dim, self.f_dim)
        selected_cell_features = torch.gather(time_featurized_outputs_exp, 2, selection_indices_repeated)

        # -> (batch, self.p_dim, self.k_dim * self.f_dim)
        selected_cell_features_flat = selected_cell_features.reshape(batch, self.p_dim, -1)

        # -> (batch, self.p_dim * self.h_dim, 1) -> (batch, self.p_dim, self.h_dim)
        hidden1_applied = self.nl(self.hidden1(selected_cell_features_flat)).reshape(batch, self.p_dim, self.h_dim)

        # shape (batch, self.p_dim, 1) -> (batch, self.p_dim)
        output_layer_applied = self.output_layer(hidden1_applied).squeeze(2)

        return output_layer_applied


class NN_Decoder(nn.Module):
    def __init__(self, unit_no, t_dim, k_dim, h_dim, p_dim, f_dim):
        super().__init__()
        self.unit_no = unit_no
        self.t_dim = t_dim
        self.h_dim = h_dim
        self.p_dim = p_dim
        self.k_dim = k_dim
        self.f_dim = f_dim

        self.featurize = nn.ModuleList([nn.Linear(self.t_dim,
                                                  self.f_dim) for i in range(self.unit_no)]).cuda()

        self.hidden1 = nn.ModuleList([nn.Linear(self.k_dim * self.f_dim,
                                                self.h_dim) for i in range(self.p_dim)]).cuda()
        self.hidden1_act = nn.ModuleList([nn.PReLU() for i in range(self.p_dim)]).cuda()

        self.output_layer = nn.ModuleList([nn.Linear(self.h_dim,
                                                     1) for i in range(self.p_dim)]).cuda()

    def forward(self, S, pix_units):

        F = torch.empty(S.shape[0], self.unit_no * self.f_dim).cuda()
        for n in range(self.unit_no):
            feat_n = self.featurize[n](S[:, n * self.t_dim: (n + 1) * self.t_dim])
            F[:, n * self.f_dim: (n + 1) * self.f_dim] = feat_n

        I = torch.empty(S.shape[0], self.p_dim).cuda()

        for x in range(self.p_dim):
            unit_ids = pix_units[x]
            feat_ids = torch.empty((self.k_dim * self.f_dim))
            for i in range(self.k_dim):
                feat_ids[i * self.f_dim: (i + 1) * self.f_dim] = torch.arange(self.f_dim) + unit_ids[i] * self.f_dim

            pix_feat = self.hidden1[x](F[:, feat_ids.long()])
            pix_feat = self.hidden1_act[x](pix_feat)

            out = self.output_layer[x](pix_feat)

            I[:, x] = out.reshape(-1)

        return I
