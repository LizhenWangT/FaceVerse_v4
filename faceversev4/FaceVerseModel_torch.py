import numpy as np
import os
import torch


class FaceVerseModel_torch:
    def __init__(self, device, facevrsepath, camera_distance, focal, center):
        self.faceversepath = facevrsepath
        print("Initialize FaceVerse_v4", self.faceversepath)
        self.device = device

        self.cameraIn = [0, 1, 2, 3]
        self.cameraIn[0] = focal
        self.cameraIn[1] = focal
        self.cameraIn[2] = center
        self.cameraIn[3] = center
        self.p_mat = torch.from_numpy(self._get_p_mat()).to(self.device).unsqueeze(0)

        self.camera_distance = camera_distance
        self.cameraT = torch.from_numpy(np.array([0, 0, camera_distance])).float().to(self.device).unsqueeze(0)

        self.init_faceverse()

    def init_faceverse(self):
        self.fvd = np.load(self.faceversepath, allow_pickle=True).item()

        self.kpts_for_train = np.concatenate([self.fvd['keypoints_mediapipe'].flatten(), self.fvd['keypoints'].flatten(), 
                                    self.fvd['keypoints_68'].flatten(), self.fvd['keypoints_mediapipe'].flatten()[[468, 473]]], axis=0)
        self.kp_inds = torch.from_numpy(self.kpts_for_train).type(torch.int64).to(self.device)
        self.kp_name = self.fvd['keypoints_name_list']
        self.ver_inds = self.fvd['ver_inds']
        self.tri_inds = self.fvd['tri_inds']

        self.id_dims = self.fvd['idBase'].shape[1]
        self.tex_dims = self.fvd['texBase'].shape[1]
        self.exp_dims = self.fvd['exBase'].shape[1]
        self.all_dims = self.id_dims + self.tex_dims + self.exp_dims

        # self.fvd['meanshape'][:, [0, 1]] *= -1
        # self.fvd['idBase'].reshape(-1, 3, self.id_dims)[:, [0, 1]] *= -1
        # self.fvd['exBase'].reshape(-1, 3, self.exp_dims)[:, [0, 1]] *= -1
        
        # for tracking by landmarks
        kp_inds_lms = torch.cat([self.kp_inds[:, None] * 3, self.kp_inds[:, None] * 3 + 1, self.kp_inds[:, None] * 3 + 2], dim=1).flatten().to(self.device)
        self.idBase_lms = torch.from_numpy(self.fvd['idBase']).to(self.device)[kp_inds_lms, :].unsqueeze(0)
        self.exBase_lms = torch.from_numpy(self.fvd['exBase']).to(self.device)[kp_inds_lms, :].unsqueeze(0)
        self.meanshape_lms = torch.from_numpy(self.fvd['meanshape']).to(self.device)[self.kp_inds].unsqueeze(0)

        self.face_mask = torch.from_numpy(self.fvd['face_mask']).to(self.device).reshape(-1, self.ver_inds[-1], 1)
        self.skin_mask = torch.from_numpy(self.fvd['parsing']['skin']).to(self.device).reshape(-1, self.ver_inds[-1], 1)
        front_ver = np.arange(self.ver_inds[-1])[self.fvd['face_mask'] > 0]
        self.front_face_buf = []
        for face in self.fvd['tri']:
            if face[0] not in front_ver or face[1] not in front_ver or face[2] not in front_ver:
                continue
            else:
                self.front_face_buf.append(face)
        self.front_face_buf = torch.from_numpy(np.stack(self.front_face_buf)).to(self.device).type(torch.int64)

        self.idBase = torch.from_numpy(self.fvd['idBase']).to(self.device).unsqueeze(0) / 100
        self.exBase = torch.from_numpy(self.fvd['exBase']).to(self.device).unsqueeze(0) / 100
        self.texBase = torch.from_numpy(self.fvd['texBase']).to(self.device).unsqueeze(0)
        self.meanshape = torch.from_numpy(self.fvd['meanshape']).to(self.device).unsqueeze(0) / 100
        self.meantex = torch.from_numpy(self.fvd['meantex']).to(self.device).unsqueeze(0).type(torch.float32)
        self.tri = torch.from_numpy(self.fvd['tri']).to(self.device).type(torch.int64)
        self.point_buf = torch.from_numpy(self.fvd['point_buf']).to(self.device).type(torch.int64)
        
        self.rotXYZ = torch.eye(3).view(1, 3, 3).repeat(3, 1, 1).view(3, 1, 3, 3).to(self.device)
        self.gamma0 = torch.zeros(1, 9, 3).to(self.device)
        self.gamma0[:, 0, :] = self.gamma0[:, 0, :] + 0.8

        #self.init_coeff_tensors()

    def set_camera_pers(self, cameraIn, cameraR, cameraT):
        # cameraIn fx, fy, cx, cy
        assert cameraIn.shape == (4,), 'shape of cameraIn should be (4,), get {}'.format(cameraIn.shape)
        assert cameraR.shape == (3, 3), 'shape of cameraR should be (3, 3), get {}'.format(cameraR.shape)
        assert cameraT.shape == (3,), 'shape of cameraT should be (3,), get {}'.format(cameraT.shape)

        self.cameraIn = cameraIn
        self.cameraR = torch.from_numpy(cameraR).to(self.device).unsqueeze(0)
        self.cameraT = torch.from_numpy(cameraT).to(self.device).unsqueeze(0)
        self.p_mat = torch.from_numpy(self._get_p_mat()).to(self.device).unsqueeze(0)

    def init_coeff_tensors(self):
        self.id_tensor = torch.zeros((1, self.id_dims), dtype=torch.float32).to(self.device)
        self.exp_tensor = torch.zeros((1, self.exp_dims), dtype=torch.float32).to(self.device)
        self.tex_tensor = torch.zeros((1, self.tex_dims), dtype=torch.float32).to(self.device)
        self.lighting_tensor = torch.zeros((1, 27), dtype=torch.float32).to(self.device)
        self.rot_tensor = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        self.trans_tensor = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        self.eye_tensor = torch.zeros((1, 4), dtype=torch.float32).to(self.device)
        self.id_tensor.requires_grad = True
        self.exp_tensor.requires_grad = True
        self.tex_tensor.requires_grad = True
        self.lighting_tensor.requires_grad = True
        self.rot_tensor.requires_grad = True
        self.trans_tensor.requires_grad = True
        self.eye_tensor.requires_grad = True
    
    def init_coeff_tensor_RT(self):
        self.rot_tensor = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        self.trans_tensor = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        self.rot_tensor.requires_grad = True
        self.trans_tensor.requires_grad = True

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :self.id_dims]  # identity(shape) coeff 
        exp_coeff = coeffs[:, self.id_dims:self.id_dims+self.exp_dims]  # expression coeff 
        tex_coeff = coeffs[:, self.id_dims+self.exp_dims:self.all_dims]  # texture(albedo) coeff 
        lighting = coeffs[:, self.all_dims:self.all_dims+27] # lighting coeff for 3 channel SH function of dim 27
        angles = coeffs[:, self.all_dims+27:self.all_dims+30] # ruler angles(x,y,z) for rotation of dim 3
        translation = coeffs[:, self.all_dims+30:self.all_dims+33]  # translation coeff of dim 3
        eye_coeff = coeffs[:, self.all_dims+33:self.all_dims+37]  # eye coeff of dim 4
        return id_coeff, exp_coeff, tex_coeff, lighting, angles, translation, eye_coeff

    def split_coeffs_dict(self, coeffs):
        id_coeff = coeffs[:, :self.id_dims]  # identity(shape) coeff 
        exp_coeff = coeffs[:, self.id_dims:self.id_dims+self.exp_dims]  # expression coeff 
        tex_coeff = coeffs[:, self.id_dims+self.exp_dims:self.all_dims]  # texture(albedo) coeff 
        lighting = coeffs[:, self.all_dims:self.all_dims+27] # lighting coeff for 3 channel SH function of dim 27
        angles = coeffs[:, self.all_dims+27:self.all_dims+30] # ruler angles(x,y,z) for rotation of dim 3
        translation = coeffs[:, self.all_dims+30:self.all_dims+33]  # translation coeff of dim 3
        eye_coeff = coeffs[:, self.all_dims+33:self.all_dims+37]  # eye coeff of dim 4
        return {
            "id": id_coeff, 
            "exp": exp_coeff, 
            "tex": tex_coeff, 
            "gamma": lighting, 
            "angle": angles, 
            "trans": translation, 
            "eyes": eye_coeff
        }
    
    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 478 + 370 + 70, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 621)
        """
        id_coeff, exp_coeff, tex_coeff, lighting, angles, translation, eye_coeff = self.split_coeffs(coeffs)
        exp_coeff[:, 171:] = exp_coeff[:, 171:] * 0.0 # tongue coeff will not be computed
        rotation = self.compute_rotation_matrix(angles)

        vs = self.get_vs(id_coeff, exp_coeff, eye_coeff)
        vs_t = self.rigid_transform(vs, rotation, translation)
        vs_proj = self.project_vs(vs_t)

        norm = self.compute_norm(vs, self.tri, self.point_buf)
        norm_t = torch.matmul(norm, rotation)

        colors = self.get_color(tex_coeff)
        colors_illumin = self.add_illumination(colors, norm_t, lighting)

        return vs_t + self.cameraT.view(-1, 1, 3), colors, colors_illumin, self.get_lms(vs_proj)[:, :, :2]
    
    def compute_for_final(self, coeffs, compute_color=True):
        id_coeff, exp_coeff, tex_coeff, lighting, angles, translation, eye_coeff = self.split_coeffs(coeffs)
        rotation = self.compute_rotation_matrix(angles)
        exp_coeff_ = exp_coeff.clone()
        # for better eye closing
        target_part = exp_coeff_[:, 14:16]
        target_part[target_part > 0.0] *= 2
        target_part[target_part > 1] = 1
        exp_coeff_[:, 14:16] = target_part

        vs = self.get_vs(id_coeff, exp_coeff_, eye_coeff)
        vs_t = self.rigid_transform(vs, rotation, translation)
        vs_proj = self.project_vs(vs_t)

        norm = self.compute_norm(vs, self.tri, self.point_buf)
        norm_t = torch.matmul(norm, rotation)

        if compute_color:
            colors = self.get_color(tex_coeff)
            colors_illumin = self.add_illumination(colors, norm_t, lighting)
        else:
            colors_illumin = None
        
        return vs, vs_proj, norm_t, colors_illumin

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, lighting, angles, translation, eye_coeff):
        coeffs = torch.cat([id_coeff, exp_coeff, tex_coeff, lighting, angles, translation, eye_coeff], dim=1)
        return coeffs

    def get_packed_tensors(self):
        return self.merge_coeffs(self.id_tensor,
                                 self.exp_tensor,
                                 self.tex_tensor,
                                 self.lighting_tensor, self.rot_tensor, 
                                 self.trans_tensor, self.eye_tensor)

    def run(self, coeffs, only_lms=False, use_color=False, use_lighting=False):
        id_coeff, exp_coeff, tex_coeff, lighting, angles, translation, eye_coeff = self.split_coeffs(coeffs)
        rotation = self.compute_rotation_matrix(angles)

        if only_lms:
            lms = self.get_vs_lms(id_coeff, exp_coeff, eye_coeff)
            lms_t = self.rigid_transform(lms, rotation, translation)
            lms_proj = self.project_vs(lms_t)
            return {'lms_proj': lms_proj, 'lms':lms_t}
        else:
            vs = self.get_vs(id_coeff, exp_coeff, eye_coeff)
            vs_t = self.rigid_transform(vs, rotation, translation)
            vs_proj = self.project_vs(vs_t)

            lms_t = self.get_lms(vs_t)
            lms_proj = self.project_vs(lms_t)

            norm = self.compute_norm(vs, self.tri, self.point_buf)
            norm_t = torch.matmul(norm, rotation)

            if use_color:
                colors = self.get_color(tex_coeff)
                if use_lighting:
                    colors_illumin = self.add_illumination(colors, norm_t, lighting) * self.face_mask
                else:
                    colors_illumin = None
            else:
                colors = None
                colors_illumin = None
            
            return {'lms_proj': lms_proj, 'lms':lms_t, 'vertices': vs_t, 'vertices_proj': vs_proj, 'norms': norm_t, 'colors': colors, 'colors_illumin': colors_illumin}
    
    def get_vs(self, id_coeff, exp_coeff, eye_coeff):
        face_shape = torch.matmul(self.idBase, id_coeff.unsqueeze(2)) + torch.matmul(self.exBase, exp_coeff.unsqueeze(2))
        face_shape = face_shape.reshape(id_coeff.shape[0], -1, 3) + self.meanshape
        l_eye_c, r_eye_c = self.get_eye_centers(id_coeff)
        l_eye_mat = self.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_mat = self.compute_eye_rotation_matrix(eye_coeff[:, 2:])
        face_shape[:, self.ver_inds[0]:self.ver_inds[1]] = torch.matmul(face_shape[:, self.ver_inds[0]:self.ver_inds[1]] - l_eye_c, l_eye_mat) + l_eye_c
        face_shape[:, self.ver_inds[1]:self.ver_inds[2]] = torch.matmul(face_shape[:, self.ver_inds[1]:self.ver_inds[2]] - r_eye_c, r_eye_mat) + r_eye_c
        face_shape[:, self.ver_inds[2]:self.ver_inds[5], 2] = face_shape[:, self.ver_inds[2]:self.ver_inds[5], 2] - \
                (face_shape[:, 19496:19497, 2] - face_shape[:, 6550:6551, 2]) + (self.meanshape[:, 19496:19497, 2] - self.meanshape[:, 6550:6551, 2])
        return face_shape

    def get_vs_lms(self, id_coeff, exp_coeff, eye_coeff):
        face_shape = torch.matmul(self.idBase_lms, id_coeff.unsqueeze(2)) + torch.matmul(self.exBase_lms, exp_coeff.unsqueeze(2))
        face_shape = face_shape.reshape(id_coeff.shape[0], -1, 3) + self.meanshape_lms
        l_eye_c, r_eye_c = self.get_eye_centers(id_coeff)
        l_eye_mat = self.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_mat = self.compute_eye_rotation_matrix(eye_coeff[:, 2:])
        leye_list = list(range(473, 478)) + [eind + 478 for eind in self.kp_name['pupil_left_5']] + [478 + 370 + 69]
        reye_list = list(range(468, 473)) + [eind + 478 for eind in self.kp_name['pupil_right_5']] + [478 + 370 + 68]
        face_shape[:, leye_list] = torch.matmul(face_shape[:, leye_list] - l_eye_c, l_eye_mat) + l_eye_c
        face_shape[:, reye_list] = torch.matmul(face_shape[:, reye_list] - r_eye_c, r_eye_mat) + r_eye_c
        return face_shape
    
    def get_eye_centers(self, id_coeff):
        eye_shape = torch.matmul(self.idBase.reshape(-1, 3, self.id_dims)[self.ver_inds[0]:self.ver_inds[2]].reshape(1, -1, self.id_dims), 
                    id_coeff.unsqueeze(2)).reshape(id_coeff.shape[0], -1, 3) + self.meanshape[:, self.ver_inds[0]:self.ver_inds[2]]
        l_eye_c = torch.mean(eye_shape[:, :self.ver_inds[1] - self.ver_inds[0]], dim=1, keepdims=True)
        r_eye_c = torch.mean(eye_shape[:, self.ver_inds[1] - self.ver_inds[0]:], dim=1, keepdims=True)
        return l_eye_c, r_eye_c

    def get_color(self, tex_coeff):
        face_texture = torch.matmul(self.texBase, tex_coeff.unsqueeze(2)).reshape(tex_coeff.shape[0], -1, 3) + self.meantex
        return face_texture / 255

    def _get_p_mat(self):
        p_matrix = np.array([self.cameraIn[0], 0.0, self.cameraIn[2] - 0.5, # -0.5 for OpenGL rendering
                             0.0, self.cameraIn[1], self.cameraIn[3] - 0.5,
                             0.0, 0.0, 1.0], dtype=np.float32).reshape(3, 3)
        return np.array(p_matrix, dtype=np.float32)

    def compute_norm(self, vs, tri, point_buf):
        v1 = vs[:, tri[:, 0], :]
        v2 = vs[:, tri[:, 1], :]
        v3 = vs[:, tri[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2)

        v_norm = face_norm[:, point_buf, :].sum(2)
        v_norm = v_norm / (torch.norm(v_norm, p=2, dim=2, keepdim=True) + 1e-9)

        return v_norm

    def project_vs(self, vs):
        # change coordinates
        vs_tmp = vs + self.cameraT.view(-1, 1, 3)#torch.matmul(vs, self.cameraR.permute(0, 2, 1)) + self.cameraT.view(-1, 1, 3)
        aug_projection = torch.matmul(vs_tmp, self.p_mat.permute(0, 2, 1))
        face_projection = aug_projection[:, :, :2] / aug_projection[:, :, 2:3]
        return torch.stack([face_projection[:, :, 0], face_projection[:, :, 1], aug_projection[:, :, 2]], dim=2)

    def compute_eye_rotation_matrix(self, eye):
        # 0 left_eye + down to up
        # 1 left_eye + right to left
        # 2 right_eye + down to up
        # 3 right_eye + right to left
        sinx = torch.sin(eye[:, 0])
        siny = torch.sin(eye[:, 1])
        cosx = torch.cos(eye[:, 0])
        cosy = torch.cos(eye[:, 1])

        if eye.shape[0] != 1:
            rotXYZ = self.rotXYZ.repeat(1, eye.shape[0], 1, 1)
        else:
            rotXYZ = self.rotXYZ.detach().clone()
        
        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy

        rotation = rotXYZ[1].bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    def compute_rotation_matrix(self, angles):
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        if angles.shape[0] != 1:
            rotXYZ = self.rotXYZ.repeat(1, angles.shape[0], 1, 1)
        else:
            rotXYZ = self.rotXYZ.detach().clone()
        
        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    def add_illumination(self, face_texture, norm, gamma):
        gamma = gamma.view(-1, 9, 3) + self.gamma0#.clone()
        #gamma[:, :, 0] += 0.8
        #gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(a0 * c0 * (nx * 0 + 1))
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(gamma.shape[0], face_texture.shape[1], 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color

    def rigid_transform(self, vs, rot, trans):
        vs_r = torch.matmul(vs, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)
        return vs_t
    
    def init_mediapipe_weights(self):
        lips = [61,146,91,181,84,17,314,405,321,375,61,185,40,39,37,0,267,269,270,409,
                78,95,88,178,87,14,317,402,318,324,78,191,80,81,82,13,312,311,310,415]
        leye = [263,249,390,373,374,380,381,382,263,466,388,387,386,385,384,398]
        reye = [33,7,163,144,145,154,154,155,33,246,161,160,159,158,157,173]
        lbrow = [276,283,282,295,300,293,334,296]
        rbrow = [46,53,52,65,70,63,105,66]
        oval = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
        w = torch.ones(478)
        w[lips] = 5
        w[leye] = 5
        w[reye] = 5
        w[lbrow] = 5
        w[rbrow] = 5
        w[oval] = 2
        w[468:] = 5
        norm_w = w / w.sum()
        return norm_w
    
    def init_weights(self):
        w = torch.ones(370)
        w[self.kp_name['out_17']] = 2
        w[self.kp_name['left_brow_5']] = 5
        w[self.kp_name['right_brow_5']] = 5
        w[self.kp_name['left_eye_6']] = 5
        w[self.kp_name['right_eye_6']] = 5
        w[48:78] = 5
        norm_w = w / w.sum()
        return norm_w
    
    def init_70_weights(self):
        w = torch.ones(70)
        w[self.kp_name['out_17']] = 2
        w[self.kp_name['left_brow_5']] = 5
        w[self.kp_name['right_brow_5']] = 5
        w[self.kp_name['left_eye_6']] = 5
        w[self.kp_name['right_eye_6']] = 5
        w[48:70] = 5
        norm_w = w / w.sum()
        return norm_w

