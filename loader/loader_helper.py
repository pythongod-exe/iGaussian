import numpy as np

def quaternion_to_rotation_matrix(qvec): # wxyz
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotation_matrix_to_quaternion(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def combine_3dgs_rotation_translation(R_c2w, T_w2c):
    RT_w2c = np.eye(4)
    RT_w2c[:3, :3] = R_c2w.T
    RT_w2c[:3, 3] = T_w2c
    RT_c2w=np.linalg.inv(RT_w2c)
    return RT_c2w

def normalize_quaternion(q):
    """
    对四元数进行归一化
    参数:
    q -- 长度为4的四元数 [w, x, y, z]

    返回:
    归一化后的四元数
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("四元数的模长为0，无法进行归一化")
    return q / norm



