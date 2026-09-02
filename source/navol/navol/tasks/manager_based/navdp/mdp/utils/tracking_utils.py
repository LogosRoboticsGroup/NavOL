import casadi as ca
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scipy.interpolate import interp1d
from typing import Optional, List, Tuple
from dataclasses import dataclass
import torchinterp1d
import cvxpy as cp

from abc import ABC, abstractmethod

from isaaclab.utils.types import ArticulationActions

class PlanningInput:
    current_goal: Optional[np.ndarray] = None
    current_image: Optional[np.ndarray] = None
    current_depth: Optional[np.ndarray] = None
    camera_pos: Optional[np.ndarray] = None
    camera_rot: Optional[np.ndarray] = None

@dataclass
class PlanningOutput:
    trajectory_points_world: Optional[np.ndarray] = None
    all_trajectories_world: Optional[List[np.ndarray]] = None
    all_values_camera: Optional[np.ndarray] = None
    is_planning: bool = False
    planning_error: Optional[str] = None

class Timing:
    """
    From https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/svox2/utils.py#L611
    
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")
        

from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor

class LinearMPC_Controller_Batch:
    def __init__(self, global_planed_trajs, N=5, desired_v=0.5, 
                 v_max=0.5, w_max=0.5, ref_gap=3, batch_size=1):
        """
        批量线性MPC控制器
        Args:
            global_planed_trajs: [B, T, 2] - 批量规划轨迹
            batch_size: 并行处理的批次大小
        """
        self.N = N
        self.desired_v = desired_v
        self.ref_gap = ref_gap
        self.T = 0.1
        self.v_max = v_max
        self.w_max = w_max
        self.batch_size = batch_size
        
        # 密集化参考轨迹
        self.ref_trajs = self.make_ref_denser(global_planed_trajs)
        self.ref_traj_len = N // ref_gap + 1
        
        # 权重矩阵
        self.Q = np.diag([10.0, 10.0, 1.0])
        self.R = np.diag([0.02, 0.15])
        
        # 预计算优化问题结构
        self.optimization_structures = []
        self.executor = ThreadPoolExecutor(max_workers=batch_size)
        
        # 为每个轨迹创建优化结构
        for _ in range(batch_size):
            opt_struct = {
                'bounds': [(0, v_max) if j == 0 else (-w_max, w_max) 
                          for _ in range(N) for j in range(2)],
                'args': (None, None, None)  # 将在solve时填充
            }
            self.optimization_structures.append(opt_struct)
    
    def make_ref_denser(self, ref_trajs, ratio=50):
        """向量化轨迹密集化"""
        B, L = ref_trajs.shape[:2]
        x_orig = torch.arange(L, device=ref_trajs.device)[None].repeat(B, 1)
        xnew = torch.linspace(0, L - 1, L * ratio, device=ref_trajs.device)[None].repeat(B, 1)
        
        for b in range(B):
            uniform_x = torchinterp1d.interp1d(x_orig, ref_trajs[b, :, 0], xnew, None)
            uniform_y = torchinterp1d.interp1d(x_orig, ref_trajs[b, :, 1], xnew, None)
            dense_traj = torch.stack((uniform_x, uniform_y), axis=-1)
        
        return dense_traj
    
    
    def linearize_model(self, theta):
        if isinstance(theta, np.ndarray) and theta.ndim > 0:
            # 批量处理
            A = np.zeros((len(theta), 3, 3))
            B = np.zeros((len(theta), 3, 2))

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # 连续时间系统矩阵
            A[:, 0, 2] = -self.desired_v * sin_theta
            A[:, 1, 2] = self.desired_v * cos_theta
            
            B[:, 0, 0] = cos_theta
            B[:, 1, 0] = sin_theta
            B[:, 2, 1] = 1
            
            # 离散化（前向欧拉）
            A = np.eye(3, device=theta.device) + A * self.T
            B = B * self.T
            
            return A, B
        else:
            # 标量处理
            A_cont = np.array([
                [0, 0, -self.desired_v * np.sin(theta)],
                [0, 0,  self.desired_v * np.cos(theta)],
                [0, 0, 0]
            ])

            B_cont = np.array([
                [np.cos(theta), 0],
                [np.sin(theta), 0],
                [0, 1]
            ])

            # 离散化
            A = np.eye(3) + A_cont * self.T
            B = B_cont * self.T
            
            return A, B
    
    def compute_reference_points(self, x0, global_traj):
        """找到局部参考轨迹（批量处理）"""
        # 计算到所有点的距离
        distances = torch.norm(global_traj - x0[:2], dim=1)
        nearest_idx = torch.argmin(distances)
        
        # 选择参考点
        ref_points = []
        for i in range(self.ref_traj_len):
            target_idx = min(nearest_idx + i * self.ref_gap, len(global_traj) - 1)
            # 添加参考点（x, y, 计算朝向）
            px, py = global_traj[target_idx]
            
            # 计算朝向（下一个点的方向）
            next_idx = min(target_idx + 1, len(global_traj) - 1)
            next_point = global_traj[next_idx]
            theta_ref = torch.arctan2(next_point[1] - py, next_point[0] - px)

            ref_points.append(torch.tensor([px, py, theta_ref], device=global_traj.device))

        return torch.stack(ref_points)

    def objective_function(self, u_flat, x0, A, B, ref_points):
        """优化目标函数（支持批量处理）"""
        U = u_flat.reshape((self.N, 2))
        J = 0
        x = x0.copy()
        
        for i in range(self.N):
            # 状态更新
            if A.ndim == 3:  # 批量处理
                x = np.einsum('ijk,ik->ij', A, x) + np.einsum('ijk,ik->ij', B, U[i])
            else:  # 单样本处理
                x = A @ x + B @ U[i]
            
            # 计算代价（仅在某些点匹配参考轨迹）
            if i % self.ref_gap == 0:
                ref_idx = min(i // self.ref_gap, len(ref_points) - 1)
                state_error = x - ref_points[ref_idx]
                
                if state_error.ndim == 1:  # 单样本
                    state_cost = state_error.T @ self.Q @ state_error
                else:  # 批量
                    state_cost = np.einsum('bi,ij,bj->b', state_error, self.Q, state_error).sum()
                
                control_cost = U[i] @ self.R @ U[i]
                J += state_cost + control_cost
        
        return J
    
    def solve_single(self, idx, x00_batch, ref_trajs):
        """求解单个轨迹的MPC问题"""
        x0 = x00_batch[idx]
        global_traj = ref_trajs[idx]
        
        # 找到局部参考轨迹
        ref_points = self.compute_reference_points(x0, global_traj).cpu().numpy()
        
        # 在当前状态下线性化模型
        x0 = x0.cpu().numpy()
        A, B = self.linearize_model(x0[2])
        
        # 设置优化参数
        self.optimization_structures[idx]['args'] = (x0, A, B, ref_points)
        
        # 初始控制序列
        u0 = np.zeros((self.N, 2)).flatten()
        
        # 求解优化问题
        res = minimize(self.objective_function, u0,
                       args=self.optimization_structures[idx]['args'],
                       bounds=self.optimization_structures[idx]['bounds'],
                       method='SLSQP', options={'maxiter': 30, 'ftol': 1e-4})
        
        return res.x.reshape((self.N, 2))[1]  # 只返回第一步控制
    
    def solve(self, x00_batch):
        """批量求解函数"""
        # 并行处理
        futures = []
        for b in range(self.batch_size):
            futures.append(self.executor.submit(self.solve_single, b, x00_batch, self.ref_trajs))
        
        # 收集结果
        control_outputs = [future.result() for future in futures]
        
        return torch.tensor(control_outputs, dtype=torch.float32)

class TorchPIDController_Batch:
    def __init__(self, global_planed_trajs, desired_v=0.5, 
                 v_max=0.5, w_max=0.5, lookahead=0.8, device='cuda'):
        self.desired_v = desired_v
        self.v_max = v_max
        self.w_max = w_max
        self.lookahead = lookahead
        self.device = device
        
        # PID参数
        self.kp_theta = 1.5
        self.ki_theta = 0.01
        self.kd_theta = 0.1
        
        # 状态变量
        self.prev_errors = None
        self.integrals = None
        
        # 密集化参考轨迹
        self.ref_trajs = self.make_ref_denser(global_planed_trajs)  # [B, L, 2]
        
    def make_ref_denser(self, ref_trajs, ratio=50):
        
        B, L = ref_trajs.shape[:2]
        x_orig = torch.arange(L, device=ref_trajs.device)[None].repeat(B, 1)
        xnew = torch.linspace(0, L - 1, L * ratio, device=ref_trajs.device)[None].repeat(B, 1)
        
        for b in range(B):
            uniform_x = torchinterp1d.interp1d(x_orig, ref_trajs[b, :, 0], xnew, None)
            uniform_y = torchinterp1d.interp1d(x_orig, ref_trajs[b, :, 1], xnew, None)
            dense_traj = torch.stack((uniform_x, uniform_y), axis=-1)
        
        return dense_traj
    


    
    def solve(self, x00_batch):
        """
        求解控制命令
        Args:
            x00_batch: [B, 3] - 初始状态
        Returns:
            u: [B, 2] - 控制命令 [v, w]
        """
        with Timing("PID_solve"):
            B = x00_batch.shape[0]
            x00_batch = x00_batch.to(self.device)
            
            # 初始化状态变量
            if self.prev_errors is None:
                self.prev_errors = torch.zeros(B, device=self.device)
                self.integrals = torch.zeros(B, device=self.device)
            
            # 存储控制输出
            controls = torch.zeros(B, 2, device=self.device)

            with Timing("****PID_solve_batch"):
                for b in range(B):
                    x0 = x00_batch[b]
                    global_traj = self.ref_trajs[b]

                    # 找到预瞄点
                    with Timing("find"):
                        target_pt = self.find_target_point(x0, global_traj)
                    
                    # 计算角度误差
                    with Timing("pid"):
                        dx = target_pt[0] - x0[0]
                        dy = target_pt[1] - x0[1]
                        target_angle = torch.atan2(dy, dx)
                        angle_error = target_angle - x0[2]
                        
                        # 归一化角度误差(-pi到pi)
                        angle_error = (angle_error + torch.pi) % (2 * torch.pi) - torch.pi
                        
                        # PID计算
                        self.integrals[b] += angle_error
                        derivative = angle_error - self.prev_errors[b]
                        self.prev_errors[b] = angle_error
                        
                        w = (self.kp_theta * angle_error + 
                            self.ki_theta * self.integrals[b] + 
                            self.kd_theta * derivative)
                        
                        # 限幅处理
                        w = torch.clamp(w, -self.w_max, self.w_max)
                        v = self.desired_v
                        
                        # 接近终点时减速
                        dist_to_end = torch.norm(global_traj[-1] - x0[:2])
                        if dist_to_end < 0.5:
                            v = v * dist_to_end * 2

                        controls[b, 0] = v
                        controls[b, 1] = w

        return controls


class MPC_Controller_simple:
    def __init__(self, global_planed_traj, N = 15, desired_v = 0.5, v_max = 0.5, w_max = 0.5, ref_gap = 3):
        self.N, self.desired_v, self.ref_gap, self.T = N, desired_v, ref_gap, 0.1
        
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.lookahead_s = 0.5

        # setup mpc problem
        opti = ca.Opti()
        opt_controls = opti.variable(2)
        v, w = opt_controls[0], opt_controls[1]

        opt_states = opti.variable(3)
        x, y, theta = opt_states[0], opt_states[1], opt_states[2]

        # parameters 
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3) # the intermidia state may also be the parameter

        # system dynamics for mobile manipulator
        f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])

        # init_condition
        x_next = opt_x0 + f(opt_x0, opt_controls)*self.T
        opti.subject_to(opt_states==x_next)

        # define the cost function
        Q = np.diag([10.0,10.0,0.0])
        R = np.diag([0.02,0.15])
        
        err = opt_states - opt_xs
        obj = ca.mtimes([err.T, Q, err]) + ca.mtimes([opt_controls.T, R, opt_controls])
        
        # boundrary and control conditions
        opti.subject_to(opti.bounded(0.0, v, v_max))
        opti.subject_to(opti.bounded(-w_max, w, w_max))
        
        # opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':3, 'print_time':3, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        opts_setting = {
            'ipopt.max_iter': 20,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0, 'print_time': 0
        }
        opti.solver('ipopt', opts_setting)
        
        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_controls = opt_controls
        self.opt_states = opt_states
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        
    def make_ref_denser(self, ref_traj, ratio = 50):
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(0, len(ref_traj) - 1, num=len(ref_traj) * ratio)
        interp_func_x = interp1d(x_orig, ref_traj[:, 0], kind='linear')
        interp_func_y = interp1d(x_orig, ref_traj[:, 1], kind='linear')
        uniform_x = interp_func_x(new_x)
        uniform_y = interp_func_y(new_x)
        ref_traj = np.stack((uniform_x, uniform_y), axis=1)
        return ref_traj
    
    def _find_lookahead_point(self, x0):
        """按弧长在轨迹上前视 desired_v * lookahead_s; 若不足则取终点。"""
        pts = self.ref_traj
        d = np.linalg.norm(pts - x0[:2].reshape(1, 2), axis=1)
        nearest = int(np.argmin(d))
        target_dist = max(1e-6, float(self.desired_v * self.lookahead_s))

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        # 从最近点开始累计弧长
        acc = 0.0
        for i in range(nearest, len(pts) - 1):
            acc += seg[i]
            if acc >= target_dist:
                return pts[i + 1]
        return pts[-1]
    
    def solve(self, x00):
        ref_xy = self._find_lookahead_point(x00)
        xs = np.array([ref_xy[0], ref_xy[1], 0.0])  # 不跟踪朝向
        self.opti.set_value(self.opt_xs, xs) 
        u0 = np.zeros((2, ))
        x0 = np.zeros((3, ))
        self.opti.set_value(self.opt_x0, x00)
        self.opti.set_initial(self.opt_controls, u0)
        self.opti.set_initial(self.opt_states, x0)
        sol = self.opti.solve()
        self.last_opt_u_controls = sol.value(self.opt_controls)
        self.last_opt_x_states = sol.value(self.opt_states)

        return sol.value(self.opt_controls), sol.value(self.opt_states)
        
    def find_reference_traj(self, x0, global_planed_traj):
        ref_traj_pts = []
        # find the nearest point in global_planed_traj
        nearest_idx = np.argmin(np.linalg.norm(global_planed_traj - x0[:2].reshape((1, 2)), axis=1))
        desire_arc_length = self.desired_v * self.ref_gap * self.T 
        cum_dist = np.cumsum(np.linalg.norm(np.diff(global_planed_traj, axis=0), axis=1))

        # select the reference points from the nearest point to the end of global_planed_traj
        for i in range(nearest_idx, len(global_planed_traj) - 1):
            if cum_dist[i] - cum_dist[nearest_idx] >= desire_arc_length * len(ref_traj_pts):
                ref_traj_pts.append(global_planed_traj[i, :])
                if len(ref_traj_pts) == self.ref_traj_len:
                    break
        # if the target is reached before the reference trajectory is complete, add the last point of global_planed_traj 
        while len(ref_traj_pts) < self.ref_traj_len:
            ref_traj_pts.append(global_planed_traj[-1, :])
        return np.array(ref_traj_pts)
    
class MPC_Controller:
    def __init__(self, N = 15, desired_v = 0.5, v_max = 0.5, w_max = 0.5, ref_gap = 3):
        self.N, self.desired_v, self.ref_gap, self.T = N, desired_v, ref_gap, 0.1
        
        self.ref_traj_len = N // ref_gap + 1

        # setup mpc problem
        opti = ca.Opti()
        opt_controls = opti.variable(N, 2)
        v, w = opt_controls[:, 0], opt_controls[:, 1]

        opt_states = opti.variable(N+1, 3)
        x, y, theta = opt_states[:, 0], opt_states[:, 1], opt_states[:, 2]

        # parameters 
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3 * self.ref_traj_len) # the intermidia state may also be the parameter

        # system dynamics for mobile manipulator
        f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])

        # init_condition
        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(N):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*self.T
            opti.subject_to(opt_states[i+1, :]==x_next)

        # define the cost function
        Q = np.diag([10.0,10.0,0.0])
        R = np.diag([0.02,0.15])
        obj = 0 
        for i in range(N):
            obj = obj +ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
            if i % ref_gap == 0:
                nn = i // ref_gap
                obj = obj + ca.mtimes([(opt_states[i, :]-opt_xs[nn*3:nn*3+3].T), Q, (opt_states[i, :]-opt_xs[nn*3:nn*3+3].T).T])
        opti.minimize(obj)

        # boundrary and control conditions
        opti.subject_to(opti.bounded(0.0, v, v_max))
        opti.subject_to(opti.bounded(-w_max, w, w_max))
        
        opts_setting = {
            'ipopt.max_iter':100, 
            'ipopt.print_level':0, 
            'print_time':0,
            'ipopt.acceptable_tol':1e-8, 
            'ipopt.acceptable_obj_change_tol':1e-6,
            # 'expand': True,                      # 先把图展开，评估更快
            # 'jit': True, 'compiler': 'shell',    # JIT 到本地代码
            # 'jit_options': {'flags': '-O3'},
            }
        opti.solver('ipopt', opts_setting)
            
        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_controls = opt_controls
        self.opt_states = opt_states
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        
    def make_ref_denser(self, ref_traj, ratio = 50):
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(0, len(ref_traj) - 1, num=len(ref_traj) * ratio)
        interp_func_x = interp1d(x_orig, ref_traj[:, 0], kind='linear')
        interp_func_y = interp1d(x_orig, ref_traj[:, 1], kind='linear')
        uniform_x = interp_func_x(new_x)
        uniform_y = interp_func_y(new_x)
        ref_traj = np.stack((uniform_x, uniform_y), axis=1)
        return ref_traj
    
    def solve(self, x00, global_planed_traj):
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        ref_traj = self.find_reference_traj(x00, self.ref_traj)
        # fake a yaw angle
        ref_traj = np.concatenate((ref_traj, np.zeros((ref_traj.shape[0], 1))), axis=1).reshape(-1, 1)
        self.opti.set_value(self.opt_xs, ref_traj.reshape(-1, 1)) 
        u0 = np.zeros((self.N, 2)) if self.last_opt_u_controls is None else self.last_opt_u_controls
        x0 = np.zeros((self.N+1, 3)) if self.last_opt_x_states is None else self.last_opt_x_states
        self.opti.set_value(self.opt_x0, x00)
        self.opti.set_initial(self.opt_controls, u0)
        self.opti.set_initial(self.opt_states, x0)
        sol = self.opti.solve()
        self.last_opt_u_controls = sol.value(self.opt_controls)
        self.last_opt_x_states = sol.value(self.opt_states)

        return self.last_opt_u_controls, self.last_opt_x_states
    
    def reset(self):
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        
    def find_reference_traj(self, x0, global_planed_traj):
        ref_traj_pts = []
        # find the nearest point in global_planed_traj
        nearest_idx = np.argmin(np.linalg.norm(global_planed_traj - x0[:2].reshape((1, 2)), axis=1))
        desire_arc_length = self.desired_v * self.ref_gap * self.T 
        cum_dist = np.cumsum(np.linalg.norm(np.diff(global_planed_traj, axis=0), axis=1))

        # select the reference points from the nearest point to the end of global_planed_traj
        for i in range(nearest_idx, len(global_planed_traj) - 1):
            if cum_dist[i] - cum_dist[nearest_idx] >= desire_arc_length * len(ref_traj_pts):
                ref_traj_pts.append(global_planed_traj[i, :])
                if len(ref_traj_pts) == self.ref_traj_len:
                    break
        # if the target is reached before the reference trajectory is complete, add the last point of global_planed_traj 
        while len(ref_traj_pts) < self.ref_traj_len:
            ref_traj_pts.append(global_planed_traj[-1, :])
        return np.array(ref_traj_pts)

class MPC_Controller_Batch:
    def __init__(self, global_planed_trajs, N=5, desired_v=0.5, v_max=0.5, w_max=0.5, ref_gap=3):
        """
        Batch MPC Controller for parallel processing
        Args:
            global_planed_trajs: [B, T, 3] - batch of planned trajectories
        """
        self.N, self.desired_v, self.ref_gap, self.T = N, desired_v, ref_gap, 0.1
        self.batch_size = global_planed_trajs.shape[0]
        self.ref_traj_len = N // ref_gap + 1
        
        # Process all trajectories at once using vectorized operations
        self.ref_trajs = self.make_ref_denser(global_planed_trajs)
        
        # Setup batch MPC problem
        opti = ca.Opti()
        
        # Batch variables: [B, N, 2] for controls, [B, N+1, 3] for states
        opt_controls0 = opti.variable(self.batch_size, N)
        opt_controls1 = opti.variable(self.batch_size, N)
        opt_states0 = opti.variable(self.batch_size, N+1)
        opt_states1 = opti.variable(self.batch_size, N+1)
        opt_states2 = opti.variable(self.batch_size, N+1)
        
        # Batch parameters
        opt_x0 = opti.parameter(self.batch_size, 3)  # [B, 3]
        opt_xs = opti.parameter(self.batch_size, 3 * self.ref_traj_len)  # [B, 3*ref_traj_len]
        
        f = lambda x_, u_: ca.vertcat(*[u_[:, 0]*ca.cos(x_[:, 2]), u_[:, 0]*ca.sin(x_[:, 2]), u_[:, 1]])
        
        # init_condition
        opti.subject_to(opt_states0[:, 0] == opt_x0[:, 0])
        opti.subject_to(opt_states1[:, 0] == opt_x0[:, 1])
        opti.subject_to(opt_states2[:, 0] == opt_x0[:, 2])
        for i in range(N):
            x_next0 = opt_states0[:, i] + opt_controls0[:, i]*ca.cos(opt_states2[:, i])*self.T
            x_next1 = opt_states1[:, i] + opt_controls0[:, i]*ca.sin(opt_states2[:, i])*self.T
            x_next2 = opt_states2[:, i] + opt_controls1[:, i]*self.T
            opti.subject_to(opt_states0[:, i+1]==x_next0)
            opti.subject_to(opt_states1[:, i+1]==x_next1)
            opti.subject_to(opt_states2[:, i+1]==x_next2)

        obj = 0 
        for i in range(N):
            obj = obj + 0.02 * ca.mtimes([opt_controls0[:, i].T, opt_controls0[:, i]])
            obj = obj + 0.15 * ca.mtimes([opt_controls1[:, i].T, opt_controls1[:, i]])
            if i % ref_gap == 0:
                nn = i // ref_gap
                obj = obj + 10.0 * ca.mtimes([(opt_states0[:, i]-opt_xs[:, nn*3]).T, (opt_states0[:, i]-opt_xs[:, nn*3])])
                obj = obj + 10.0 * ca.mtimes([(opt_states1[:, i]-opt_xs[:, nn*3+1]).T, (opt_states1[:, i]-opt_xs[:, nn*3+1])])

        opti.minimize(obj)

        # boundrary and control conditions
        opti.subject_to(opti.bounded(0.0, opt_controls0, v_max))
        opti.subject_to(opti.bounded(-w_max, opt_controls1, w_max))
        
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        opti.solver('ipopt', opts_setting)
            
        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_controls0 = opt_controls0
        self.opt_controls1 = opt_controls1
        self.opt_states0 = opt_states0
        self.opt_states1 = opt_states1
        self.opt_states2 = opt_states2
        self.last_opt_states0 = None
        self.last_opt_states1 = None
        self.last_opt_states2 = None
        self.last_opt_controls0 = None
        self.last_opt_controls1 = None
        
    def make_ref_denser(self, ref_trajs, ratio=50):
        """Vectorized trajectory densification for batch"""
        device = ref_trajs.device
        B, L = ref_trajs.shape[:2]
        x_orig = torch.arange(L, device=device)[None].repeat(B, 1)
        xnew = torch.linspace(0, L - 1, L * ratio, device=ref_trajs.device)[None].repeat(B, 1)
        uniform_x = torchinterp1d.interp1d(x_orig, ref_trajs[..., 0], xnew, None)
        uniform_y = torchinterp1d.interp1d(x_orig, ref_trajs[..., 1], xnew, None)
        dense_traj = torch.stack((uniform_x, uniform_y), axis=-1)
        return dense_traj.cpu().numpy()
    
    def find_reference_traj_batch(self, x00_batch):
        """Vectorized reference trajectory finding for batch"""
        batch_ref_trajs = []
        for b in range(self.batch_size):
            x0 = x00_batch[b]
            global_traj = self.ref_trajs[b]
            
            ref_traj_pts = []
            nearest_idx = np.argmin(np.linalg.norm(global_traj - x0[:2].reshape((1, 2)), axis=1))
            desire_arc_length = self.desired_v * self.ref_gap * self.T
            cum_dist = np.cumsum(np.linalg.norm(np.diff(global_traj, axis=0), axis=1))
            
            for i in range(nearest_idx, len(global_traj) - 1):
                if cum_dist[i] - cum_dist[nearest_idx] >= desire_arc_length * len(ref_traj_pts):
                    ref_traj_pts.append(global_traj[i, :])
                    if len(ref_traj_pts) == self.ref_traj_len:
                        break
            
            while len(ref_traj_pts) < self.ref_traj_len:
                ref_traj_pts.append(global_traj[-1, :])
            
            # Add fake yaw angle and flatten
            ref_traj = np.array(ref_traj_pts)
            ref_traj_with_yaw = np.concatenate((ref_traj, np.zeros((ref_traj.shape[0], 1))), axis=1)
            batch_ref_trajs.append(ref_traj_with_yaw.flatten())
        
        return np.array(batch_ref_trajs)
    
    def solve(self, x00_batch_torch):
        """
        Solve batch MPC problem
        Args:
            x00_batch: [B, 3] - batch of initial states
        Returns:
            u_controls: [B, N, 2] - batch of control sequences
            x_states: [B, N+1, 3] - batch of state sequences
        """
        x00_batch = x00_batch_torch.cpu().numpy()
        # Get reference trajectories for all batch elements
        ref_trajs_batch = self.find_reference_traj_batch(x00_batch)
        
        # Set parameters
        self.opti.set_value(self.opt_x0, x00_batch)
        self.opti.set_value(self.opt_xs, ref_trajs_batch)
        
        opt_controls0 = np.zeros((self.batch_size, self.N)) if self.last_opt_controls0 is None else self.last_opt_controls0
        opt_controls1 = np.zeros((self.batch_size, self.N)) if self.last_opt_controls1 is None else self.last_opt_controls1
        opt_states0 = np.zeros((self.batch_size, self.N+1)) if self.last_opt_states0 is None else self.last_opt_states0
        opt_states1 = np.zeros((self.batch_size, self.N+1)) if self.last_opt_states1 is None else self.last_opt_states1
        opt_states2 = np.zeros((self.batch_size, self.N+1)) if self.last_opt_states2 is None else self.last_opt_states2
        
        # Warm start if available
        self.opti.set_initial(self.opt_controls0, opt_controls0)
        self.opti.set_initial(self.opt_controls1, opt_controls1)
        self.opti.set_initial(self.opt_states0, opt_states0)
        self.opti.set_initial(self.opt_states1, opt_states1)
        self.opti.set_initial(self.opt_states2, opt_states2)
        
        # Solve
        sol = self.opti.solve()
        
        # Extract results
        self.last_opt_controls0 = sol.value(self.opt_controls0).reshape(self.batch_size, -1)
        self.last_opt_controls1 = sol.value(self.opt_controls1).reshape(self.batch_size, -1)
        self.last_opt_states0 = sol.value(self.opt_states0).reshape(self.batch_size, -1)
        self.last_opt_states1 = sol.value(self.opt_states1).reshape(self.batch_size, -1)
        self.last_opt_states2 = sol.value(self.opt_states2).reshape(self.batch_size, -1)

        return torch.as_tensor(np.stack((self.last_opt_controls0, self.last_opt_controls1), axis=-1), dtype=torch.float32, device=x00_batch_torch.device)

    def reset(self):
        self.last_opt_controls0 = None
        self.last_opt_controls1 = None
        self.last_opt_states0 = None
        self.last_opt_states1 = None
        self.last_opt_states2 = None

class BaseController(ABC):
    """[summary]

    Args:
        name (str): [description]
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArticulationActions:
        """A controller should take inputs and returns an ArticulationAction to be then passed to the
           ArticulationController.

        Args:
            observations (dict): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            ArticulationAction: [description]
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets state of the controller."""
        return

class DifferentialController(BaseController):
    r"""


    This controller uses a unicycle model of a differential drive. The Controller consumes a command in the form of a linear and angular velocity, and then computes the circular arc that satisfies this command given the distance between the wheels.  This can then be used to compute the necessary angular velocities of the joints that will propell the midpoint between the wheels along the curve. The conversion is

        .. math::

            \omega_R = \\frac{1}{2r}(2V + \omega b) \n
            \omega_L = \\frac{1}{2r}(2V - \omega b)

    where :math:`\omega` is the desired angular velocity, :math:`V` is the desired linear velocity, :math:`r` is the radius of the wheels, and :math:`b` is the distance between them.


    Args:
        name (str): [description]
        wheel_radius (float): Radius of left and right wheels in cms
        wheel_base (float): Distance between left and right wheels in cms
        max_linear_speed (float): OPTIONAL: limits the maximum linear speed that will be produced by the controller. Defaults to 1E20.
        max_angular_speed (float): OPTIONAL: limits the maximum angular speed that will be produced by the controller. Defaults to 1E20.
        max_wheel_speed (float): OPTIONAL: limits the maximum wheel speed that will be produced by the controller. Defaults to 1E20.
    """
    def __init__(
        self,
        name: str,
        wheel_radius: float,
        wheel_base: float,
        max_linear_speed: float = 1.0e20,
        max_angular_speed: float = 1.0e20,
        max_wheel_speed: float = 1.0e20,
    ) -> None:
        super().__init__(name)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_wheel_speed = max_wheel_speed

        assert self.max_linear_speed >= 0
        assert self.max_angular_speed >= 0
        assert self.max_wheel_speed >= 0

    def forward(self, command: np.ndarray) -> ArticulationActions:
        """convert from desired [signed linear speed, signed angular speed] to [Left Drive, Right Drive] joint targets.

        Args:
            command (np.ndarray): desired vehicle [forward, rotation] speed

        Returns:
            ArticulationAction: the articulation action to be applied to the robot.
        """
        if isinstance(command, list):
            command = np.array(command)
        if command.shape[0] != 2:
            raise Exception("command should be of length 2")

        # limit vehicle speed
        command = np.clip(
            command,
            a_min=[-self.max_linear_speed, -self.max_angular_speed],
            a_max=[self.max_linear_speed, self.max_angular_speed],
        )
        # calculate wheel speed
        joint_velocities = [0.0, 0.0]
        joint_velocities[0] = ((2 * command[0]) - (command[1] * self.wheel_base)) / (2 * self.wheel_radius)
        joint_velocities[1] = ((2 * command[0]) + (command[1] * self.wheel_base)) / (2 * self.wheel_radius)
        joint_velocities = np.clip(
            joint_velocities,
            a_min=[-self.max_wheel_speed, -self.max_wheel_speed],
            a_max=[self.max_wheel_speed, self.max_wheel_speed],
        )
        return ArticulationActions(joint_velocities=joint_velocities)

    def forward_batch(self, commands: np.ndarray) -> torch.Tensor:
        """convert from desired [signed linear speed, signed angular speed] to [Left Drive, Right Drive] joint targets.

        Args:
            commands (np.ndarray): desired vehicle [forward, rotation] speed for a batch of commands

        Returns:
            torch.Tensor: the raw tensor of joint velocities for the batch.
        """
        # calculate wheel speed
        joint_velocities = np.zeros((commands.shape[0], 2))
        joint_velocities[:, 0] = ((2 * commands[:, 0]) - (commands[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)
        joint_velocities[:, 1] = ((2 * commands[:, 0]) + (commands[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)

        return torch.tensor(joint_velocities, dtype=torch.float32)
    
    def forward_torch(self, commands):
        joint_velocities = torch.stack([
            ((2 * commands[:, 0]) - (commands[:, 1] * self.wheel_base)) / (2 * self.wheel_radius),
            ((2 * commands[:, 0]) + (commands[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)
        ], dim=-1)
        return joint_velocities
