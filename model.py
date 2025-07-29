import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import (VehicleAttitude, VehicleImu, ActuatorOutputs, ActuatorMotors, 
                          VehicleLocalPosition,SensorCombined,VehicleAngularVelocity, 
                          VehicleAngularAccelerationSetpoint, VehicleMagnetometer, SensorBaro, EscStatus)
import numpy as np

 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
 
from filterpy.kalman import ExtendedKalmanFilter
from datetime import datetime
from sensor_msgs.msg import Imu, MagneticField, FluidPressure
import os

import csv
import time
from std_msgs.msg import String
from datetime import datetime
   
from quad_flip_msgs.msg import OptimizedTraj
from rclpy.qos import QoSProfile

import threading
from jax import jit, grad, jacobian, hessian, vmap, lax
import jax.numpy as jnp 
from scipy.spatial.transform import Rotation as Rot 
 

# ============ CONSTANTS ====================================================
SEA_LEVEL_PRESSURE = 101325.0
EKF_DT = 0.01
# ============ DRONE CONSTRUCT PARAMETERS ===================================
MASS = 0.82
INERTIA = np.diag([0.045, 0.045, 0.045])
ARM_LEN = 0.15
K_THRUST = 1.48e-6
K_TORQUE = 9.4e-8
MOTOR_TAU = 0.02
MAX_SPEED = 2100.0
DRAG = 0.1
MAX_RATE = 25.0  # ограничение на угловую скорость (roll/pitch) рад/с

# ============ Гиперпараметры для ModelPredictiveController =========================================
dt = 0.1
horizon = 10  # Горизонт предсказания
n = 13        # Размерность состояния квадрокоптера (позиция, скорость, ориентация, угловая скорость) 
m = 4         # Размерность управления (4 мотора)

# ****************** Настройка стоимостей iLQR ******************
Q = jnp.diag(jnp.array([
    1.0, 1.0, 10.0,       # x, y — менее важны, z — важна
    1.0, 1.0, 1.0,        # vx, vy, vz
    0.0, 50.0, 50.0, 0.0, # ориентация
    5.0, 5.0, 1.0         # угловые скорости
]))

R = jnp.diag(jnp.array([
        0.001, 0.001, 0.001, 0.001  # все моторы слабо штрафуются
    ]))

Qf = jnp.diag(jnp.array([
    1.0, 1.0, 10.0,       # позиции: x, y — меньше важны, z — важна
    0.1, 0.1, 0.1,        # скорости
    0.0, 100.0, 100.0, 0.0, # ориентация (qx, qy)
    10.0, 10.0, 1.0       # угловые скорости
]))
 
# ===== MATRIX OPERTIONS =====
# QUATERNION UTILS (SCIPY-based)
def quat_to_rot_matrix_numpy(quat):
    # Кватернион: [w, x, y, z]
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])
    return R

def quat_multiply_numpy(q, r):
    # Кватернионы [w, x, y, z]
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])
 
def f_numpy(x, u, dt):
    m = MASS
    I = INERTIA
    arm = ARM_LEN
    kf = K_THRUST
    km = K_TORQUE
    drag = DRAG
    g = np.array([0.0, 0.0, 9.81])
    max_speed = MAX_SPEED

    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    omega = x[10:13]

    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        quat = quat / quat_norm

    R_bw = quat_to_rot_matrix_numpy(quat)

    rpm = np.clip(u, 0.0, max_speed)
    w_squared = rpm ** 2
    thrusts = kf * w_squared

    Fz_body = np.array([0.0, 0.0, np.sum(thrusts)])
    F_world = R_bw @ Fz_body - m * g - drag * vel
    acc = F_world / m

    new_vel = vel + acc * dt
    new_pos = pos + vel * dt + 0.5 * acc * dt ** 2

    tau = np.array([
        arm * (thrusts[1] - thrusts[3]),
        arm * (thrusts[2] - thrusts[0]),
        km * (w_squared[0] - w_squared[1] + w_squared[2] - w_squared[3])
    ])

    omega_cross = np.cross(omega, I @ omega)
    omega_dot = np.linalg.solve(I, tau - omega_cross)
    new_omega = omega + omega_dot * dt

    omega_quat = np.concatenate(([0.0], new_omega))
    dq = 0.5 * quat_multiply_numpy(quat, omega_quat)
    new_quat = quat + dq * dt
    new_quat /= np.linalg.norm(new_quat) + 1e-8  # безопасная нормализация

    x_next = np.concatenate([new_pos, new_vel, new_quat, new_omega])
    return x_next

@jit
def quat_multiply(q1, q2):
    """
    Умножение кватернионов q1 * q2
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z])

@jit
def quat_to_rot_matrix(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return jnp.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

@jit
def f(x, u, dt):
    m = MASS
    I = INERTIA
    arm = ARM_LEN
    kf = K_THRUST
    km = K_TORQUE
    drag = DRAG
    g = jnp.array([0.0, 0.0, 9.81])
    max_speed = MAX_SPEED

    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    omega = x[10:13]

    # нормализация кватерниона через jax.lax.cond
    quat_norm = jnp.linalg.norm(quat)
    quat = lax.cond(
        quat_norm < 1e-8,
        lambda _: jnp.array([1.0, 0.0, 0.0, 0.0]),
        lambda _: quat / quat_norm,
        operand=None
    )

    R_bw = quat_to_rot_matrix(quat)
    rpm = jnp.clip(u, 0.0, max_speed)
    w_squared = rpm ** 2
    thrusts = kf * w_squared

    Fz_body = jnp.array([0.0, 0.0, jnp.sum(thrusts)])
    F_world = R_bw @ Fz_body - m * g - drag * vel

    acc = F_world / m
    new_vel = vel + acc * dt
    new_pos = pos + vel * dt + 0.5 * acc * dt ** 2

    tau = jnp.array([
        arm * (thrusts[1] - thrusts[3]), # Roll: правый - левый
        arm * (thrusts[2] - thrusts[0]), # Pitch: задний - передний
        km * (w_squared[0] - w_squared[1] + w_squared[2] - w_squared[3]) # Yaw
    ])

    omega_cross = jnp.cross(omega, I @ omega)
    omega_dot = jnp.linalg.solve(I, tau - omega_cross)
    new_omega = omega + omega_dot * dt

    omega_quat = jnp.concatenate([jnp.array([0.0]), new_omega])
    dq = 0.5 * quat_multiply(quat, omega_quat)
    new_quat = quat + dq * dt
    new_quat /= jnp.linalg.norm(new_quat + 1e-8)  # безопасная нормализация

    x_next = jnp.concatenate([new_pos, new_vel, new_quat, new_omega])
    return x_next

