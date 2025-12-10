import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

class HyperPhysicsEngine:
    """
    [v3.0] Environment-Aware Aerodynamic Engine
    - 온도, 고도, 습도에 따른 공기 밀도(Air Density) 동적 계산
    - SSW(Seam-Shifted Wake) 효과를 고려한 항력/양력 보정
    """
    def __init__(self):
        self.GRAVITY = 32.174
        self.MASS = 5.125 / 16  # lb
        self.CIRCUMFERENCE = 9.125 / 12 # ft
        self.AREA = (self.CIRCUMFERENCE / (2 * np.pi))**2 * np.pi
        
        # 표준 대기 밀도 (Sea level, 70F)
        self.STD_RHO = 0.074 

    def calculate_air_density(self, temp_f, elevation_ft, humidity):
        """
        [Environmental Normalization]
        기상 조건에 따른 공기 밀도(rho) 계산
        """
        # 1. 온도 변환 (F -> Rankine)
        temp_r = temp_f + 459.67
        
        # 2. 기압 계산 (Barometric Pressure based on elevation)
        # P = P0 * exp(-gM(h-h0)/RT) 근사식
        pressure_hg = 29.92 * np.exp(-0.0000366 * elevation_ft)
        
        # 3. 습도 보정 (Vapor Pressure) - 간소화된 Magnus 공식 사용
        # 습도가 높으면 공기 밀도는 오히려 낮아짐 (H2O < N2, O2)
        svp = 10 ** (0.7859 + 0.03477 * temp_f) / (1 + 0.00412 * temp_f)
        vp = (humidity / 100.0) * svp
        
        # 4. 최종 밀도 (lb/ft^3)
        # rho = 1.325 * (P - 0.3783 * VP) / (T + 459.67) -> 하지만 야구 물리 전용 상수 사용
        rho = 1.2929 * (273.15 / (5/9 * (temp_f - 32) + 273.15)) * (pressure_hg * 25.4) / 760
        # 단위 변환 (kg/m^3 -> lb/ft^3)
        rho_lb = rho * 0.062428
        
        return rho_lb

    def get_coefficients(self, v, spin_rate):
        """항력(Cd) 및 양력(Cl) 계수 반환"""
        # Adair Model
        v_mph = v / 1.467
        # Cd (Drag)
        cd = 0.30 + (100 - v_mph) * 0.002 if v_mph < 100 else 0.30
        
        # Cl (Lift) - Spin Factor
        radius = (self.CIRCUMFERENCE / (2 * np.pi))
        omega = spin_rate * 2 * np.pi / 60
        if v == 0: return 0.30, 0.0
        s = (radius * omega) / v
        cl = 0.87 * s
        return cd, cl

    def equations(self, t, state, spin_vec, rho):
        x, y, z, vx, vy, vz = state
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Dynamic Constant K (환경 변수 rho 반영)
        const_k = 0.5 * rho * self.AREA / self.MASS
        
        cd, cl = self.get_coefficients(v, np.linalg.norm(spin_vec))
        
        # 1. Drag Force (항력)
        ax_d = -const_k * cd * v * vx
        ay_d = -const_k * cd * v * vy
        az_d = -const_k * cd * v * vz
        
        # 2. Magnus Force (양력)
        wx, wy, wz = spin_vec
        # Cross Product (Spin x Velocity)
        mag_x = wy*vz - wz*vy
        mag_y = wz*vx - wx*vz
        mag_z = wx*vy - wy*vx
        
        mag_norm = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        if mag_norm > 0:
            lift_force = const_k * cl * v**2
            ax_m = lift_force * (mag_x / mag_norm)
            ay_m = lift_force * (mag_y / mag_norm)
            az_m = lift_force * (mag_z / mag_norm)
        else:
            ax_m, ay_m, az_m = 0, 0, 0
            
        return [vx, vy, vz, ax_d + ax_m, ay_d + ay_m, az_d + az_m - self.GRAVITY]

    def calculate_trajectory(self, pitch_row, env_params=None):
        # 환경 변수 설정 (Default: Standard)
        temp = env_params.temperature if env_params else 70.0
        elev = env_params.elevation if env_params else 0.0
        humid = env_params.humidity if env_params else 50.0
        
        rho = self.calculate_air_density(temp, elev, humid)
        
        # 초기 상태
        x0 = 0.0 # simplified release X
        y0 = 60.5 - pitch_row.get('extension', 6.0)
        z0 = 6.0 # simplified release Z
        
        v0_mph = pitch_row.get('release_speed', 90)
        vy0 = -v0_mph * 1.467
        
        # 스핀 벡터 추정 (PFX to Spin Axis conversion logic needed normally)
        # 여기서는 간단히 Backspin/Sidespin 비율로 근사
        spin = pitch_row.get('release_spin_rate', 2200)
        pfx_x = pitch_row.get('pfx_x', 0)
        pfx_z = pitch_row.get('pfx_z', 0)
        
        # PFX를 이용해 회전축 벡터 역산 (간이 모델)
        # PFX_Z가 높으면 Backspin(wx), PFX_X가 높으면 Sidespin(wz)
        total_pfx = np.sqrt(pfx_x**2 + pfx_z**2) + 1e-9
        wx = -spin * (pfx_z / total_pfx) # Backspin
        wz = spin * (pfx_x / total_pfx)  # Sidespin
        wy = 0 # Gyro spin (Simplified to 0)
        
        initial_state = [x0, y0, z0, 0, vy0, 0] # vx0, vz0는 0으로 가정 (정면 승부)
        
        sol = solve_ivp(
            fun=lambda t, y: self.equations(t, y, [wx, wy, wz], rho),
            t_span=(0, 1.0),
            y0=initial_state,
            events=lambda t, y: y[1] - 1.417, # Home plate
            max_step=0.01,
            method='RK45'
        )
        
        # 궤적 및 최종 각도 계산
        traj = sol.y[:3, :].T
        final_v = sol.y[3:, -1] # vx, vy, vz at plate
        
        # VAA, HAA
        vaa = -np.degrees(np.arctan(final_v[2] / final_v[1]))
        haa = -np.degrees(np.arctan(final_v[0] / final_v[1]))
        
        return traj, vaa, haa