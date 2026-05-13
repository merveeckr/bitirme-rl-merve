"""
JSBSimHelicopterEnv — JSBSim tabanlı helikopter uçuş ortamı
============================================================
FlightControlEnv3D'nin JSBSim FDM kullanan versiyonu.

Mimari:
  - Tutum (roll/pitch/yaw) : Python kinematik modeli (base_env gibi)
  - Translasyon (x/y/z)    : JSBSim 6-DOF rigid body + yerçekimi
  - Kuvvetler              : LOCAL (NED) çerçevede ExternalReactions

Eylem alanı (4-dim, [-1, 1]):  base_env.py ile özdeş:
  [0] roll_rate    → ±45 deg/s
  [1] pitch_rate   → ±45 deg/s
  [2] yaw_rate     → ±60 deg/s
  [3] vert_vel     → collective (dikey kuvvet kontrolü)

JSBSim'in katkısı (base_env'den farkı):
  - Gerçek yerçekimi (9.81 m/s² JSBSim tarafından uygulanır)
  - Gerçek rijit cisim ataletini yansıtan hız entegrasyonu
  - Koordinat sistemi: gerçek lat/lon → yerel ENU metre (sim-to-real için)
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import jsbsim
    _JSBSIM_OK = True
except ImportError:
    _JSBSIM_OK = False

# ── Birim dönüşümleri ─────────────────────────────────────────────────────
FT2M      = 0.3048
M2FT      = 1.0 / FT2M
LBF_PER_N = 0.224809
R_EARTH   = 6_371_000.0

# ── JSBSim simülasyon adımı ───────────────────────────────────────────────
_JSBSIM_DT  = 0.02
_RL_DT      = 0.1
_N_SUBSTEPS = int(_RL_DT / _JSBSIM_DT)   # 5

# ── Helikopter fizik parametreleri ────────────────────────────────────────
_MASS_KG      = 630.0
_G            = 9.81
_W_N          = _MASS_KG * _G             # hover kaldırma = 6 180 N

_MAX_TILT     = 30.0     # maks eğim açısı [deg]
_MAX_TILT_RATE= 45.0     # [deg/s]
_MAX_YAW_RATE = 60.0     # [deg/s]
_MAX_VERT     = 5.0      # [m/s]
_MAX_HORIZ    = 20.0     # [m/s]
_VEL_ALPHA    = 0.2      # hız filtresi (base_env ile aynı)

# ── JSBSim kök dizini ─────────────────────────────────────────────────────
_JSB_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', 'jsbsim_root'))


class JSBSimHelicopterEnv(gym.Env):
    """
    JSBSim ile güçlendirilmiş 3D helikopter navigasyon ortamı.

    obstacle_env.ObstacleHelicopterEnv bu sınıfı miras alır
    (FlightControlEnv3D ile aynı arayüz).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode=None,
        max_steps: int = 1000,
        world_size: float = 500.0,
        wind_strength_max: float = 3.0,
        target_radius: float = 10.0,
    ):
        super().__init__()

        if not _JSBSIM_OK:
            raise ImportError(
                "JSBSim bulunamadı.\n"
                "  pip install jsbsim\n"
                "komutunu çalıştır.")

        # ── Dünya parametreleri ───────────────────────────────────────
        self.world_size        = world_size
        self.min_altitude      = 5.0
        self.max_altitude      = 200.0
        self.target_radius     = target_radius
        self.max_steps         = max_steps
        self.wind_strength_max = wind_strength_max
        self.render_mode       = render_mode

        self.dt = _RL_DT

        # base_env ile özdeş referans değerler
        self.max_tilt        = _MAX_TILT
        self.max_tilt_rate   = _MAX_TILT_RATE
        self.max_yaw_rate    = _MAX_YAW_RATE
        self.max_horiz_speed = _MAX_HORIZ
        self.max_vert_speed  = _MAX_VERT

        # ── Gym uzayları ──────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

        # İç durum değişkenleri
        self._fdm      = None
        self._ref_lat  = 0.0
        self._ref_lon  = 0.0

        self._init_state()

    # ──────────────────────────────────────────────────────────────────
    # İç durum
    # ──────────────────────────────────────────────────────────────────

    def _init_state(self):
        self.pos        = np.zeros(3, dtype=np.float64)
        self.vel        = np.zeros(3, dtype=np.float64)
        self.roll       = 0.0
        self.pitch      = 0.0
        self.yaw        = 0.0
        self.roll_rate  = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate   = 0.0
        self.target     = np.zeros(3, dtype=np.float64)
        self.wind       = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.prev_dist  = 0.0
        self.trajectory: list = []
        self.episode_start = np.zeros(2, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────
    # JSBSim başlatma
    # ──────────────────────────────────────────────────────────────────

    def _make_fdm(self):
        if self._fdm is not None:
            del self._fdm
            self._fdm = None
        fdm = jsbsim.FGFDMExec(_JSB_ROOT, None)
        fdm.set_debug_level(0)
        ok = fdm.load_model('heli_rl')
        if not ok:
            raise RuntimeError(
                f"JSBSim modeli yüklenemedi: {_JSB_ROOT}/aircraft/heli_rl/heli_rl.xml")
        fdm.set_dt(_JSBSIM_DT)
        return fdm

    # ──────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()

        # Rastgele başlangıç
        x0 = float(self.np_random.uniform(-150.0, 150.0))
        y0 = float(self.np_random.uniform(-150.0, 150.0))
        z0 = float(self.np_random.uniform(20.0, 60.0))

        # Rastgele hedef (80–380 m mesafe)
        for _ in range(200):
            xt = float(self.np_random.uniform(-200.0, 200.0))
            yt = float(self.np_random.uniform(-200.0, 200.0))
            zt = float(self.np_random.uniform(15.0, 100.0))
            if 80.0 <= np.linalg.norm([xt-x0, yt-y0, zt-z0]) <= 380.0:
                break
        self.target = np.array([xt, yt, zt], dtype=np.float64)

        # Rastgele rüzgar
        ws = float(self.np_random.uniform(0.0, self.wind_strength_max))
        wd = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self.wind = np.array([
            ws * np.cos(wd),
            ws * np.sin(wd),
            float(self.np_random.uniform(-0.3, 0.3)),
        ])

        yaw0 = float(self.np_random.uniform(0.0, 360.0))

        # JSBSim başlangıç koşulları
        fdm = self._make_fdm()
        self._fdm = fdm

        # Global ENU ↔ JSBSim lat/lon dönüşümü
        # Referans DAIMA (0°,0°) = global origin.
        # x0,y0 ise bu origin'den metre cinsinden ofset.
        self._ref_lat = 0.0
        self._ref_lon = 0.0

        lat0_rad = y0 / R_EARTH             # kuzey offseti → latitude
        lon0_rad = x0 / R_EARTH             # doğu offseti → longitude (cos(0°)=1)

        fdm['ic/lat-geod-deg'] = float(np.degrees(lat0_rad))
        fdm['ic/long-gc-deg']  = float(np.degrees(lon0_rad))
        fdm['ic/h-sl-ft']      = float(z0 * M2FT)
        fdm['ic/psi-true-deg'] = float(yaw0)
        fdm['ic/u-fps']        = 0.0
        fdm['ic/v-fps']        = 0.0
        fdm['ic/w-fps']        = 0.0
        fdm['ic/p-rad_sec']    = 0.0
        fdm['ic/q-rad_sec']    = 0.0
        fdm['ic/r-rad_sec']    = 0.0

        fdm.run_ic()

        # Sıfır eylemle ilk kuvvetleri uygula (hovering başlangıcı)
        self._apply_forces(np.zeros(4, dtype=np.float32))

        self.pos  = np.array([x0, y0, z0], dtype=np.float64)
        self.yaw  = yaw0
        self.step_count   = 0
        self.prev_dist    = float(np.linalg.norm(self.target - self.pos))
        self.episode_start = self.pos[:2].copy()
        self.trajectory   = [self.pos.copy()]

        return self._obs(), {}

    # ──────────────────────────────────────────────────────────────────
    # Tutum kinematik modeli (base_env ile özdeş)
    # ──────────────────────────────────────────────────────────────────

    def _update_attitude(self, action):
        """roll/pitch/yaw Python kinematik modeliyle günceller."""
        self.roll_rate  = float(action[0]) * _MAX_TILT_RATE
        self.pitch_rate = float(action[1]) * _MAX_TILT_RATE
        self.yaw_rate   = float(action[2]) * _MAX_YAW_RATE
        vert_cmd        = float(action[3]) * _MAX_VERT

        self.roll  = float(np.clip(
            self.roll  + self.roll_rate  * _RL_DT, -_MAX_TILT, _MAX_TILT))
        self.pitch = float(np.clip(
            self.pitch + self.pitch_rate * _RL_DT, -_MAX_TILT, _MAX_TILT))
        self.yaw   = (self.yaw + self.yaw_rate * _RL_DT) % 360.0

        return vert_cmd

    # ──────────────────────────────────────────────────────────────────
    # Kuvvet enjeksiyonu — LOCAL (NED) çerçevesinde
    # ──────────────────────────────────────────────────────────────────

    def _apply_forces(self, action):
        """
        Tutum bilgisine göre LOCAL (NED) çerçevede net kuvvetleri hesaplar
        ve JSBSim ExternalReactions'a enjekte eder.

        JSBSim yerçekimini zaten uygular (emptywt × g).
        Biz yalnızca: kaldırma − ağırlık = net dikey kuvvet
        ve yatay bileşenler (eğim + sürükleme + rüzgar) enjekte ederiz.
        """
        fdm   = self._fdm
        a     = np.clip(action, -1.0, 1.0)
        yr    = np.deg2rad(self.yaw)
        pr    = np.deg2rad(self.pitch)
        rr    = np.deg2rad(self.roll)

        # ── Collective → toplam rotor itiş kuvveti ───────────────────
        vert_cmd = float(a[3]) * _MAX_VERT  # m/s hedef dikey hız
        # Kaldırma = hover + dikey komut orantılı ekstra kuvvet
        extra_up = vert_cmd * _MASS_KG * 2.0   # N — orantısal kontrol
        T_N = _W_N + extra_up
        T_N = float(np.clip(T_N, _W_N * 0.2, _W_N * 1.8))

        # ── Tilt → yatay kuvvet bileşenleri (NED) ────────────────────
        # Pitch ileri (a[1]>0) → kuzey veya ileri yön
        # Roll sağa  (a[0]>0) → doğu veya sağ yön
        pf = self.pitch / _MAX_TILT   # [-1, 1]
        rf = self.roll  / _MAX_TILT

        # Helikopter başlığına göre ileri/sağ bileşenler
        F_fwd_N   = T_N * pf * np.sin(np.deg2rad(_MAX_TILT))
        F_right_N = T_N * rf * np.sin(np.deg2rad(_MAX_TILT))

        F_north_N =  F_fwd_N * np.cos(yr) - F_right_N * np.sin(yr)
        F_east_N  =  F_fwd_N * np.sin(yr) + F_right_N * np.cos(yr)
        # JSBSim yerçekimini (emptywt × g) zaten uygular.
        # Biz sadece rotor kaldırmasını (tam değeriyle) enjekte ederiz;
        # net kuvvet = T_N*cos(θ)cos(φ) − W, JSBSim tarafından hesaplanır.
        F_up_N = T_N * np.cos(pr) * np.cos(rr)

        # ── Aerodinamik sürükleme ─────────────────────────────────────
        DRAG = 80.0   # N / (m/s)
        vn = fdm['velocities/v-north-fps'] * FT2M
        ve = fdm['velocities/v-east-fps']  * FT2M
        vd = fdm['velocities/v-down-fps']  * FT2M
        F_north_N -= DRAG * vn
        F_east_N  -= DRAG * ve
        F_up_N    += DRAG * vd   # vd pozitif = aşağı → drag yukarı

        # ── Rüzgar ───────────────────────────────────────────────────
        WIND_COEF = 15.0   # N / (m/s)
        F_north_N += self.wind[1] * WIND_COEF   # wind[1] = kuzey
        F_east_N  += self.wind[0] * WIND_COEF   # wind[0] = doğu

        # ── JSBSim'e enjekte et (lbf) ─────────────────────────────────
        fdm['external_reactions/force_up/magnitude']    = F_up_N    * LBF_PER_N
        fdm['external_reactions/force_north/magnitude'] = F_north_N * LBF_PER_N
        fdm['external_reactions/force_east/magnitude']  = F_east_N  * LBF_PER_N

    # ──────────────────────────────────────────────────────────────────
    # Durum okuma — sadece pozisyon ve dünya-çerçevesi hız
    # (tutum Python kinematik modelinden gelir, JSBSim'den okunmaz)
    # ──────────────────────────────────────────────────────────────────

    def _read_state(self):
        fdm = self._fdm

        # Konum: JSBSim lat/lon → yerel ENU metre
        lat = fdm['position/lat-geod-rad']
        lon = fdm['position/long-gc-rad']
        alt = fdm['position/h-sl-ft'] * FT2M

        dy = (lat - self._ref_lat) * R_EARTH
        dx = (lon - self._ref_lon) * R_EARTH * np.cos(self._ref_lat)
        self.pos = np.array([dx, dy, float(alt)], dtype=np.float64)

        # Hız: NED fps → ENU m/s
        vn = fdm['velocities/v-north-fps'] * FT2M
        ve = fdm['velocities/v-east-fps']  * FT2M
        vd = fdm['velocities/v-down-fps']  * FT2M
        self.vel = np.array([ve, vn, -vd], dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────
    # Gözlem
    # ──────────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        delta   = self.target - self.pos
        dist    = float(np.linalg.norm(delta)) + 1e-8
        dir_vec = (delta / dist).astype(np.float32)

        obs = np.array([
            dir_vec[0], dir_vec[1], dir_vec[2],              # 0-2
            dist / self.world_size,                           # 3
            self.vel[0] / self.max_horiz_speed,               # 4
            self.vel[1] / self.max_horiz_speed,               # 5
            self.vel[2] / self.max_vert_speed,                # 6
            self.roll   / self.max_tilt,                      # 7
            self.pitch  / self.max_tilt,                      # 8
            np.sin(np.deg2rad(self.yaw)),                     # 9
            np.cos(np.deg2rad(self.yaw)),                     # 10
            self.roll_rate  / self.max_tilt_rate,             # 11
            self.pitch_rate / self.max_tilt_rate,             # 12
            self.yaw_rate   / self.max_yaw_rate,              # 13
            self.wind[0] / 10.0,                              # 14
            self.wind[1] / 10.0,                              # 15
            self.wind[2] / 3.0,                               # 16
            self.step_count / self.max_steps,                 # 17
            # [18-25] LiDAR — alt sınıf doldurur (varsayılan = 1.0 = açık)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)

        return obs

    # ──────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────

    def step(self, action):
        # 1) Tutumu Python'da güncelle (roll/pitch/yaw)
        self._update_attitude(action)

        # 2) JSBSim substep'leri: her substep'te kuvvet enjekte et + çalıştır
        fdm = self._fdm
        for _ in range(_N_SUBSTEPS):
            self._apply_forces(action)
            fdm.run()

        # 3) Pozisyon ve hızı JSBSim'den oku
        self._read_state()

        # Zemin tabanı
        if self.pos[2] < self.min_altitude:
            self.pos[2] = self.min_altitude
            self.vel[2] = max(0.0, self.vel[2])

        self.step_count += 1
        self.trajectory.append(self.pos.copy())

        reward, terminated, info = self._reward()
        truncated = self.step_count >= self.max_steps
        return self._obs(), reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────
    # Ödül
    # ──────────────────────────────────────────────────────────────────

    def _reward(self):
        curr_dist  = float(np.linalg.norm(self.target - self.pos))
        info: dict = {}
        terminated = False

        # İlerleme
        r_progress = (self.prev_dist - curr_dist) * 10.0
        self.prev_dist = curr_dist

        # Hedefe varış
        r_goal = 0.0
        if curr_dist < self.target_radius:
            r_goal     = 500.0
            terminated = True
            info["success"] = True

        # Denge stabilitesi
        r_stability = -(abs(self.roll)  / self.max_tilt +
                        abs(self.pitch) / self.max_tilt) * 0.5

        # Minimum irtifa
        r_altitude = -2.0 if self.pos[2] <= self.min_altitude + 2.0 else 0.0

        # Sınır dışı
        r_bounds = 0.0
        if np.any(np.abs(self.pos[:2]) > self.world_size):
            r_bounds   = -200.0
            terminated = True
            info["out_of_bounds"] = True

        # Zaman cezası
        r_time = -0.1

        # Yol sapma ödülü
        r_path = self._path_deviation_reward()

        total = r_progress + r_goal + r_stability + r_altitude + r_bounds + r_time + r_path
        return total, terminated, info

    def _path_deviation_reward(self) -> float:
        path_vec = self.target[:2] - self.episode_start
        path_len = float(np.linalg.norm(path_vec))
        if path_len < 1e-3:
            return 0.0
        path_dir    = path_vec / path_len
        pos_vec     = self.pos[:2] - self.episode_start
        proj        = float(np.dot(pos_vec, path_dir))
        lateral     = pos_vec - proj * path_dir
        return -0.02 * float(np.linalg.norm(lateral)) / 100.0

    # ──────────────────────────────────────────────────────────────────
    # Yardımcılar
    # ──────────────────────────────────────────────────────────────────

    def get_trajectory(self) -> np.ndarray:
        return np.array(self.trajectory)

    def render(self):
        pass

    def close(self):
        if self._fdm is not None:
            del self._fdm
            self._fdm = None
