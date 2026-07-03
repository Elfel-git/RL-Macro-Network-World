"""
Vectorized Financial Environment
Mô phỏng bài toán Financial Precarity (Nokhiz et al., 2024)
Sử dụng NumPy vectorization để chạy 10,000+ agents không dùng vòng lặp for.
"""

import numpy as np
from scipy import interpolate as intrp


# ──────────────────────────────────────────────
# DỮ LIỆU THU NHẬP & TIẾT KIỆM (từ notebook gốc)
# ──────────────────────────────────────────────

# Percentile income data from 2019 US labor statistics (annual, in USD)
INC_PERC_2019 = np.array([
    0, 610.12, 4067.48, 7118.1, 9151.84, 10168.71, 11185.58, 12308.21,
    13592.52, 14849.37, 15912, 17143.43, 18303.68, 19341.91, 20337.42,
    21386.83, 22585.73, 24021.55, 25197.05, 25986.14, 27292.82, 28472.39,
    29710.94, 30510.2, 31727.4, 32844.94, 34237.04, 35590.49, 36511.78,
    37624.23, 38702.12, 40381.99, 41062.27, 42664.86, 43766.13, 45351.44,
    46320.52, 47792.94, 49114.88, 50843.56, 51120.15, 52674.94, 53953.15,
    55816.06, 56944.79, 58404, 59995.4, 61027.52, 62557.91, 64093.39,
    65953.25, 67225.35, 69039.45, 70736.61, 71992.44, 73744.51, 75899.26,
    77109.34, 78874.63, 80867.7, 82366.56, 84420.64, 86474.72, 88471.86,
    90961.16, 92536.29, 94841.54, 97184.41, 99805.9, 101891.51, 104403.18,
    106873.16, 109407.2, 112059.2, 114916.61, 118074, 121705.24, 124363.34,
    128125.77, 132193.25, 135856.02, 140328.22, 144539.08, 149581.75,
    153673.64, 159080.34, 165244.62, 171851.23, 178813.74, 187307.67,
    197314.7, 207041.07, 219817.04, 235075.19, 252493.18, 276588.96,
    310655.16, 365232.67, 483131.76
], dtype=float)

PRCTG = np.array([
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
    0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
    0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
    0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
    0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
    0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70,
    0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
    0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
    0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99
], dtype=float)

SAVING_VAL = np.array([
    -94516.52, -54867.26, -35734.82, -25936.24, -18386.58, -11926.06,
    -7469.49, -3835.64, -1832.38, -466.58, 3.43, 178.2, 502.44, 860.92,
    1577.19, 2592.8, 3452.04, 4480.43, 5378.37, 6368.35, 7363.47,
    8511.79, 9689.16, 10777.34, 12430.11, 14197.79, 16060.99, 18106.62,
    20340.88, 23695.62, 27250.07, 30994.35, 36382.96, 40616.62, 45081.82,
    50409.19, 55408.58, 59543.95, 63103.9, 67469.07, 71802.48, 75801.39,
    80406.68, 84408.2, 89548.35, 94983.2, 100589.66, 106335.51, 114194.83,
    121411.37, 127446.43, 134207.73, 141621.65, 150028.88, 158430.46,
    167469.68, 174943.76, 182100.06, 191929.7, 201311.36, 211096.7,
    219405.74, 228563.2, 238313.46, 249137.43, 260147.01, 271875.08,
    288498.14, 301999.85, 314920.61, 328617.37, 349362.49, 365919.18,
    382911.64, 403283.56, 428623.03, 455610.73, 485176.88, 523925.5,
    558189.68, 591350.95, 637050.12, 681782.41, 737122.98, 795218.85,
    854908.75, 928665.81, 991188.75, 1085969.92, 1219126.46, 1355268.26,
    1541905.98, 1767510.16, 2080569.86, 2584130.26, 3294388.49,
    4640603.15, 6557022.79, 11099166.07
], dtype=float)


class VectorizedFinancialEnv:
    """
    Vectorized Environment mô phỏng bài toán Income Fluctuation (IFP)
    với N agents hoạt động song song.

    Parameters
    ----------
    n_agents : int
        Số lượng agent (mặc định 10,000)
    r : float
        Lãi suất phi rủi ro (mặc định 1.02)
    c_min : float
        Mức tiêu dùng tối thiểu (nếu None, dùng basic_expenditure)
    grid_max : float
        Giá trị tối đa của lưới tài sản (dùng để nội suy chính sách tiêu dùng)
    grid_size : int
        Kích thước lưới nội suy
    seed : int
        Seed ngẫu nhiên
    shock_perm_gap : int
        Chu kỳ shock vĩnh viễn (tháng)
    shock_temp_gap : int
        Chu kỳ shock tạm thời (tháng)
    shock_perm_size : float
        Mức độ shock vĩnh viễn (vd 0.4 = 40%)
    shock_temp_size : float
        Mức độ shock tạm thời (vd 0.6 = 60%)
    shock_prob : float
        Xác suất xảy ra shock mỗi chu kỳ
    """

    def __init__(
        self,
        n_agents=10_000,
        r=1.02,
        c_min=None,
        grid_max=500,
        grid_size=10_000,
        seed=42,
        shock_perm_gap=25,
        shock_temp_gap=20,
        shock_perm_size=0.4,
        shock_temp_size=0.6,
        shock_prob=0.3,
    ):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.n_agents = n_agents
        self.r = r
        self.c_min = c_min  # None = dùng basic_expenditure theo income
        self.shock_perm_gap = shock_perm_gap
        self.shock_temp_gap = shock_temp_gap
        self.shock_perm_size = shock_perm_size
        self.shock_temp_size = shock_temp_size
        self.shock_prob = shock_prob

        # ── Khởi tạo dữ liệu thu nhập ─────────────────
        interp_inc = intrp.interp1d(PRCTG, INC_PERC_2019, fill_value="extrapolate")
        raw_incomes = interp_inc(self.rng.random(size=n_agents))
        self.incomes = raw_incomes / 12_000.0      # monthly income (scaled)

        # Thresholds cho income decile
        self.inc_thresholds = np.percentile(self.incomes, [10, 20, 30, 40, 50, 60, 70, 80, 90, 99])

        # ── Khởi tạo dữ liệu tiết kiệm ────────────────
        interp_sav = intrp.interp1d(PRCTG, SAVING_VAL, fill_value="extrapolate")
        raw_savings = interp_sav(self.rng.random(size=n_agents))
        self.savings = raw_savings / 1_000.0        # initial savings (scaled)

        # Thresholds cho savings decile
        self.sav_thresholds = np.percentile(self.savings, [10, 20, 30, 40, 50, 60, 70, 80, 90, 99])

        # ── Chính sách tiêu dùng (consumption policy) ─
        # Sử dụng quy tắc đơn giản: tiêu dùng = (1 - beta) * (r * assets + income)
        # Tham khảo từ mô hình IFP chuẩn với CRRA utility
        self._build_consumption_policy(grid_max, grid_size)

        # Trạng thái sẽ được khởi tạo trong reset()
        self.assets = None
        self.current_income = None
        self.bankrupt = None
        self.t = 0
        self.history = None

    def _basic_expenditure(self, income):
        """Vectorized basic expenditure dựa trên income bracket."""
        condlist = [
            income <= self.inc_thresholds[0],
            (income > self.inc_thresholds[0]) & (income <= self.inc_thresholds[1]),
            (income > self.inc_thresholds[1]) & (income <= self.inc_thresholds[2]),
            (income > self.inc_thresholds[2]) & (income <= self.inc_thresholds[3]),
            (income > self.inc_thresholds[3]) & (income <= self.inc_thresholds[4]),
            (income > self.inc_thresholds[4]) & (income <= self.inc_thresholds[5]),
            (income > self.inc_thresholds[5]) & (income <= self.inc_thresholds[6]),
            (income > self.inc_thresholds[6]) & (income <= self.inc_thresholds[7]),
            (income > self.inc_thresholds[7]) & (income <= self.inc_thresholds[8]),
            income > self.inc_thresholds[8],
        ]
        # Annual basic needs from BLS / 12 months, scaled by 1000
        choices = [
            25856.0 / 12.0,
            31499.0 / 12.0,
            37131.0 / 12.0,
            43822.0 / 12.0,
            49367.0 / 12.0,
            56720.0 / 12.0,
            66435.0 / 12.0,
            75945.0 / 12.0,
            96913.0 / 12.0,
            145967.0 / 12.0,
        ]
        return np.select(condlist, choices, default=5000.0 / 12.0) / 1000.0

    def _income_state(self, income):
        """Vectorized: trả về income state (0..10)."""
        return np.digitize(income, self.inc_thresholds, right=False)

    def _build_consumption_policy(self, grid_max, grid_size):
        """
        Xây dựng lưới chính sách tiêu dùng.

        Sử dụng quy tắc: c = alpha * (r * a + y)
        với alpha = 1 - beta, beta ~ 0.90
        (xấp xỉ từ lời giải EGM trong notebook gốc).
        """
        self.s_grid = np.linspace(0, grid_max, grid_size)
        # Consumption policy: c = (1 - beta) * (r * s + y)
        # với beta = 0.90
        self.beta = 0.90
        # Chúng ta sẽ tính c trực tiếp trong step() dựa trên assets và income
        # Lưu s_grid để dùng cho interpolation
        self.grid_max = grid_max

    def reset(self, return_history=False):
        """
        Khởi tạo lại tất cả agents.

        Parameters
        ----------
        return_history : bool
            Nếu True, trả về toàn bộ lịch sử (assets theo thời gian).

        Returns
        -------
        dict hoặc None
        """
        self.assets = self.savings.copy().astype(float)
        self.current_income = self.incomes.copy().astype(float)
        self.bankrupt = np.zeros(self.n_agents, dtype=bool)
        self.t = 0

        if return_history:
            self.history = [self.assets.copy()]
        else:
            self.history = None

        return {
            "assets": self.assets.copy(),
            "income": self.current_income.copy(),
            "bankrupt": self.bankrupt.copy(),
        }

    def step(self, dt=1):
        """
        Cập nhật trạng thái cho tất cả agents trong dt tháng (vectorized).

        Returns
        -------
        dict với các keys: assets, income, bankrupt
        """
        t_now = self.t

        # ── 1. Income shocks (chỉ tác động lên agent chưa bankrupt) ────
        active = ~self.bankrupt

        # Permanent shock: xảy ra theo chu kỳ với xác suất shock_prob
        perm_shock_mask = (
            (t_now > 0)
            & ((t_now % self.shock_perm_gap) == 0)
            & active
            & (self.rng.random(self.n_agents) < self.shock_prob)
        )
        perm_sign = self.rng.choice([-1, 1], size=self.n_agents) * 1.0
        if perm_shock_mask.any():
            self.current_income[perm_shock_mask] *= (
                1.0 + perm_sign[perm_shock_mask] * self.shock_perm_size
            )

        # Temporary shock: xảy ra theo chu kỳ
        temp_shock_mask = (
            (t_now > 0)
            & ((t_now % self.shock_temp_gap) == 0)
            & active
            & (self.rng.random(self.n_agents) < self.shock_prob)
        )
        temp_sign = self.rng.choice([-1, 1], size=self.n_agents) * 1.0
        if temp_shock_mask.any():
            self.current_income[temp_shock_mask] *= (
                1.0 + temp_sign[temp_shock_mask] * self.shock_temp_size
            )

        # ── 2. Income state ──────────────────────────────────────
        z = self._income_state(self.current_income)

        # ── 3. Consumption ───────────────────────────────────────
        # c = min((1 - beta) * (r * a + y), a)  -- không vay quá khả năng
        cash_on_hand = self.r * self.assets + self.current_income
        c_opt = (1.0 - self.beta) * cash_on_hand

        # Giới hạn: tiêu dùng <= tài sản hiện tại (không vay nặng lãi)
        c_opt = np.minimum(c_opt, self.assets)

        # Tiêu dùng tối thiểu
        if self.c_min is not None:
            c_min_arr = np.full(self.n_agents, self.c_min)
        else:
            c_min_arr = self._basic_expenditure(self.current_income)

        c = np.maximum(c_opt, c_min_arr)

        # ── 4. Asset update ──────────────────────────────────────
        s = self.assets - c
        self.assets = self.r * s + self.current_income

        # ── 5. Bankruptcy check ──────────────────────────────────
        self.bankrupt = self.bankrupt | (self.assets <= 0)
        self.assets[self.bankrupt] = 0.0
        self.current_income[self.bankrupt] = 0.0

        self.t += dt

        if self.history is not None:
            self.history.append(self.assets.copy())

        return {
            "assets": self.assets.copy(),
            "income": self.current_income.copy(),
            "bankrupt": self.bankrupt.copy(),
        }

    def run_episode(self, T=60, return_history=True):
        """
        Chạy mô phỏng T tháng.

        Parameters
        ----------
        T : int
            Số tháng mô phỏng
        return_history : bool
            Nếu True, trả về toàn bộ lịch sử

        Returns
        -------
        history : np.ndarray shape (T+1, n_agents)
            Lịch sử tài sản của tất cả agents
        """
        self.reset(return_history=True)
        for _ in range(T):
            self.step()
        return np.array(self.history)

    @property
    def n_active(self):
        return int((~self.bankrupt).sum())

    @property
    def bankruptcy_rate(self):
        return float(self.bankrupt.mean())
