"""
D-E-A/EDOT/λ门槛 三模型联合蒙特卡洛仿真
服务论文一（E/(D+A)稳定性）、论文二（边际成本递减）、论文三（λ动态）

核心设定：
- N个任务，每个有独立的注意力边际产出 V_i ~ LogNormal(mu, sigma)
- Σα_i = 1 硬约束
- λ 由最优配置的KKT条件内生决定
- EDOT周期：方法论升级降低评估标准建立的固定成本
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ============================================================
# 核心函数
# ============================================================

def compute_lambda(V, alpha_budget=1.0):
    """
    给定V_i分布和总注意力预算，求解λ。
    最优条件：α_i > 0 iff V_i >= λ
    即：按V降序排列，依次分配注意力直到预算耗尽。
    λ = 最后一个获得正注意力的任务的V值。
    """
    V_sorted = np.sort(V)[::-1]
    cumsum = np.cumsum(V_sorted)
    # 内点解: λ使前k个任务的V_i/λ之和=1
    # 简化: λ位于V_sorted[k]和V_sorted[k+1]之间
    for k in range(1, len(V_sorted)):
        lambda_candidate = cumsum[k-1] / alpha_budget
        if lambda_candidate <= V_sorted[k-1] and (k == len(V_sorted) or lambda_candidate >= V_sorted[k]):
            # 检查是否满足角点条件
            n_active = np.sum(V >= lambda_candidate)
            if n_active == k or abs(n_active - k) <= 1:
                return lambda_candidate
    # 回退: 取中位数V作为λ
    return np.median(V)

def compute_attention_allocation(V, lam):
    """给定V和λ，计算最优α分配（线性边际回报）"""
    alpha = np.zeros_like(V)
    active = V >= lam
    alpha[active] = V[active] / np.sum(V[active])
    return alpha

def compute_n_active(V, lam):
    """计算获得正注意力的任务数量"""
    return np.sum(V >= lam)

# ============================================================
# 图1: λ随N_out扩张而上升
# ============================================================

def fig1_lambda_dynamics():
    """AI扩大可外包任务池→λ内生上升→更多低V任务被注意力放弃"""
    N_base = 50
    N_expansions = np.arange(N_base, 500, 10)

    # 基准任务池: V_i~LogNormal(0, 0.8), 代表传统经济中的任务分布
    V_base = np.random.lognormal(0, 0.8, N_base)

    lambdas = []
    n_abandoned = []
    n_active_list = []

    for N in N_expansions:
        # AI新增任务: V_i~LogNormal(-0.5, 1.0), 均值较低（更多低V任务涌入）
        n_new = N - N_base
        V_new = np.random.lognormal(-0.5, 1.0, n_new)
        V_all = np.concatenate([V_base, V_new])

        lam = compute_lambda(V_all)
        lambdas.append(lam)
        n_active_list.append(compute_n_active(V_all, lam))
        # 被放弃的任务 = V_i < λ的任务
        n_abandoned.append(np.sum(V_all < lam))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 子图1: λ随N上升
    axes[0].plot(N_expansions, lambdas, 'b-', linewidth=1.5)
    axes[0].axhline(y=np.median(V_base), color='gray', linestyle='--', alpha=0.5, label='基准λ (N=50)')
    axes[0].set_xlabel('可外包任务池大小 N_out')
    axes[0].set_ylabel('注意力影子价格 λ')
    axes[0].set_title('λ随N_out扩张而上升')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 子图2: 获得注意力的任务比例
    active_ratio = np.array(n_active_list) / N_expansions
    axes[1].plot(N_expansions, active_ratio, 'r-', linewidth=1.5)
    axes[1].set_xlabel('可外包任务池大小 N_out')
    axes[1].set_ylabel('获正注意力的任务比例')
    axes[1].set_title('注意力覆盖率随N_out下降')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    # 子图3: 注意力放弃的任务数
    axes[2].plot(N_expansions, n_abandoned, 'orange', linewidth=1.5)
    axes[2].set_xlabel('可外包任务池大小 N_out')
    axes[2].set_ylabel('被注意力放弃的任务数')
    axes[2].set_title('注意力放弃随N_out加速增长')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig1_lambda_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图1完成: λ动态")


# ============================================================
# 图2: E/(D+A)比率的随机稳定性
# ============================================================

def fig2_ratio_stability():
    """模拟1000次随机拆分36实例→计算E/(D+A)→验证跨样本稳定性"""
    n_simulations = 1000
    n_instances = 36

    # D-E-A缺陷生成参数（基于P1基线校准）
    # D ~ Poisson(λ_d), E ~ Poisson(λ_e), A ~ Poisson(λ_a)
    # P1: D=20, E=27, A=5 → D_per_instance≈0.56, E≈0.75, A≈0.14
    lambda_d = 0.56
    lambda_e = 0.75
    lambda_a = 0.14

    ratios = []
    d_counts = []
    e_counts = []
    a_counts = []

    for _ in range(n_simulations):
        d = np.random.poisson(lambda_d, n_instances)
        e = np.random.poisson(lambda_e, n_instances)
        a = np.random.poisson(lambda_a, n_instances)

        D_total = np.sum(d > 0)
        E_total = np.sum(e > 0)
        A_total = np.sum(a > 0)

        d_counts.append(D_total)
        e_counts.append(E_total)
        a_counts.append(A_total)

        ratio = E_total / (D_total + A_total) if (D_total + A_total) > 0 else float('inf')
        ratios.append(ratio)

    ratios = np.array(ratios)
    ratios = ratios[np.isfinite(ratios)]

    # 模拟两次独立36样本的E/(D+A)差异
    diff_null = []
    for _ in range(10000):
        idx1 = np.random.choice(len(ratios), 36, replace=False)
        idx2 = np.random.choice(len(ratios), 36, replace=False)
        diff = np.mean(ratios[idx1]) - np.mean(ratios[idx2])
        diff_null.append(diff)

    diff_null = np.array(diff_null)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 子图1: E/(D+A)分布
    axes[0].hist(ratios, bins=50, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    axes[0].axvline(x=1.08, color='red', linestyle='--', linewidth=1.5, label='旧36题P1: 1.08')
    axes[0].axvline(x=1.04, color='green', linestyle='--', linewidth=1.5, label='新36题P1: 1.04')
    axes[0].axvline(x=np.mean(ratios), color='black', linestyle='-', linewidth=1, label=f'仿真均值: {np.mean(ratios):.2f}')
    axes[0].set_xlabel('E/(D+A)')
    axes[0].set_ylabel('密度')
    axes[0].set_title('E/(D+A)的随机分布 (1000次仿真)')
    axes[0].legend(fontsize=8)

    # 子图2: Bootstrap零分布
    axes[1].hist(diff_null, bins=50, color='gray', edgecolor='white', alpha=0.7, density=True)
    axes[1].axvline(x=0.04, color='red', linestyle='--', linewidth=1.5, label='观测差异: 0.04 (旧-新)')
    axes[1].axvline(x=np.percentile(diff_null, 2.5), color='blue', linestyle=':', alpha=0.7, label='95% CI下界')
    axes[1].axvline(x=np.percentile(diff_null, 97.5), color='blue', linestyle=':', alpha=0.7, label='95% CI上界')
    p_value = np.mean(np.abs(diff_null) >= 0.04)
    axes[1].set_xlabel('E/(D+A)差异 (两组36样本)')
    axes[1].set_ylabel('密度')
    axes[1].set_title(f'E/(D+A)差异的Bootstrap零分布\np={p_value:.2f} (H0: 两组无差异)')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('fig2_ratio_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图2完成: E/(D+A)稳定性, bootstrap p={p_value:.2f}")


# ============================================================
# 图3: EDOT周期的边际成本递减
# ============================================================

def fig3_edot_marginal_cost():
    """模拟EDOT周期：每次诊断→方法论文本写入→每份报告的边际注意力成本递减"""
    n_reports_per_cycle = 13
    n_cycles = 5

    # 第一周期（无EDOT）：每份报告的注意力成本 ~ Exp(1)
    base_cost_per_report = 1.0
    learning_rate = 0.35  # EDOT每次降低的边际成本比例

    cycles = []
    cumulative_costs = []
    marginal_costs = []

    total_cost = 0
    for cycle in range(n_cycles):
        # EDOT诊断的固定成本（仅在周期开始时支付）
        diagnosis_cost = 0.8 if cycle > 0 else 0  # 第一周期无诊断先例

        # 当前周期的边际报告成本（随EDOT递减）
        current_marginal_cost = base_cost_per_report * (1 - learning_rate) ** cycle

        cycle_costs = []
        for report in range(n_reports_per_cycle):
            report_cost = current_marginal_cost + np.random.exponential(0.1)
            cycle_costs.append(report_cost)
            total_cost += report_cost

        # 加诊断成本
        if cycle > 0:
            total_cost += diagnosis_cost

        cycles.append(cycle + 1)
        cumulative_costs.append(total_cost)
        marginal_costs.append(np.mean(cycle_costs))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 子图1: 累积成本曲线
    axes[0].plot(cycles, cumulative_costs, 'b-o', linewidth=1.5, markersize=6)
    axes[0].set_xlabel('EDOT周期')
    axes[0].set_ylabel('累积注意力成本')
    axes[0].set_title('EDOT周期的累积成本曲线\n(凹形=边际成本递减)')
    axes[0].grid(True, alpha=0.3)

    # 子图2: 每份报告的边际成本
    axes[1].bar(cycles, marginal_costs, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('EDOT周期')
    axes[1].set_ylabel('每份报告的平均注意力成本')
    axes[1].set_title('边际注意力成本逐周期递减')
    axes[1].grid(True, alpha=0.3, axis='y')

    # 子图3: 方法论知识积累
    knowledge = [1 - (1 - learning_rate) ** c for c in range(n_cycles)]
    axes[2].plot(cycles, knowledge, 'g-o', linewidth=1.5, markersize=6)
    axes[2].fill_between(cycles, 0, knowledge, alpha=0.2, color='green')
    axes[2].set_xlabel('EDOT周期')
    axes[2].set_ylabel('方法论知识积累度')
    axes[2].set_title('方法论知识的复利积累')
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig3_edot_marginal_cost.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图3完成: EDOT边际成本递减")


# ============================================================
# 图4: E/(D+A)跨阶段攀升 (模拟P1→P2→P3)
# ============================================================

def fig4_cross_stage_trajectory():
    """模拟三阶段优化过程：D和A先被修复→E成为相对瓶颈→E/(D+A)攀升"""
    stages = ['P1\n(Baseline)', 'P2\n(Self-reflection)', 'P3\n(External Diag)']

    # D-E-A缺陷数 (校准至实验数据)
    d_flags = [20, 9, 7]
    e_flags = [27, 34, 27]
    a_flags = [5, 3, 2]

    x = np.arange(len(stages))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 子图1: 分组柱状图
    axes[0].bar(x - width, d_flags, width, label='D旗', color='#4472C4', edgecolor='white')
    axes[0].bar(x, e_flags, width, label='E旗', color='#ED7D31', edgecolor='white')
    axes[0].bar(x + width, a_flags, width, label='A旗', color='#A5A5A5', edgecolor='white')

    # 标注变化率
    axes[0].annotate('-55%', xy=(0.5, 24), fontsize=9, color='#4472C4', fontweight='bold',
                ha='center', va='bottom')
    axes[0].annotate('+26%', xy=(0.65, 37), fontsize=9, color='#ED7D31', fontweight='bold',
                ha='center', va='bottom')

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages)
    axes[0].set_ylabel('缺陷旗数量')
    axes[0].set_title('D-E-A缺陷旗的三阶段变化')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2, axis='y')

    # 子图2: E/(D+A)比率
    ratios = [e_flags[i] / (d_flags[i] + a_flags[i]) for i in range(3)]
    axes[1].plot(stages, ratios, 'o-', color='#ED7D31', linewidth=2, markersize=10, markerfacecolor='white')
    axes[1].fill_between(range(3), 0, ratios, alpha=0.15, color='#ED7D31')

    for i, (s, r) in enumerate(zip(stages, ratios)):
        axes[1].annotate(f'{r:.2f}', xy=(i, r), xytext=(i, r + 0.15),
                    ha='center', fontsize=11, fontweight='bold', color='#ED7D31')

    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='E=D+A (基准线)')
    axes[1].set_ylabel('E/(D+A)')
    axes[1].set_title('E/(D+A)比率的三阶段攀升\n(E瓶颈的持续暴露)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('fig4_cross_stage_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图4完成: E/(D+A)跨阶段攀升")


# ============================================================
# 图5: 批量诊断 vs 实时诊断的方差对比
# ============================================================

def fig5_batch_vs_realtime():
    """模拟批量诊断(低方差) vs 实时诊断(高方差)的效应分布"""
    np.random.seed(123)
    n_sim = 1000

    # 批量诊断: 效应量 ~ N(+5, 1) — 基于P3的+5pp
    batch_effects = np.random.normal(5, 1.5, n_sim)

    # 实时诊断: 效应量 ~ N(0, 4) — 基于转折点实验的净效应0pp但高方差
    realtime_effects = np.random.normal(0, 4, n_sim)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.hist(batch_effects, bins=40, alpha=0.6, color='#4472C4', label='批量诊断\n(跨实例视野)', density=True)
    ax.hist(realtime_effects, bins=40, alpha=0.6, color='#ED7D31', label='实时诊断\n(单实例视野)', density=True)

    ax.axvline(x=5, color='#4472C4', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color='#ED7D31', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('诊断效应量 (PASS率变化, pp)')
    ax.set_ylabel('密度')
    ax.set_title('批量诊断 vs 实时诊断的效应量分布\n(基于SWE-bench P3 +5pp & 转折点实验 0pp校准)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # 标注统计量
    textstr = f'批量: μ={np.mean(batch_effects):.1f}pp, σ={np.std(batch_effects):.1f}pp\n实时: μ={np.mean(realtime_effects):.1f}pp, σ={np.std(realtime_effects):.1f}pp'
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('fig5_batch_vs_realtime.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图5完成: 批量 vs 实时诊断")


# ============================================================
# 综合统计输出
# ============================================================

def print_summary_stats():
    """输出关键统计量，供论文引用"""
    print("\n" + "="*60)
    print("蒙特卡洛仿真 — 关键统计量")
    print("="*60)

    # λ动态
    V_base = np.random.lognormal(0, 0.8, 50)
    V_large = np.concatenate([V_base, np.random.lognormal(-0.5, 1.0, 450)])
    lam_small = compute_lambda(V_base)
    lam_large = compute_lambda(V_large)
    print(f"\nλ (N=50): {lam_small:.3f}")
    print(f"λ (N=500): {lam_large:.3f}")
    print(f"λ上升幅度: {(lam_large/lam_small - 1)*100:.1f}%")

    # 注意力放弃
    n_abandoned_small = np.sum(V_base < lam_small)
    n_abandoned_large = np.sum(V_large < lam_large)
    print(f"\n注意力放弃 (N=50): {n_abandoned_small}/{len(V_base)} ({n_abandoned_small/len(V_base)*100:.1f}%)")
    print(f"注意力放弃 (N=500): {n_abandoned_large}/{len(V_large)} ({n_abandoned_large/len(V_large)*100:.1f}%)")

    # EDOT边际成本递减
    print(f"\nEDOT边际成本递减:")
    for cycle in range(5):
        cost = 1.0 * (0.65) ** cycle
        print(f"  周期{cycle+1}: 边际成本 = {cost:.3f} (相对第1周期: {cost/1.0*100:.1f}%)")

    print(f"\n仿真完成。5张图已保存。")


if __name__ == '__main__':
    print("D-E-A/EDOT/λ 蒙特卡洛仿真")
    print("="*60)

    fig1_lambda_dynamics()
    fig2_ratio_stability()
    fig3_edot_marginal_cost()
    fig4_cross_stage_trajectory()
    fig5_batch_vs_realtime()
    print_summary_stats()
