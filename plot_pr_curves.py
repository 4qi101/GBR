import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 预定义 11 种对比方法的顺序（若目录中存在则按此顺序绘制）
METHOD_ORDER = [
    "UCCH",
    "VTMUCH",
    "UCMFH",
    "JDSH",
    "DJSRH",
    "CIRH",
    "UCHSTM",
    "IMH",
    "CVH",
    "CMFH",
    "GBR",  # 我们的方法，若存在则一并画出
]

TASKS = ["I2T", "T2I"]
BITS = [16, 32, 64, 128]

LEGEND_NAME_MAP = {
    "CMFH+": "CMFH",
    "VTMUCH": "VTM-UCH",
}

def get_interp_pr(p, r, num_points=101):
    """
    对 P-R 曲线进行插值，使其平滑并在固定的 Recall 点上取值。
    """
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    mask = np.isfinite(p) & np.isfinite(r)
    if not np.any(mask):
        recall_grid = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
        precision_grid = np.zeros_like(recall_grid)
        return precision_grid, recall_grid

    p = p[mask]
    r = r[mask]

    if p.size == 0:
        recall_grid = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
        precision_grid = np.zeros_like(recall_grid)
        return precision_grid, recall_grid

    # 按 recall 排序
    order = np.argsort(r)
    r = r[order]
    p = p[order]

    r = np.clip(r, 0.0, 1.0)

    # 保证 Precision 随着 Recall 单调非递增 (Standard Interpolation)
    for i in range(p.size - 2, -1, -1):
        if p[i] < p[i + 1]:
            p[i] = p[i + 1]

    recall_grid = np.linspace(0.0, 1.0, num_points, dtype=np.float64)

    unique_r, idx = np.unique(r, return_index=True)
    unique_p = p[idx]

    if unique_r.size == 1:
        precision_grid = np.full_like(recall_grid, unique_p[0])
        return precision_grid, recall_grid

    precision_grid = np.interp(recall_grid, unique_r, unique_p,
                               left=unique_p[0], right=unique_p[-1])

    return precision_grid, recall_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PR curves for a single task/bit from multiple methods."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Root directory that contains subdirectories for each method.",
        default="F:\\project\\pythonCode\\Cross-modal retrieval\\Granular-ball\\GBR\\pr",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        help="Dataset subdirectory name under each method, e.g. NUSWIDE, MIRFlickr.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["I2T", "T2I"],
        default="T2I",
        help="Retrieval task direction (I2T or T2I).",
    )
    parser.add_argument(
        "--bit",
        type=int,
        choices=[8, 16, 32, 64, 128],
        default=64,
        help="Hash code length (number of bits).",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=BITS.copy(),
        help="Bit lengths (space-separated) to include when plotting the full grid.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="F:\\project\\pythonCode\\Cross-modal retrieval\\Granular-ball\\GBR\\pr_plot",
        help="Optional directory to save the output figure (default: root_dir).",
    )
    parser.add_argument(
        "--plot_all",
        default=True,
        help="Plot all task/bit combinations (2 tasks x 5 bits) for the selected dataset.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.5,
        help="Lower bound of the Precision axis.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
        help="Upper bound of the Precision axis.",
    )
    parser.add_argument(
        "--ytick",
        type=float,
        default=0.05,
        help="Spacing between Precision ticks (set <=0 to disable manual ticks).",
    )
    return parser.parse_args()


def get_method_style(name: str):
    """
    为每个方法分配固定的线型/颜色/marker，解决颜色和形状冲突。
    重点突显 GBR 方法。
    """
    name = name.upper()

    # 定义基础样式
    base_style = {"linewidth": 1.5, "markersize": 7, "zorder": 5}

    # ---------------------------------------------------------
    # 1. 你的方法 GBR：红色、加粗、星号、置顶
    # ---------------------------------------------------------
    if name == "GBR":
        return {
            "color": "red",
            "marker": "*",
            "linestyle": "-",
            "linewidth": 1.5,  # 线条加粗
            "markersize": 7,  # 标记加大
            "zorder": 11  # 保证画在最上层
        }

    # ---------------------------------------------------------
    # 2. 其他对比方法：手动指定互斥颜色和形状
    # ---------------------------------------------------------
    style_map = {
        "UCCH": {"color": "#ff7f0e", "marker": "s"},  # Orange, Square
        "VTM-UCH": {"color": "#2ca02c", "marker": "^"},  # Green, Triangle Up
        "UCMFH": {"color": "#1f77b4", "marker": "v"},  # Blue (原红色改为蓝色), Triangle Down
        "JDSH": {"color": "#9467bd", "marker": "<"},  # Purple, Triangle Left
        "DJSRH": {"color": "#8c564b", "marker": ">"},  # Brown, Triangle Right
        "CIRH": {"color": "#e377c2", "marker": "p"},  # Pink, Pentagon (原Diamond改为五边形)
        "UCHSTM": {"color": "#7f7f7f", "marker": "H"},  # Gray, Hexagon
        "IMH": {"color": "#bcbd22", "marker": "h"},  # Olive, Hexagon2
        "CVH": {"color": "#17becf", "marker": "X"},  # Cyan, X
        "CMFH": {"color": "black", "marker": "o"},  # Black, Circle
        "UCGBH": {"color": "#aec7e8", "marker": "D"},  # Light Blue, Diamond
    }

    # 获取样式，如果未定义则使用默认灰色
    style = style_map.get(name, {"color": "gray", "marker": ".", "linestyle": "--"})

    merged = base_style.copy()
    merged.update(style)
    return merged


def get_task_label(task: str) -> str:
    task_upper = task.upper()
    if task_upper == "I2T":
        return "I->T"
    if task_upper == "T2I":
        return "T->I"
    return task


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def get_ordered_methods(root_dir: str):
    method_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    ordered = [m for m in METHOD_ORDER if m in method_dirs]
    for m in sorted(method_dirs):
        if m not in ordered:
            ordered.append(m)
    return ordered


def plot_curves_for_task_bit(ax, ordered_methods, root_dir, dataset, task, bit):
    handles = []
    labels = []

    for method in ordered_methods:
        method_dataset_dir = os.path.join(root_dir, method, dataset)
        if not os.path.isdir(method_dataset_dir):
            continue

        filename = f"{method}_{task}_{bit}.npy"
        path = os.path.join(method_dataset_dir, filename)
        if not os.path.exists(path):
            continue

        data = np.load(path, allow_pickle=True).item()
        if not isinstance(data, dict) or "P" not in data or "R" not in data:
            continue

        P = np.asarray(data["P"], dtype=np.float64)
        R = np.asarray(data["R"], dtype=np.float64)
        if P.size == 0 or R.size == 0:
            continue

        P_interp, R_interp = get_interp_pr(P, R)

        legend_name = LEGEND_NAME_MAP.get(method, method)
        style = get_method_style(legend_name)
        line, = ax.plot(
            R_interp,
            P_interp,
            label=legend_name,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            linewidth=style.get("linewidth", 1.8),
            markersize=style.get("markersize", 5),
            markevery=10,
            zorder=style.get("zorder", 5),
        )

        handles.append(line)
        labels.append(legend_name)

    return handles, labels


def format_axis(ax, dataset, task, bit, y_min, y_max, y_tick, show_xlabel=True, show_ylabel=True):
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, y_max)
    if y_tick > 0:
        ax.set_yticks(np.arange(y_min, y_max + 1e-6, y_tick))
    ax.tick_params(axis="y", labelleft=True)

    ax.tick_params(axis="x", labelbottom=show_xlabel)
    if show_xlabel:
        ax.set_xlabel("Recall", fontsize=12)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("Precision", fontsize=12)
    else:
        ax.set_ylabel("")

    task_label = get_task_label(task)
    ax.set_title(f"{dataset} - {bit} bits ({task_label})", fontsize=13, pad=12)


def main():
    args = parse_args()

    root_dir = args.root_dir
    dataset = args.dataset
    task = args.task.upper()
    bit = args.bit
    save_dir = args.save_dir if args.save_dir is not None else root_dir
    bits = args.bits
    y_min = args.ymin
    y_max = args.ymax
    y_tick = args.ytick

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    ordered_methods = get_ordered_methods(root_dir)
    if not ordered_methods:
        raise RuntimeError(f"No method subdirectories found in {root_dir}")

    ensure_dir(save_dir)

    if args.plot_all:
        num_cols = max(1, len(bits))
        fig_width = 4 * num_cols
        fig_height = 4 * len(TASKS)
        fig, axes = plt.subplots(len(TASKS), num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        if axes.shape != (len(TASKS), num_cols):
            axes = axes.reshape(len(TASKS), num_cols)

        for row, task_name in enumerate(TASKS):
            for col, bit_val in enumerate(bits):
                ax = axes[row, col]
                handles, labels = plot_curves_for_task_bit(
                    ax,
                    ordered_methods,
                    root_dir,
                    dataset,
                    task_name,
                    bit_val,
                )
                format_axis(
                    ax,
                    dataset,
                    task_name,
                    bit_val,
                    y_min,
                    y_max,
                    y_tick,
                    show_xlabel=True,
                    show_ylabel=True,
                )

        # 构造统一的顶部图例，按照 ordered_methods 顺序
        legend_handles = []
        legend_labels = []
        for method in ordered_methods:
            label = LEGEND_NAME_MAP.get(method, method)
            style = get_method_style(label)
            handle = Line2D(
                [0],
                [0],
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker", "o"),
                linewidth=style.get("linewidth", 1.8),
                markersize=style.get("markersize", 5),
            )
            legend_handles.append(handle)
            legend_labels.append(label)

        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncol=len(legend_handles),
                frameon=False,
                fontsize=11,
                columnspacing=1.0,
                handletextpad=0.4,
            )

        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_name = f"PR_{dataset}_ALL.pdf"
        out_path = os.path.join(save_dir, out_name)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved PR grid figure to {out_path}")
        return

    # 单图模式
    fig, ax = plt.subplots(figsize=(8, 6))
    handles, labels = plot_curves_for_task_bit(
        ax,
        ordered_methods,
        root_dir,
        dataset,
        task,
        bit,
    )
    format_axis(ax, dataset, task, bit, y_min, y_max, y_tick)

    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            frameon=False,
            fontsize=10,
            columnspacing=1.2,
            handletextpad=0.5,
        )

    fig.tight_layout()
    out_name = f"PR_{dataset}_{task}_{bit}bits.pdf"
    out_path = os.path.join(save_dir, out_name)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved PR figure to {out_path}")


if __name__ == "__main__":
    main()