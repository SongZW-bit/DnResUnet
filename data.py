# ----------------------------------------------------------------------
# 步骤 1: 导入所有必需的库
# ----------------------------------------------------------------------
import numpy as np
import torch
import os
from tqdm import tqdm  # 导入tqdm来显示进度条

# 导入 SimPEG 和 discretize 库
from discretize import TensorMesh
from discretize.utils import mkvc, active_from_xyz
from simpeg.utils import model_builder
from simpeg import maps
from simpeg.potential_fields import gravity
from torch.utils.data import TensorDataset

print(f"PyTorch 版本: {torch.__version__}")
print(f"NumPy 版本: {np.__version__}")

# ----------------------------------------------------------------------
# 步骤 2: 定义数据集和模拟的关键参数
# ----------------------------------------------------------------------
# [关键修改 1: 定义多个噪声水平]
NOISE_LEVELS_STD = [0.01, 0.025, 0.05, 0.1, 0.2] # 5个不同的噪声标准差 (mGal)
N_SAMPLES_PER_LEVEL = 20000   # 每个噪声水平的样本数
N_TOTAL_SAMPLES = len(NOISE_LEVELS_STD) * N_SAMPLES_PER_LEVEL # 样本总数 (N) = 250,000

C_CHANNELS = 1      # 通道数 (C)
L_SIGNAL = 512      # 信号长度 (L)

# [关键修改 2: 更改文件名]
DATA_FILENAME = 'gravity_dataset_100k_5SNR_1ch_512L_NORMALIZED.pt' # 新文件名

# ----------------------------------------------------------------------
# 步骤 3: 设置 SimPEG 张量网格 (TensorMesh) 和地形
# (此部分无变化)
# ----------------------------------------------------------------------
dh = 5.0
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)] # Z轴网格
mesh = TensorMesh([hx, hy, hz], "CCN")

# 定义地形
[x_topo, y_topo] = np.meshgrid(np.linspace(-200, 200, 41), np.linspace(-200, 200, 41))
z_topo = -15 * np.exp(-(x_topo**2 + y_topo**2) / 80**2)
topo_xyz = np.c_[mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)]

# 获取地表以下的活动单元
ind_active = active_from_xyz(mesh, topo_xyz)
nC_active = int(ind_active.sum()) # 活动单元的总数

# ----------------------------------------------------------------------
# 步骤 4: 定义井中重力测量 (Survey)
# (此部分无变化)
# ----------------------------------------------------------------------
x_borehole = 0.0
y_borehole = 0.0

z_min = -20.0
z_max = z_min - (L_SIGNAL - 1) * 1.0  # 假设点距为 1米
z_stations = np.linspace(z_min, z_max, L_SIGNAL)

# 创建接收点位置
n_stations = len(z_stations)
x_locations = np.full(n_stations, x_borehole)
y_locations = np.full(n_stations, y_borehole)
receiver_locations = np.c_[x_locations, y_locations, z_stations]

components = ["gz"] # C=1

receiver_list = gravity.receivers.Point(receiver_locations, components=components)
source_field = gravity.sources.SourceField(receiver_list=[receiver_list])
survey = gravity.survey.Survey(source_field)

# ----------------------------------------------------------------------
# 步骤 5: 定义 SimPEG 模拟 (Simulation)
# (此部分无变化)
# ----------------------------------------------------------------------
model_map = maps.IdentityMap(nP=nC_active)

simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    rhoMap=model_map,
    active_cells=ind_active,
    store_sensitivities="forward_only", # 只做正演，不存敏感度
    engine="choclo",
)

# ----------------------------------------------------------------------
# 步骤 6: 定义随机模型生成函数
# (此部分无变化)
# ----------------------------------------------------------------------
def create_random_model(mesh, ind_active):
    """
    为 SimPEG simulation 创建一个随机的密度模型。
    """
    # 1. 初始化背景模型
    model = np.zeros(nC_active)
    
    # 2. 随机决定要添加多少个异常体 (例如 1 到 3 个)
    num_anomalies = np.random.randint(1, 4)
    
    for _ in range(num_anomalies):
        # 3. 随机决定异常体类型 (0=块体, 1=球体)
        anomaly_type = np.random.randint(0, 2)
        
        # 4. 随机定义属性
        # 随机密度 (g/cm^3), 避免为 0
        rand_density = np.random.uniform(-0.5, 0.5)
        while abs(rand_density) < 0.05:
            rand_density = np.random.uniform(-0.5, 0.5)

        # 随机中心位置 (在网格范围内，且在井眼附近)
        rand_x = np.random.uniform(-100, 100)
        rand_y = np.random.uniform(-100, 100)
        rand_z = np.random.uniform(-500, -30) # 确保在地表以下
        center = [rand_x, rand_y, rand_z]

        if anomaly_type == 0: # 块体
            # 随机尺寸 (米)
            width_x = np.random.uniform(10, 50)
            width_y = np.random.uniform(10, 50)
            height_z = np.random.uniform(10, 40)
            
            # 计算块体边界
            x1, x2 = center[0] - width_x/2, center[0] + width_x/2
            y1, y2 = center[1] - width_y/2, center[1] + width_y/2
            z1, z2 = center[2] - height_z/2, center[2] + height_z/2
            
            # 获取块体索引 (只在活动单元内)
            ind_anomaly = (
                (mesh.gridCC[ind_active, 0] > x1) & (mesh.gridCC[ind_active, 0] < x2) &
                (mesh.gridCC[ind_active, 1] > y1) & (mesh.gridCC[ind_active, 1] < y2) &
                (mesh.gridCC[ind_active, 2] > z1) & (mesh.gridCC[ind_active, 2] < z2)
            )
            
        else: # 球体
            # 随机半径 (米)
            rand_radius = np.random.uniform(10, 40)
            
            # 获取球体索引 (只在活动单元内)
            ind_sphere = model_builder.get_indices_sphere(center, rand_radius, mesh.gridCC)
            ind_anomaly = ind_sphere[ind_active] # 只保留活动单元

        # 5. 将异常体密度赋值给模型
        model[ind_anomaly] = rand_density
        
    return model.astype(np.float32)

# ----------------------------------------------------------------------
# 步骤 7: [关键修改] 主循环：为多个噪声水平生成数据
# ----------------------------------------------------------------------
print(f"开始生成 {N_TOTAL_SAMPLES} 个 (共 {len(NOISE_LEVELS_STD)} 个噪声水平) 数据对...")

# 初始化两个列表来存储所有信号
clean_signals_list = []
noisy_signals_list = []

# 定义一个最小信号幅度阈值（mGal），低于此值的将被拒绝
SIGNAL_AMPLITUDE_THRESHOLD = 0.01 

# 使用 tqdm 创建一个总进度条
pbar = tqdm(total=N_TOTAL_SAMPLES, desc="Overall Progress")

# 外循环：遍历定义的每个噪声水平
for noise_std in NOISE_LEVELS_STD:
    
    print(f"\n正在为噪声水平 {noise_std} mGal 生成 {N_SAMPLES_PER_LEVEL} 个样本...")
    samples_generated_for_this_level = 0
    
    # 内循环：为当前噪声水平生成 N_SAMPLES_PER_LEVEL 个样本
    while samples_generated_for_this_level < N_SAMPLES_PER_LEVEL:
        
        # a. 生成一个独一无二的随机地质模型
        model_vector = create_random_model(mesh, ind_active)
        
        # b. 计算“干净信号” (Y_label)
        dpred_clean = simulation.dpred(model_vector)
        
        # --- 拒绝采样检查 ---
        if np.max(np.abs(dpred_clean)) < SIGNAL_AMPLITUDE_THRESHOLD:
            continue # 信号太弱，丢弃并重新生成
        # --- 检查结束 ---

        # c. 生成“纯噪声” (使用当前外循环的 noise_std)
        noise_vector = (np.random.randn(L_SIGNAL) * noise_std).astype(np.float32)
        
        # d. 计算“带噪信号” (X_input)
        dpred_noisy = dpred_clean + noise_vector
        
        # e. 将 Numpy 数组添加到 *总* 列表中
        clean_signals_list.append(dpred_clean)
        noisy_signals_list.append(dpred_noisy)
        
        # f. 更新计数器
        samples_generated_for_this_level += 1
        pbar.update(1) # 更新总进度条

pbar.close() # 完成后关闭进度条
print("所有噪声水平的数据生成完毕。")

# ----------------------------------------------------------------------
# 步骤 8: 格式化为 (N, C, L) 的 PyTorch 张量并保存
# ----------------------------------------------------------------------
print("正在将数据转换为 PyTorch 张量...")

# 1. 将 Python 列表转换为 Numpy 数组 (总共 250,000 个样本)
X_array_nl = np.array(noisy_signals_list)
Y_array_nl = np.array(clean_signals_list)

# --- [归一化步骤] ---
print("正在对数据进行归一化...")

# 2. 找到 *所有 250,000 个带噪信号* 中的“全局最大绝对值”
global_max = np.max(np.abs(X_array_nl))
print(f"全局最大幅度 (Global Max): {global_max:.4f} mGal")

# 3. 使用全局最大值对 X 和 Y 进行归一化
X_array_nl_norm = X_array_nl / global_max
Y_array_nl_norm = Y_array_nl / global_max

# 4. [关键] 重塑 (Reshape) 归一化后的数组
X_array_ncl = X_array_nl_norm.reshape(N_TOTAL_SAMPLES, C_CHANNELS, L_SIGNAL)
Y_array_ncl = Y_array_nl_norm.reshape(N_TOTAL_SAMPLES, C_CHANNELS, L_SIGNAL)

print(f"最终输入 X (带噪) 的张量形状: {X_array_ncl.shape}")
print(f"最终标签 Y (干净) 的张量形状: {Y_array_ncl.shape}")

# 5. 转换为 PyTorch Float Tensors
X_tensor = torch.from_numpy(X_array_ncl).float()
Y_tensor = torch.from_numpy(Y_array_ncl).float()

# 6. [关键] 保存包含归一化因子 (global_max) 的 .pt 文件
print(f"正在将数据集保存到: {DATA_FILENAME} ...")
torch.save({
    'X_noisy': X_tensor,  # 归一化的带噪信号 (网络输入)
    'Y_clean': Y_tensor,   # 归一化的干净信号 (训练标签)
    'global_max': global_max # 必须保存这个值用于测试！
}, DATA_FILENAME)

print("数据集保存完毕！")

# ----------------------------------------------------------------------
# 步骤 9: (可选) 如何在训练脚本中加载此数据集
# ----------------------------------------------------------------------
print("\n--- 如何在您的训练脚本中加载此数据 ---")
print("print(\"正在加载数据集...\")")
# [修改] 更新了文件名
print(f"data = torch.load('{DATA_FILENAME}', weights_only=False) # 推荐添加 weights_only=False")
print("X_tensor = data['X_noisy']")
print("Y_tensor = data['Y_clean']")
# 注意：我们不需要在这里加载 global_max，但测试脚本需要
print("train_dataset = TensorDataset(X_tensor, Y_tensor)")
train_dataset = TensorDataset(X_tensor, Y_tensor)
# [修改] 更新了总样本数
print(f"print(f\"数据集加载完毕，总样本数: {len(train_dataset)}\")") 
# [修改] 更新了形状
print(f"print(f\"样本形状 (N, C, L): {X_tensor.shape}\")")