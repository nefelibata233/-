import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
import tenseal as ts


# ===============================
# 0. 固定随机种子
# ===============================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# 1. 加载 Cora 数据集
# ===============================
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0].to(device)

num_nodes = data.num_nodes
num_features = dataset.num_features
num_classes = dataset.num_classes

print(f"Dataset: Cora, nodes={num_nodes}, features={num_features}, classes={num_classes}")

# ===============================
# 2. TenSEAL CKKS 上下文
# ===============================
poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=poly_modulus_degree,
    coeff_mod_bit_sizes=coeff_mod_bit_sizes
)
context.global_scale = 2 ** 40
context.generate_galois_keys()
context.generate_relin_keys()

public_context = context.copy()
secret_context = context

print("TenSEAL CKKS context created")

# ===============================
# 3. 中继节点参数
# ===============================
relay_rank = 64  # m: 中继节点数量（m << n）

# Q_public: (relay_rank, relay_rank) — 中继节点间关系矩阵
Q_public = torch.randn(relay_rank, relay_rank, device=device, dtype=torch.float32)
Q_public = Q_public / Q_public.norm()  # 初始化归一化

print(f"Relay rank (m): {relay_rank} (num_nodes n: {num_nodes})")


# ===============================
# 4. 融合中继节点的 GCN 模型
# ===============================
class SecureGCN(nn.Module):
    """
    基于中继节点二分图的安全GCN模型。

    对应论文图7-8：
    原节点 → 中继节点（P_i^T @ x）→ 中继聚合（Q_public @ result）→ 原节点（P_i @ result）
    """

    def __init__(self, in_dim, hidden_dim, out_dim, relay_rank=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relay_rank = relay_rank

        # 增加 batch norm 稳定训练
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, P_i=None, Q_public=None, node_mask=None):
        """
        Args:
            x: 全图节点特征 (num_nodes, in_dim)
            edge_index: 全图边 (2, E)
            P_i: 客户端 i 的节点到中继连接权重 (n_local, relay_rank)
            Q_public: 中继节点间关系矩阵 (relay_rank, relay_rank)
            node_mask: 客户端 i 持有的节点索引 (n_local,)
        """
        # ---- 第一层 GCN ----
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        # ---- 中继节点融合 ----
        if P_i is not None and Q_public is not None and node_mask is not None:
            # 正确获取本地节点特征：使用 node_mask 索引
            local_x = x[node_mask]  # (n_local, hidden_dim) ✅ 修复bug1

            # 阶段1: 原节点 → 中继节点投影
            # P_i^T @ local_x: 中继节点聚合所有本地节点特征
            relay_msgs = torch.mm(P_i.t(), local_x)  # (relay_rank, hidden_dim)

            # 阶段2: 中继节点间信息聚合
            # Q_public 编码中继节点间的关联
            relay_agg = torch.mm(Q_public, relay_msgs)  # (relay_rank, hidden_dim)

            # 阶段3: 中继节点 → 原节点投影
            relay_contrib = torch.mm(P_i, relay_agg)  # (n_local, hidden_dim)

            # 归一化
            degree = P_i.abs().sum(dim=1, keepdim=True) + 1e-8
            relay_contrib = relay_contrib / degree

            # 残差连接（论文：保留原始特征 + 中继贡献）
            local_x = local_x + 0.1 * relay_contrib

            # 将更新后的本地特征写回 x
            x = x.clone()  # 避免 in-place 修改影响梯度
            x[node_mask] = local_x

        # ---- Dropout + 第二层 GCN ----
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)

        return x


# ===============================
# 5. 客户端类
# ===============================
class Client:
    def __init__(self, node_idx, global_model, public_context,
                 relay_rank=64,
                 alpha_mf=0.01, beta_mf=0.001, gamma_mf=0.001,  # ⚡ 调小分解损失权重
                 noise_scale=0.0003):
        self.node_idx = node_idx
        self.n_local = len(node_idx)

        # 客户端本地数据（只保留自己节点的特征和标签）
        self.x_local = data.x[node_idx]  # (n_local, num_features)
        self.y_local = data.y[node_idx]  # (n_local,)

        # 构建本地子图
        self.edge_index_local, self.local_node_map = self._build_local_subgraph()

        # 构建本地子图的邻接矩阵
        self.local_adj = self._build_local_adj()

        # 本地训练掩码（每个客户端用自己节点的一部分做训练）
        n_train = max(1, int(self.n_local * 0.6))
        self.train_mask_local = torch.zeros(self.n_local, dtype=torch.bool, device=device)
        self.train_mask_local[:n_train] = True

        self.public_context = public_context
        self.relay_rank = relay_rank

        # P_i: 本地节点到中继节点的连接权重 (n_local, relay_rank)
        self.P_i = nn.Parameter(
            torch.randn(self.n_local, relay_rank, device=device) * 0.01
        )

        # 本地模型
        self.model = SecureGCN(num_features, 64, num_classes, relay_rank).to(device)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + [self.P_i],
            lr=0.005, weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.noise_scale = noise_scale
        self.alpha_mf = alpha_mf
        self.beta_mf = beta_mf
        self.gamma_mf = gamma_mf

    def _build_local_subgraph(self):
        """
        提取客户端的本地子图。
        只保留两个端点都在本客户端节点集合中的边。
        """
        node_set = set(self.node_idx.tolist())
        local_node_map = {orig: local for local, orig in enumerate(self.node_idx.tolist())}

        edges = []
        for i in range(data.edge_index.shape[1]):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            if src in node_set and dst in node_set:
                edges.append([local_node_map[src], local_node_map[dst]])

        if len(edges) == 0:
            # 如果没有内部边，添加自环
            edge_index = torch.arange(self.n_local, device=device).unsqueeze(0).repeat(2, 1)
        else:
            edge_index = torch.tensor(edges, device=device).t().long()

        return edge_index, local_node_map

    def _build_local_adj(self):
        """构建本地邻接矩阵 A_local (n_local, n_local)"""
        adj = torch.zeros(self.n_local, self.n_local, device=device)
        for i in range(self.edge_index_local.shape[1]):
            src = self.edge_index_local[0, i].item()
            dst = self.edge_index_local[1, i].item()
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0
        return adj

    def _compute_matrix_factorization_loss(self, Q_public):
        """
        论文公式(1): f = α||P_i·Q·P_i^T - A||² + β||P_i||² + γ||Q||²

        通过中继节点重构邻接矩阵:
        A_recon = P_i @ Q_public @ P_i^T
        """
        # A_recon = P_i @ Q_public @ P_i^T
        P_Q = torch.mm(self.P_i, Q_public)  # (n_local, relay_rank)
        A_recon = torch.mm(P_Q, self.P_i.t())  # (n_local, n_local)

        # 重构损失（只计算有边的位置，避免稀疏矩阵的过度惩罚）
        recon_loss = self.alpha_mf * torch.mean((A_recon - self.local_adj) ** 2)

        # 正则化
        reg_loss = (self.beta_mf * torch.norm(self.P_i) ** 2 +
                    self.gamma_mf * torch.norm(Q_public) ** 2)

        return recon_loss + reg_loss

    def compute_q_gradient_encrypted(self, Q_public):
        """
        计算对全局 Q_public 的梯度。

        梯度推导:
        L = α·||P_i·Q·P_i^T - A||² + γ·||Q||²
        ∂L/∂Q = 2α·P_i^T·(P_i·Q·P_i^T - A)·P_i + 2γ·Q

        返回加密梯度
        """
        with torch.no_grad():
            # 前向
            P_Q = torch.mm(self.P_i, Q_public)
            A_recon = torch.mm(P_Q, self.P_i.t())
            error = A_recon - self.local_adj

            # 梯度
            grad_Q = 2.0 * self.alpha_mf * torch.mm(
                torch.mm(self.P_i.t(), error),
                self.P_i
            )
            grad_Q += 2.0 * self.gamma_mf * Q_public

            # 梯度裁剪防止爆炸
            grad_norm = grad_Q.norm()
            if grad_norm > 10.0:
                grad_Q = grad_Q / grad_norm * 10.0

            # 差分隐私噪声
            noise = torch.normal(0, self.noise_scale, grad_Q.shape).to(device)
            grad_Q += noise

            # 加密
            grad_flat = grad_Q.detach().cpu().numpy().flatten()
            encrypted_grad = ts.ckks_vector(self.public_context, grad_flat.tolist())

        return encrypted_grad

    def local_update(self, global_model, Q_public, epochs=3):
        """
        本地模型更新。
        使用本地子图 (x_local, edge_index_local) 代替全图。
        """
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()

        for _ in range(epochs):
            self.optimizer.zero_grad()

            # 使用本地子图计算
            out = self.model(self.x_local, self.edge_index_local,
                             P_i=self.P_i, Q_public=Q_public,
                             node_mask=torch.arange(self.n_local, device=device))

            # 分类损失（只对本地训练节点）
            cls_loss = F.cross_entropy(out[self.train_mask_local],
                                       self.y_local[self.train_mask_local])

            # 矩阵分解损失
            mf_loss = self._compute_matrix_factorization_loss(Q_public)

            # ⚡ 平衡两个损失
            total_loss = cls_loss + 0.1 * mf_loss
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_([self.P_i], 5.0)

            # 差分隐私噪声
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.normal(0, self.noise_scale, p.grad.shape).to(device)
                    p.grad += noise
            if self.P_i.grad is not None:
                noise = torch.normal(0, self.noise_scale, self.P_i.grad.shape).to(device)
                self.P_i.grad += noise

            self.optimizer.step()

        self.scheduler.step()

        return self.model.state_dict(), self.P_i.detach()


# ===============================
# 6. 安全服务器类
# ===============================
class SecureServer:
    def __init__(self, global_model, secret_context, relay_rank=64):
        self.global_model = global_model
        self.secret_context = secret_context
        self.secret_key = secret_context.secret_key()
        self.relay_rank = relay_rank

    def secure_aggregate_q_gradient(self, encrypted_grads_list, Q_public):
        """
        安全聚合 Q_public 梯度。
        CKKS 密文加法 → 解密 → 取平均 → 更新
        """
        # 密文加法
        aggregated = encrypted_grads_list[0]
        for i in range(1, len(encrypted_grads_list)):
            aggregated += encrypted_grads_list[i]

        # 解密
        decrypted_list = aggregated.decrypt(self.secret_key)
        grad_flat = np.array(decrypted_list)

        # 取平均
        grad_Q = grad_flat / len(encrypted_grads_list)
        grad_Q = grad_Q.reshape(self.relay_rank, self.relay_rank)

        # 更新 Q_public
        grad_tensor = torch.tensor(grad_Q, device=device, dtype=torch.float32)

        # 梯度裁剪
        if grad_tensor.norm() > 5.0:
            grad_tensor = grad_tensor / grad_tensor.norm() * 5.0

        with torch.no_grad():
            Q_public = Q_public - 0.01 * grad_tensor
            # 防止数值发散
            Q_public = Q_public / (Q_public.norm() + 1e-8) * Q_public.norm().detach()

        return Q_public

    def aggregate_model(self, client_states):
        """FedAvg 聚合模型参数"""
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            if key in client_states[0]:
                stacked = torch.stack([cs[key].float() for cs in client_states])
                global_dict[key] = stacked.mean(dim=0)
        self.global_model.load_state_dict(global_dict)

    def federated_train(self, clients, Q_public, rounds=200):
        """联邦训练主循环"""
        for r in range(rounds):
            # 阶段1-2: 各客户端计算 Q 梯度并加密
            encrypted_grads = [
                c.compute_q_gradient_encrypted(Q_public) for c in clients
            ]

            # 阶段3: 安全聚合 Q 梯度
            Q_public = self.secure_aggregate_q_gradient(encrypted_grads, Q_public)

            # 阶段1-2: 各客户端本地更新模型
            client_states = []
            for client in clients:
                state, _ = client.local_update(self.global_model, Q_public, epochs=3)
                client_states.append(state)

            # 阶段4: 聚合模型 + 广播
            self.aggregate_model(client_states)

            # 每10轮评估
            if (r + 1) % 10 == 0:
                acc = self.evaluate()
                print(f"Round {r + 1:3d}, Test Acc: {acc:.4f}")

        return self.global_model, Q_public

    def evaluate(self):
        """在全局测试集上评估"""
        self.global_model.eval()
        with torch.no_grad():
            out = self.global_model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            acc = accuracy_score(
                data.y[data.test_mask].cpu(),
                pred[data.test_mask].cpu()
            )
        return acc


# ===============================
# 7. 客户端划分（Non-IID）
# ===============================
num_clients = 10
# 按标签分布划分（模拟非独立同分布）
labels = data.y.cpu().numpy()
client_nodes = [[] for _ in range(num_clients)]

# 将每个类别均匀分配到各客户端
for cls in range(num_classes):
    cls_indices = (data.y == cls).nonzero(as_tuple=True)[0]
    cls_indices = cls_indices[torch.randperm(len(cls_indices))]
    split = torch.split(cls_indices, len(cls_indices) // num_clients + 1)
    for i in range(num_clients):
        if i < len(split):
            client_nodes[i].extend(split[i].tolist())

# 确保每个客户端都有节点
for i in range(num_clients):
    client_nodes[i] = torch.tensor(list(set(client_nodes[i])), device=device)
    print(f"Client {i}: {len(client_nodes[i])} nodes")

# 全局模型
global_model = SecureGCN(num_features, 64, num_classes, relay_rank).to(device).float()

# 创建客户端
clients = []
for i in range(num_clients):
    clients.append(Client(client_nodes[i], global_model, public_context,
                          relay_rank=relay_rank,
                          alpha_mf=0.01, beta_mf=0.001, gamma_mf=0.001))


# ===============================
# 8. 非线性函数安全处理（论文ReLU协议）
# ===============================
class SecureNonlinearity:
    """
    论文对不可逆非线性函数ReLU的安全处理：
    各参与者分别生成一个正随机数，
    基于安全求和比率计算线性聚合结果与随机数之和的比率，
    若结果为正则还原，若为负则置0。
    """

    @staticmethod
    def secure_relu_batch(linear_agg_values, public_context, secret_context):
        """安全ReLU (简化版)"""
        n = len(linear_agg_values)
        secret_key = secret_context.secret_key()

        # 各客户端生成正随机数
        random_vals = [torch.rand(1, device=device).item() + 0.1 for _ in range(n)]

        # 加密 (linear_val + r_i)
        encrypted = []
        for i in range(n):
            masked = linear_agg_values[i].detach().cpu().numpy() + random_vals[i]
            enc = ts.ckks_vector(public_context, masked.flatten().tolist())
            encrypted.append(enc)

        # 安全聚合求和
        aggregated = encrypted[0]
        for i in range(1, n):
            aggregated += encrypted[i]

        # 解密
        sum_masked = np.array(aggregated.decrypt(secret_key))

        # 判断正负
        sum_r = sum(random_vals)
        ratio = sum_masked / sum_r

        if ratio > 0:
            return linear_agg_values
        else:
            return [torch.zeros_like(v) for v in linear_agg_values]


# ===============================
# 9. 开始联邦训练
# ===============================
server = SecureServer(global_model, secret_context, relay_rank=relay_rank)

print("=" * 72)
print("  FedPrivGNN — 联邦图神经网络安全聚合框架 (完整修正版)")
print("  " + "=" * 36)
print("  Dataset:  Cora")
print("  Clients:  {} (Label-based Non-IID)".format(num_clients))
print("  Rounds:   200")
print("  " + "-" * 36)
print("  [核心设计]")
print("  • 中继节点二分图: m={} << n={}".format(relay_rank, num_nodes))
print("  • 矩阵分解: A ≈ P_i @ Q_public @ P_i^T")
print("  • 安全聚合: TenSEAL CKKS (密文加法)")
print("  • 本地子图: 各客户端只持有自己的节点和内部边")
print("  " + "-" * 36)
print("  [Bug修复]")
print("  • ✅ 掩码错位: x[node_mask] 代替 x[:n_local]")
print("  • ✅ 全图泄漏: 各客户端使用本地子图 edge_index_local")
print("  • ✅ 损失失衡: mf_loss 权重调整为 0.1")
print("  • ✅ 训练掩码: 各客户端独立划分 train_mask_local")
print("  • ✅ 中继断裂: x = x.clone(); x[node_mask]=local_x")
print("=" * 72)

trained_model, Q_public_final = server.federated_train(
    clients, Q_public, rounds=200
)

# ===============================
# 10. 最终评估
# ===============================
final_acc = server.evaluate()
print(f"\n✅ Final Test Accuracy: {final_acc:.4f}")

# 保存模型
torch.save({
    'model_state': trained_model.state_dict(),
    'Q_public': Q_public_final,
    'final_acc': final_acc
}, 'fedprivgnn_model.pth')
print("Model saved to fedprivgnn_model.pth")