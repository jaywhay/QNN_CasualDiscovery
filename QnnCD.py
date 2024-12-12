import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import random
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms import VQE
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from pygam import GAM, s
from CDdata import TubingenDatasetHighDim
from scipy.stats import ttest_rel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 파라미터 설정
hidden_dim = 32
output_dim = 2
num_qubits = 5

# ANM 블록 정의
class AdditiveNoiseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_splines=50, ):
        super().__init__()
        self.input_dim = input_dim
        self.n_splines = n_splines
        
        if input_dim == 1:
            self.gam_model = GAM(s(0, n_splines=self.n_splines))
            self.use_gam = True
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.use_gam = False

    def forward(self, x):
        if self.use_gam:
            raise RuntimeError("GAM은 forward 사용 안됨")
        else:
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def fit_gam(self, x, y):
        if self.use_gam:
            x_np = x.numpy().reshape(-1, 1)  # PyTorch 텐서를 numpy로 변환
            y_np = y.numpy()  # 동일하게 변환
            self.gam_model.fit(x_np, y_np)
        else:
            raise RuntimeError("GAM은 input_dim == 1일 때만 사용할 수 있습니다.")
    
    def predict_gam(self, x):
        if self.use_gam:
            x_np = x.numpy().reshape(-1, 1)
            return torch.tensor(self.gam_model.predict(x_np), dtype=torch.float32)
        else:
            raise RuntimeError("GAM은 input_dim == 1일 때만 사용할 수 있습니다.")

# QAOA 사용 양자회로 설계
class QuantumCircuitQAOA:
    def __init__(self):
        self.cached_result = None

    def determine_num_qubits(self, data_sample):
        """
        데이터 샘플 크기에 따라 필요한 최소 큐비트 수를 계산.
        """
        data_dim = len(data_sample)
        return max(1, int(np.ceil(np.log2(data_dim))))  # 최소 1개의 큐비트 필요

    def create_data_based_hamiltonian(self, data_sample):
        """
        데이터 기반 해밀토니안 생성.
        """
        num_qubits = self.determine_num_qubits(data_sample)
        weights = data_sample.numpy()

        # Z 연산자와 ZZ 상호작용 생성
        hamiltonian = SparsePauliOp.from_list([
            ("Z" * num_qubits, weights[0]),  # 단일 Z 항
            ("ZZ" + "I" * (num_qubits - 2), weights[1])  # Z-Z 상호작용 항
        ])
        return hamiltonian

    def create_manual_qaoa_ansatz(self, data_sample):
        """
        데이터 기반 QAOA ansatz 생성.
        """
        num_qubits = self.determine_num_qubits(data_sample)
        hamiltonian = self.create_data_based_hamiltonian(data_sample)

        circuit = QuantumCircuit(num_qubits)
        gamma = Parameter('γ')
        beta = Parameter('β')

        for i in range(num_qubits):
            circuit.rz(2 * gamma * hamiltonian.coeffs[i], i)

        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rx(2 * beta, i)
            circuit.rx(2 * beta, i + 1)

        return circuit

    def optimize_circuit_qaoa(self, data_sample):
        """
        QAOA 회로를 최적화.
        """
        if self.cached_result is not None:
            return self.cached_result

        hamiltonian = self.create_data_based_hamiltonian(data_sample)
        optimizer = COBYLA(maxiter=100)
        backend = AerSimulator()
        estimator = Estimator()

        circuit = self.create_manual_qaoa_ansatz(data_sample)
        vqe = VQE(estimator, ansatz=circuit, optimizer=optimizer)
        result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

        self.cached_result = result.optimal_value if result else 0
        return self.cached_result

# 양자회로와 ANM 블록을 결합한 모델 정의
class QNN_CD_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNN_CD_Model, self).__init__()
        self.anm_block = AdditiveNoiseModel(input_dim, hidden_dim, output_dim)
        self.quantum_circuit = QuantumCircuitQAOA()

    def forward(self, x):
        # ANM 블록 출력
        anm_output = self.anm_block(x)

        # 각 데이터 샘플에 대해 QAOA 최적화 수행
        qaoa_results = []
        for sample in x:
            qaoa_result = self.quantum_circuit.optimize_circuit_qaoa(sample)
            qaoa_results.append(qaoa_result)

        # QAOA 결과를 텐서로 변환
        qaoa_tensor = torch.tensor(qaoa_results, device=anm_output.device).view(-1, 1)
        qaoa_tensor = qaoa_tensor.expand_as(anm_output)

        # ANM 결과와 결합
        combined_output = anm_output + qaoa_tensor
        return combined_output

# 정확도 계산 함수
def calculate_accuracy(predictions, labels):
    predicted_labels = torch.argmax(predictions, dim=1)
    labels = labels.cpu()
    
    
    
    return accuracy_score(labels.cpu(), predicted_labels.cpu())




# Dataset 로드
dataset = TubingenDatasetHighDim(
    folder_path='/root/QnnCD/pairs(csv)',
    meta_path='/root/QnnCD/pairs(csv)/pairmeta.csv'
)
valid_data = dataset.get_valid_data()

# 도메인별 데이터 분리
def organize_domains(valid_data):
    domains = {}
    for sample in valid_data:
        file_name = sample['file_name']
        if file_name not in domains:
            domains[file_name] = {'x': [], 'y': [], 'label': [], 'weight': []}
        domains[file_name]['x'].append(sample['x'])
        domains[file_name]['y'].append(sample['y'])
        domains[file_name]['label'].append(sample['label'])
        domains[file_name]['weight'].append(sample['weight'])
    return domains

domains = organize_domains(valid_data)
# 도메인별 학습 및 평가
def train_and_evaluate_domain(domain, data):
    print(f"\nProcessing Domain: {domain}")
    
    try:
        
        # 데이터 전처리
        x = data['x'][0].squeeze()  # [N] 형태로 변환
        y = data['y'][0].squeeze()  # [N] 형태로 변환
        features = torch.stack([x, y], dim=1)  # [N, 2] 형태로 변환
        
        # 레이블 처리 - 레이블을 N번 반복
        n_samples = features.shape[0]
        label_value = data['label'][0]  # 첫 번째 레이블 값 사용
        labels = torch.tensor([label_value] * n_samples, dtype=torch.long)
        
        # 가중치 처리 - 가중치를 N번 반복
        weight_value = data['weight'][0]  # 첫 번째 가중치 값 사용
        weights = torch.tensor([weight_value] * n_samples, dtype=torch.float32)
        
        # 데이터 스플릿
        train_data, test_data, train_labels, test_labels = train_test_split(
            features.numpy(), 
            labels.numpy(),
            test_size=0.2,
            random_state=42
        )
        
        # 텐서로 변환
        train_data = torch.FloatTensor(train_data)
        test_data = torch.FloatTensor(test_data)
        train_labels = torch.LongTensor(train_labels)
        test_labels = torch.LongTensor(test_labels)
        
        # DataLoader 생성
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        
        batch_size = min(8, len(train_data))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 모델 초기화
        model = QNN_CD_Model(input_dim=2, hidden_dim=16, output_dim=2).to(device)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # 학습 및 평가 루프
        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            batch_count = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(batch_data)
                #print(model(batch_data))
                # 배치에 해당하는 가중치 사용
                batch_indices = range(batch_count * batch_size, 
                                   min((batch_count + 1) * batch_size, len(train_data)))
                
                batch_weights = weights[list(batch_indices)].to(device)
                loss = (criterion(outputs, batch_labels) * batch_weights).mean()
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            # 평가
            model.eval()
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                    outputs = model(batch_data)
                    all_predictions.append(outputs)
                    all_labels.append(batch_labels)

            all_predictions = torch.cat(all_predictions)
            all_labels = torch.cat(all_labels)
            accuracy = calculate_accuracy(all_predictions, all_labels)
            

            print(f"Domain: {domain}, Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error processing domain {domain}: {e}")
        import traceback
        print(traceback.format_exc())
        
    

# 도메인별 실행
for domain, data in domains.items():
    train_and_evaluate_domain(domain, data)


'''
RC => 차원을 맞춰야함
모델 구조상 차원 맞추기가 조금 힘듦
'''