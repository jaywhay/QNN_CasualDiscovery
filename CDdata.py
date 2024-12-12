import os
import pandas as pd
import torch
import numpy as np
import re
from torch.utils.data import Dataset

class TubingenDatasetHighDim(Dataset):
    def __init__(self, folder_path, meta_path):
        self.data = []
        self.weights = []
        
        # 메타데이터 읽기
        meta_data = pd.read_csv(meta_path, header=None, sep=r'[ \t]+', engine='python')
        meta_dict = {}
        for row in meta_data.itertuples():
            pair_num = f"{int(row[1]):04d}"
            meta_dict[pair_num] = {
                'x_start': int(row[2]),
                'x_end': int(row[3]),
                'y_start': int(row[4]),
                'y_end': int(row[5]),
                'weight': float(row[6])
            }
            
        def read_data_file(file_path):
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()

                # CSV 형식 확인
                if ',' in first_line:
                    # 모든 행(row) 읽기
                    df = pd.read_csv(file_path, header=None)  # 헤더가 없는 경우 header=None
                    return df.values.astype(np.float32)  # NumPy 배열로 변환
                else:
                    # 공백/탭 구분자로 시도
                    data = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            # 공백으로 분리하고 모든 값 사용
                            values = [float(val) for val in line.strip().split()]
                            if values:
                                data.append(values)

                    data_array = np.array(data, dtype=np.float32)
                    return data_array
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return None

        def find_direction_in_file(file_content):
            """파일 전체 내용에서 방향성 찾기"""
            # 텍스트 전처리
            content = file_content.lower()
            
            # x->y 패턴 찾기 (다양한 형식 포함)
            x_to_y_patterns = [
                r'x\s*-+\s*>\s*y',         # x->y, x --> y, x  >  y
                r'x\s*-\s*-\s*>\s*y',      # x - - > y
                r'x\s*-+\s*-+\s*>\s*y',    # x -- > y, x --- > y
                r'x\s*→\s*y',              # x→y (유니코드 화살표)
                r'x\s+to\s+y',             # x to y
                r'y\s*<-\s*x',             # y <- x
                r'y\s*←\s*x',              # y ← x (유니코드 화살표)
                r'ground\s*truth.*x\s*-*>\s*y',  # ground truth x->y
                r'ground\s*truth.*x\s+to\s+y',   # ground truth x to y
                r'ground\s*truth.*x\s*-\s*-\s*>\s*y'  # ground truth x - - > y
            ]
            
            # y->x 패턴 찾기
            y_to_x_patterns = [
                r'y\s*-+\s*>\s*x',         # y->x, y --> x, y  >  x
                r'y\s*-\s*-\s*>\s*x',      # y - - > x
                r'y\s*-+\s*-+\s*>\s*x',    # y -- > x, y --- > x
                r'y\s*→\s*x',              # y→x (유니코드 화살표)
                r'y\s+to\s+x',             # y to x
                r'x\s*<-\s*y',             # x <- y
                r'x\s*←\s*y',              # x ← y (유니코드 화살표)
                r'ground\s*truth.*y\s*-*>\s*x',  # ground truth y->x
                r'ground\s*truth.*y\s+to\s+x',   # ground truth y to x
                r'ground\s*truth.*y\s*-\s*-\s*>\s*x',  # ground truth y - - > x
                r'ground\s*truth.*x\s*<-\s*y'    # ground truth x <- y
            ]
            
            # x->y 패턴 매칭 (대문자 버전 포함)
            for pattern in x_to_y_patterns:
                if re.search(pattern, content) or re.search(pattern.replace('x', 'X').replace('y', 'Y'), file_content):
                    return 1
                    
            # y->x 패턴 매칭 (대문자 버전 포함)
            for pattern in y_to_x_patterns:
                if re.search(pattern, content) or re.search(pattern.replace('x', 'X').replace('y', 'Y'), file_content):
                    return 0
            
            # 매칭되지 않는 경우
            print(f"Warning: Unable to determine direction from text: '{file_content.strip()}'")
            return -1
        
        # 파일 처리
        successful_loads = 0
        skipped_files = []  # 레이블을 결정할 수 없는 파일들 추적
        
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith('.csv') and not file_name.endswith('_des.csv'):
                pair_num = file_name[4:8]
                csv_path = os.path.join(folder_path, file_name)
                des_file_name = file_name.replace('.csv', '_des.csv')
                des_path = os.path.join(folder_path, des_file_name)
                
                if pair_num in meta_dict:
                    meta = meta_dict[pair_num]
                    
                    # 데이터 읽기
                    raw_data = read_data_file(csv_path)
                    
                    if raw_data is not None and len(raw_data) > 0:
                        try:
                            # 메타데이터의 범위에 따라 X와 Y 데이터 분리
                            x_start = meta['x_start'] - 1  # 1-based to 0-based
                            x_end = meta['x_end']
                            y_start = meta['y_start'] - 1  # 1-based to 0-based
                            y_end = meta['y_end']

                            # x_data는 첫 번째 열만 사용
                            x_data = raw_data[:, x_start:x_end]  # 행(row)을 기준으로 범위를 지정하고, 열(column) 명시
                            # y_data는 두 번째 열만 사용
                            y_data = raw_data[:, y_start:y_end]  # 행(row)을 기준으로 범위를 지정하고, 열(column) 명시

                            # 빈 데이터 확인
                            if len(x_data) == 0 or len(y_data) == 0:
                                print(f"Skipping {file_name} due to empty data")
                                continue
                            
                            # 텐서로 변환
                            x_tensor = torch.tensor(x_data, dtype=torch.float32)
                            y_tensor = torch.tensor(y_data, dtype=torch.float32)
                            
                            # des 파일에서 방향성 정보 읽기
                            if os.path.exists(des_path):
                                with open(des_path, 'r') as des_file:
                                    des_content = des_file.read()
                                    label = find_direction_in_file(des_content)
                                    
                                    if label == -1:
                                        skipped_files.append((file_name, des_content.strip()))
                            else:
                                label = -1
                                print(f"\nWarning: No description file found for {file_name}")
                                print("Default label assigned: -1 (unknown)")
                                skipped_files.append((file_name, "No description file"))
                            
                            # 데이터 저장 (레이블이 유효한 경우만)
                            if label != -1:
                                self.data.append({
                                    'x': x_tensor,
                                    'y': y_tensor,
                                    'label': label,
                                    'weight': meta['weight'],
                                    'file_name': file_name
                                })
                                successful_loads += 1
                            
                        except Exception as e:
                            print(f"Data processing error in {file_name}: {e}")
                            continue
        
        print(f"\nSuccessfully loaded {successful_loads} files")
        
        # 레이블 통계 출력
        label_counts = {1: 0, 0: 0, -1: 0}
        for item in self.data:
            label_counts[item['label']] += 1
            '''
        print("\nLabel Distribution:")
        print(f"Label 1 (x→y): {label_counts[1]} samples")
        print(f"Label 0 (y→x): {label_counts[0]} samples")'''
        
        # 스킵된 파일 정보 출력
        if skipped_files:
            print("\nSkipped files due to unclear direction:")
            for file_name, content in skipped_files:
                print(f"\n{file_name}:")
                print(f"Content: {content}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample['x'], sample['y'], sample['label'], sample['weight']
    
    def get_dimensions(self):
        """각 데이터셋의 차원 정보를 반환"""
        dimensions = []
        for sample in self.data:
            dimensions.append({
                'file': sample.get('file_name', 'unknown'),
                'x_dim': tuple(sample['x'].shape),
                'y_dim': tuple(sample['y'].shape)
            })
        return dimensions
    
    def get_label_summary(self):
        """데이터셋의 라벨 분포를 반환"""
        label_count = {0: 0, 1: 0, -1: 0}
        file_labels = []
        
        for sample in self.data:
            label = sample['label']
            label_count[label] += 1
            file_labels.append({
                'file': sample['file_name'],
                'label': label,
                'weight': sample['weight']
            })
            
        return {
            'summary': label_count,
            'details': file_labels
        }
    def get_valid_data(self):
        special_cases = ["pair0052.csv", "pair0053.csv", "pair0054.csv", "pair0055.csv", "pair0071.csv", "pair0105.csv"]
        return [
            sample for sample in self.data if sample['file_name'] not in special_cases
        ]
    def sample_out(self):
        for sample in self.data:
            print(f"File: {sample['file_name']}, x shape: {sample['x'].shape}, y shape: {sample['y'].shape}")   



# 테스트 코드
if __name__ == "__main__":
    dataset = TubingenDatasetHighDim(
        folder_path='/root/QnnCD/pairs(csv)',
        meta_path='/root/QnnCD/pairs(csv)/pairmeta.csv'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # 라벨링 정보 확인
    label_info = dataset.get_label_summary()
    print("\nLabel Distribution:")
    print(f"Label 0 (y→x): {label_info['summary'][0]} samples")
    print(f"Label 1 (x→y): {label_info['summary'][1]} samples")
    print(f"Label -1 (unknown): {label_info['summary'][-1]} samples")
    
    print("\nDetailed Label Information:")
    for item in label_info['details']:
        print(f"File: {item['file']}, Label: {item['label']}, Weight: {item['weight']}")
    
    # 차원 정보 확인
    dimensions = dataset.get_dimensions()
    print("\nDimension Information for Special Cases:")
    for dim in dimensions:
        if dim['x_dim'][0] > 1 or dim['y_dim'][0] > 1:  # 특이 케이스만 출력
            print(f"\nFile: {dim['file']}")
            print(f"X dimensions: {dim['x_dim']}")
            print(f"Y dimensions: {dim['y_dim']}")