import pandas as pd

def making_minor_Y(gt_path, output_path):
    labels = [
        '가족정책', '돌봄', '저출산', '일생활균형', '건강', '교육정책', '직업교육', '성평등교육', 
        '평생교육', '국제협력', 'ODA', '고용', '경력단절', '노동정책', '인적자원개발', 
        '법제도', '남성', '다문화', '미래', '이주민', '취약계층', '통일', '성주류화정책', '성별영향평가',
        '성인지예산', '성인지통계', '성평등문화', '성평등의식', '성평등정책', '안전', '젠더폭력', '인권', '공공대표성', '민간대표성',
        '정치대표성'
    ]
    
    with open(gt_path, 'r', encoding='utf-8') as file:
        label_data = file.readlines()

    Y = []
    for idx, label_str in enumerate(label_data):
        label_list = [item.split('-')[1].strip() for item in label_str.strip().split(',') if '-' in item]
        label_vector = [1 if label in label_list else 0 for label in labels]
        Y.append(label_vector)
    Y_df = pd.DataFrame(Y, columns=labels)
    Y_df.to_csv(output_path, index=False, encoding='utf-8-sig')


gt_path = '/home/women/doyoung/Top2Vec/preprocessing/input/train_test_labels.txt'
output_path = '/home/women/doyoung/Top2Vec/preprocessing/output/Y_minor.csv'

making_minor_Y(gt_path, output_path)



def making_major_Y(gt_path, output_path):
    major_labels = [
        '가족', '건강', '교육', '국제협력', '노동', '법제도', '사회통합', '성주류화', '성평등사회', '안전', '젠더폭력', '대표성'
    ]
    with open(gt_path, 'r', encoding='utf-8') as file:
        label_data = file.readlines()

    Y = []
    for idx, label_str in enumerate(label_data):
        major_label_list = [item.split('-')[0].strip() for item in label_str.strip().split(',') if '-' in item]
        
        label_vector = [1 if label in major_label_list else 0 for label in major_labels]
        Y.append(label_vector)
    
    Y_df = pd.DataFrame(Y, columns=major_labels)
    
    Y_df.to_csv(output_path, index=False, encoding='utf-8-sig')


gt_path = '/home/women/doyoung/Top2Vec/preprocessing/input/train_test_labels.txt'
output_path = '/home/women/doyoung/Top2Vec/preprocessing/output/Y_major.csv'

making_major_Y(gt_path, output_path)


