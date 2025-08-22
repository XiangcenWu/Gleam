import os
import random
import h5py
import torch
from monai.data import Dataset, DataLoader
import logging
from monai.transforms import *
from sklearn.metrics import auc

def data_spilt(base_dir, num_train, seed=325):
    list = os.listdir(base_dir)
    list = [os.path.join(base_dir, item) for item in list]
    random.seed(seed)
    random.shuffle(list)
    return list[:num_train], list[num_train:]

def data_all(*base_dirs):
    all_list = []
    for base_dir in base_dirs:
        items = os.listdir(base_dir)
        items = [os.path.join(base_dir, item) for item in items]
        all_list.extend(items)
    return all_list





class ReadH5Pkld():
    def __init__(self):
        super().__init__()

    def __call__(self, file_dir):
        img, pirads, gleason, patient_name, patient_uid = self.load_h5_file(file_dir)
        

        return {
            'img': img,
            'patient_name': patient_name,
            'patient_uid': patient_uid,
            'pirads': pirads,
            'gleason': gleason,
            'file_dir': file_dir
        }
        
    def load_h5_file(self, file_dir):
    
        h5f = h5py.File(file_dir, 'r')


        mri = torch.from_numpy(h5f['img'][:])
        patient_name = h5f['patient_name'][()].decode('utf-8')
        patient_uid = h5f['patient_uid'][()].decode('utf-8')
        
        pirads = int(h5f['pirads'][()])
        gleason = int(h5f['gleason'][()])

        
        h5f.close()

        return mri, pirads, gleason, patient_name, patient_uid
        
        

def get_loader(
        list, 
        transform, 
        batch_size: int,
        shuffle: bool=True, 
        drop_last: bool=False, 
    ):
    _ds = Dataset(list, transform=transform)

    return DataLoader(
        dataset = _ds,
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last
    )



def read_series_labels(txt_path):
    result = {}
    with open(txt_path, 'r') as f:
        for line in f:
            uid, label = line.strip().split('\t')
            result[uid] = int(label)
    return result
def check_patient_PN(batch, patient_pathology_dict):

    uid = batch['patient_uid']
    print(uid[0])
    patient_PN = patient_pathology_dict[uid[0]]
    
    return patient_PN



train_transform = Compose([
        ReadH5Pkld(),
        RandAffined(['img'], spatial_size=(128, 128, 64), prob=0.25, shear_range=(0., 0.5, 0., 0.5, 0., 0.2), mode='nearest', padding_mode='zeros'),
        RandGaussianSmoothd(['img'], prob=0.25),
        RandGaussianNoised(['img'], prob=0.25, std=0.05),
        RandAdjustContrastd(['img'], prob=0.25, gamma=(0.5, 2.))
    ])
def sample_two_different(lists):

    # pick one sample from each class
    sample1 = random.choice(lists[0])
    sample2 = random.choice(lists[1])
    return sample1, sample2
def read_txt_to_list(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines
gleason_0_list = read_txt_to_list('gleason_spilts/gleason_0.txt')
gleason_1_list = read_txt_to_list('gleason_spilts/gleason_1.txt')
gleason_2_list = read_txt_to_list('gleason_spilts/gleason_2.txt')
gleason_3_list = read_txt_to_list('gleason_spilts/gleason_3.txt')
gleason_4_list = read_txt_to_list('gleason_spilts/gleason_4.txt')
gleason_5_list = read_txt_to_list('gleason_spilts/gleason_5.txt')
gleason_list_list = [
    gleason_0_list+gleason_1_list,
    gleason_2_list+gleason_3_list+gleason_4_list+gleason_5_list,
]
def train_net(
        model, 
        train_loader,
        train_optimizer,
        train_loss,
        contrast_loss,
        device='cpu',
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.train()
    
    _step = 0.
    _loss = 0.
    for batch in train_loader:

        img, label = batch["img"].to(device), batch["gleason"].to(device)

        pirads = batch["pirads"].to(device)

        output = model(img)


        loss = train_loss(output, label, pirads)

        loss.backward()
        train_optimizer.step()
        train_optimizer.zero_grad()
        for _ in range(1):
            data_pos_dir, data_neg_dir = sample_two_different(gleason_list_list)
            train_net_contrast(model, data_pos_dir, data_neg_dir, train_transform, train_optimizer, contrast_loss, device)

        _loss += loss.item()
        _step += 1.
    _epoch_loss = _loss / _step

    return _epoch_loss



@torch.no_grad()
def get_roc(model, 
            train_loader,
            device='cpu', 
            steps=10,
            gleason_cutoff=2,
            mode=None,
            cspc_img=None,
            non_cspc_img = None
        ):
    model.to(device)
    model.eval()

    tp_tensor = torch.zeros(size=(steps, ))
    tn_tensor = torch.zeros(size=(steps, ))
    roc_cutoffs = torch.linspace(0, 1, steps)


    P, N = 0, 0
    
    
        

    for batch in train_loader:
        img, label = batch["img"].to(device), batch["gleason"].to(device)
        pirads = batch['pirads'].to(device)
        
        if mode == 'zero-shot':
            output = model.inference_zero_shot_cspca_classification(img, cspc_img.to(device), non_cspc_img.to(device))
            print(output)
            pos_prob = output[1]
        else:
            output = model(img)
            output = torch.softmax(output, dim=-1).squeeze(0)
            pos_prob = output[gleason_cutoff:].sum()



        if label.item() >= gleason_cutoff:
            P += 1
            for i, cutoff in enumerate(roc_cutoffs):
                if pos_prob >= cutoff:
                    tp_tensor[i] += 1
        else:
            N += 1
            for i, cutoff in enumerate(roc_cutoffs):
                if pos_prob < cutoff:
                    tn_tensor[i] += 1

    tpr, tnr = tp_tensor / P, tn_tensor / N
    fpr = 1 - tnr
    # Compute AUC using trapezoidal integration
    auroc = auc(fpr.cpu().numpy(), tpr.cpu().numpy())
    
    return tpr, tnr, auroc


@torch.no_grad()
def inference_radiologist(
        train_loader,
        gleason_cutoff=2
    ):
    tp_tensor = torch.zeros(size=(5, ))
    tn_tensor = torch.zeros(size=(5, ))
    radiologist_cutoffs = torch.tensor([0,3,4,5,10])

    P, N = 0, 0
    

    for batch in train_loader:
        label = batch["gleason"]
        pirads = batch['pirads']


        if label.item() >= gleason_cutoff:
            P += 1
            for i, cutoff in enumerate(radiologist_cutoffs):
                if pirads >= cutoff:
                    tp_tensor[i] += 1
        else:
            N += 1
            for i, cutoff in enumerate(radiologist_cutoffs):
                if pirads < cutoff:
                    tn_tensor[i] += 1
                    
    tpr, tnr = tp_tensor / P, tn_tensor / N
    fpr = 1 - tnr
    # Compute AUC using trapezoidal integration
    auroc = auc(fpr.cpu().numpy(), tpr.cpu().numpy())
    
    return tpr, tnr, auroc








def train_net_contrast(
        model, 
        positive_dir,
        negative_dir,
        file_reader,
        train_optimizer,
        contrast_loss,
        device='cpu',
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.train()
    img_pos, img_neg = file_reader(positive_dir)["img"].to(device), file_reader(negative_dir)["img"].to(device)
    output_pos = model.forward_encoder(img_pos.unsqueeze(0))
    output_neg = model.forward_encoder(img_neg.unsqueeze(0))
    loss = contrast_loss(output_pos, output_neg)

    loss.backward()
    train_optimizer.step()
    train_optimizer.zero_grad()

    return loss.item()








promis_325_gleason_0_list = read_txt_to_list('/home/xiangcen/PatientBiopsyDetect/PROMIS_325_train_gleason/gleason_0.txt')
promis_325_gleason_1_list = read_txt_to_list('/home/xiangcen/PatientBiopsyDetect/PROMIS_325_train_gleason/gleason_1.txt')
promis_325_gleason_2_list = read_txt_to_list('/home/xiangcen/PatientBiopsyDetect/PROMIS_325_train_gleason/gleason_2.txt')
promis_325_gleason_3_list = read_txt_to_list('/home/xiangcen/PatientBiopsyDetect/PROMIS_325_train_gleason/gleason_3.txt')
promis_325_gleason_4_list = read_txt_to_list('/home/xiangcen/PatientBiopsyDetect/PROMIS_325_train_gleason/gleason_4.txt')
promis_325_gleason_5_list = read_txt_to_list('/home/xiangcen/PatientBiopsyDetect/PROMIS_325_train_gleason/gleason_5.txt')
promis_325_gleason_list_list = [
    promis_325_gleason_0_list+promis_325_gleason_1_list,
    promis_325_gleason_2_list+promis_325_gleason_3_list+promis_325_gleason_4_list+promis_325_gleason_5_list,
]
def lora_finetune(
        model, 
        train_loader,
        train_optimizer,
        train_loss,
        contrast_loss,
        with_contrast=True,
        device='cpu',
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.train()
    
    _step = 0.
    _loss = 0.
    for batch in train_loader:

        img, label = batch["img"].to(device), batch["gleason"].to(device)
        pirads = batch["pirads"].to(device)
        


        output = model(img)

        print(output, label)
        loss = train_loss(output, label, pirads)

        loss.backward()
        train_optimizer.step()
        train_optimizer.zero_grad()
        
        if with_contrast:
            for _ in range(1):
                data_pos_dir, data_neg_dir = sample_two_different(promis_325_gleason_list_list)
                train_net_contrast(model, data_pos_dir, data_neg_dir, train_transform, train_optimizer, contrast_loss, device)
        
        


        _loss += loss.item()
        _step += 1.
    _epoch_loss = _loss / _step

    return _epoch_loss