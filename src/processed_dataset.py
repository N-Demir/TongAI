from torch.utils.data import Dataset

class HospitalDataset(Dataset):

	def __init__(self, csv_file):
		