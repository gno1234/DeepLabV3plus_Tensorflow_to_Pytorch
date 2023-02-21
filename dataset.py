class CustomDataset(data.Dataset):
  def __init__(self, images, masks):
    self.images = images
    self.masks = masks

		# x===============================================
    self.x_data = []
    for image_path in self.images:
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = image/125.7 -1
      image = [np.transpose(image,(2,0,1))]
      self.x_data += image

    # y===============================================
    self.y_data = []
    for mask_path in self.masks:
      mask = [cv2.imread(mask_path , 0)]
      self.y_data += mask

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.LongTensor(self.y_data[idx])
    return x, y