import numpy as np

class ServerDataset:
    def __init__(self, train_data, train_labels, train_logits, test_data, test_labels):
        super(ServerDataset, self).__init__()
        self.TrainData = train_data
        self.TrainLabels = train_labels
        self.TrainLogits = train_logits
        self.TestData = test_data
        self.TestLabels = test_labels

    def concatenate(self, datas, labels, logits):
        con_data = datas[0]
        con_label = labels[0]
        con_logit = logits[0] if logits != [] else None
        
        if con_logit != None:
            for i in range(1, len(datas)):
                con_data = np.concatenate((con_data, datas[i]), axis=0)
                con_label = np.concatenate((con_label,labels[i]), axis=0)
                con_logit = np.concatenate((con_logit, logits[i]), axis=0)
        else:
            for i in range(1, len(datas)):
                con_data = np.concatenate((con_data, datas[i]), axis=0)
                con_label = np.concatenate((con_label,labels[i]), axis=0)
        
        return con_data, con_label, con_logit

    def getTrainItem(self, index):
        return index, self.TrainData[index], self.TrainLabels[index], self.TrainLogits[index]

    def getTestItem(self, index):
        return index, self.TestData[index], self.TestLabels[index]

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)
