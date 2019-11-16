import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) 
        self.fc = nn.Linear(hidden_size, vocab_size)

    
    def forward(self, features, captions):      

        captions = captions[:,:-1]
        embed = self.embed(captions)

        input = torch.cat((features.unsqueeze(dim=1), embed), dim=1)
        
        output, _ = self.lstm(input)
        captions = self.fc(output)
        return captions

    def sample(self, inputs, states=None, max_len=20):
        
        predicted_captions = []

        for i in range(max_len):

            output, states = self.lstm(inputs, states)
            output = self.fc(output)
            _, prediction = torch.max(output, dim=2)
            predicted_captions.append(torch.squeeze(prediction).item())          
            inputs = self.embed(prediction)
        
        return predicted_captions