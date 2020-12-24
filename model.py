import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,drop_prob=0.0):
        
        super().__init__()
        # define variables
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        
        # define word embedding layer: mapping dict to embedding vector
        self.embed = nn.Embedding(vocab_size, embed_size)
        # define lstm layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=drop_prob, batch_first=True)
        # define a fully connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        # init the weight
        self.init_weights()     
    
    def forward(self, features, captions):
        " Decode image feature vectors and generates captions."
        # features: batch_size x embed_size
        # caption, disgard the end word: batch_size x (caption_length-1) x embed_size
        captions_embed = self.embed(captions[:,:-1]) 
        
        # cat two together; size: batch_size x caption_length x embed_size
        all_input_embed = torch.cat((features.unsqueeze(1), captions_embed), 1) 
        hiddens, _ = self.lstm(all_input_embed)
        
        # output 
        outputs = self.fc(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """ accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) """
        
        # input size: batch_size x 1 x embed_size
        n_image = inputs.size(0)
        all_outputs =[]
        
        # for each img
        for n in range(n_image):
            
            # initialize h,c for each image
            h = torch.zeros(1,1,self.hidden_size)
            c = torch.zeros(1,1,self.hidden_size)
            outputs = [];
            
            input_sample = inputs[n,:,:].unsqueeze(1)     # n-th image

            for step in range(max_len):
                # one step output
                o,(h, c) = self.lstm(input_sample,(h,c))  # 1 x 1x embed_size
                output  = F.log_softmax(self.fc(o).squeeze())     # fc(h): 1 x vocab_size
                output_ind = output.max()[1]
                
                if step == 0: # first step
                    outputs.append(0)      # start word
                else:
                    outputs.append(output_ind)
                    
                if output_ind == 1:      # end word
                    break
                
                # prepare for the next input
                input_sample = self.embed(output_ind).unsqueeze(1)
               
            # check the last output
            if outputs[n,-1] != 1:     # out of range 
                outputs[n,-1] = 1
            
            # append outpouts
            all_outputs.append(outputs)  
            
        return all_outputs
                          
    
    def init_weights(self):
        "Initialize the weights."
        self.embed.weight.data.uniform_(-0.1, 0.1)  # W_e
        self.fc.bias.data.fill_(0) # b
        self.fc.weight.data.uniform_(-0.1, 0.1) # W_fc