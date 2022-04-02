from utils.generateVocabulary import loadVocabulary
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tqdm import tqdm
from utils.metrics import BLEU, METEOR
#from utils.metrics import BLEU, CIDEr, SPICE, ROUGE, METEOR

def plotImagesAndCaptions(model, modelParam, config, dataLoader):
    is_train = False
    # dataDict = next(iter(dataLoader.myDataDicts['val']))

    fig, ax = plt.subplots()
    

    
    
    # for dataDict in dataLoader.myDataDicts['val']:
    
    #dataDict = next(iter(dataLoader.myDataDicts['val']))

    for i, dataDict in enumerate(tqdm(dataLoader.myDataDicts['val'])):
        #print(i)
        #if i > 50: 
        #    break

        for key in ['xTokens', 'yTokens', 'yWeights', 'cnn_features']:
            dataDict[key] = dataDict[key].to(model.device)
        for idx in range(dataDict['numbOfTruncatedSequences']):
            # for iter in range(1):
            xTokens = dataDict['xTokens'][:, :, idx]
            yTokens = dataDict['yTokens'][:, :, idx]
            yWeights = dataDict['yWeights'][:, :, idx]
            cnn_features = dataDict['cnn_features']
            if idx == 0:
                logits, current_hidden_state = model.net(cnn_features, xTokens, is_train)
                predicted_tokens = logits.argmax(dim=2).detach().cpu()
            else:
                logits, current_hidden_state = model.net(cnn_features, xTokens, is_train, current_hidden_state)
                predicted_tokens = torch.cat((predicted_tokens, logits.argmax(dim=2).detach().cpu()), dim=1)


        vocabularyDict = loadVocabulary(modelParam['data_dir'])
        TokenToWord = vocabularyDict['TokenToWord']

        #wordToToken
        #TokenToWord

        #print('predicted_tokens.shape',predicted_tokens.shape)

        batchInd = 0
        #for batchInd in range(modelParam['batch_size']):

        sentence = []
        foundEnd = False
        for kk in range(predicted_tokens.shape[1]):
            word = TokenToWord[predicted_tokens[batchInd, kk].item()]
            if word == 'eeee':
                foundEnd = True
            if foundEnd == False:
                sentence.append(word)

        #meteor = METEOR().calculate(predicted_tokens, references, tokenize= False)

        """
        #print captions
        print('\n')
        print('Generated caption')
        print(" ".join(sentence))
        print('\n')
        print('Original captions:')
        for kk in range(len(dataDict['orig_captions'][batchInd])):
            print(dataDict['orig_captions'][batchInd][kk])

        print('\n')
        """
        # show image
        imgpath = modelParam['data_dir'] + modelParam['modeSetups'][0][0] + '2017/'+dataDict['imgPaths'][batchInd]
        img = mpimg.imread(imgpath)
        #plt.ion()
        


        ax.imshow(img)
        ax.axis('off')
        ref = [f"{dataDict['orig_captions'][batchInd][kk]}\n" 
                            for kk in range(len(dataDict['orig_captions'][batchInd]))]

        ax.set_title("Generated:\n" + " ".join(sentence)
                      + "\nReferences:\n"
                      + " ".join(ref),
                      loc = "left"
        )
                
        fig.tight_layout()

        pt = './captioned_images/'
        imgname = imgpath.split("/")[-1]
        if not os.path.isdir(pt):
          os.makedirs(pt)
        path = pt + modelParam['modelName'][:-1] + "_" + imgname
        
        fig.savefig(path, bbox_inches = "tight")
        #print(imgpath)
        #plt.show()
        aa = 1
    return

########################################################################################################################



