from sys import argv

from torch import nn, optim, FloatTensor, LongTensor, no_grad, save
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import cuda
import matplotlib.pyplot as plt

features = True if '-feat' in argv else False

EMBEDDING_VECTOR = 50
device = 'cuda' if cuda.is_available() else 'cpu'
corpus = argv[4]
variation = argv[1]

class NeuralNetwork(nn.Module):
    def __init__(self,output_shape,word_embedding=None, window=5,voc_emb_size=None, hidden=100):
        super().__init__()
        self.window = window
        if word_embedding is not None:
            word_embedding = FloatTensor(word_embedding)
            self.embedded = nn.Embedding.from_pretrained(word_embedding, freeze=False).to(device)
        else:
            self.embedded = nn.Embedding(voc_emb_size, EMBEDDING_VECTOR).to(device)

        input_dense = EMBEDDING_VECTOR * window
        self.hidden = nn.Linear(input_dense, hidden)
        self.output = nn.Linear(hidden, output_shape)

        self.drop_1 = nn.Dropout()
        self.drop_2 = nn.Dropout()

        # Define tanh activation and softmax output
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        if features:
            x = self.embedded(x).sum(dim=2).view(-1, EMBEDDING_VECTOR * self.window)
        else:
            x = self.embedded(x).view(-1, EMBEDDING_VECTOR * self.window)
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.drop_2(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


def train(model, data, label, batch_size=1024, lr=0.01,val_data=None, val_label=None, isNer=False, epoch=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(data, label)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_data is not None:
        dataset = TensorDataset(val_data, val_label)
        val_loader = DataLoader(dataset=dataset, batch_size=1024)
    loss_list = []
    for t in range(epoch):
        print(f"EPOCH: {t}")
        acc_batch = 0
        time_2_print = 50000
        for x_batch, y_batch in train_loader:
            if acc_batch > time_2_print:
                time_2_print += 50000
                print(f"BATCH: {acc_batch}/{len(data)}")
            acc_batch += batch_size
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Sets model to TRAIN mode
            model.train()
            # Makes predictions
            yhat = model(x_batch)
            # Computes loss
            loss = criterion(yhat,y_batch)
            # Computes gradients
            loss.backward()
            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()
            # Returns the loss
            loss_list.append(loss.item())

        # evaluate(train_loader, model, criterion, "train", isNer)
        if val_loader is not None:
            evaluate(val_loader, model, criterion, "validation", isNer)
    plt.figure()
    plt.plot(acc_global)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(variation+'_acc_'+corpus+'.png')
    plt.figure()
    plt.plot(loss_global)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(variation+'_loss_'+corpus+'.png')
    return model


loss_global = []
acc_global = []


def predict(model,data):
    dataset = TensorDataset(data)
    loader = DataLoader(dataset=dataset, batch_size=512)
    results = []
    model.eval()
    with no_grad():
        for x_batch in loader:
            x_batch = x_batch[0].to(device)
            outputs = model(x_batch)
            results += outputs.argmax(axis=1).tolist()
    return results


def evaluate(loader, model, criterion, dataset,isNer=False):
    loss_list = []
    correct = 0
    total = 0
    model.eval()
    with no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss_list.append(criterion(outputs, y_batch).item())
            if isNer:
                for y_real,yhat in zip(y_batch,outputs.argmax(axis=1)):
                    y_real,yhat = int(y_real), int(yhat)
                    if parser.index_2_label[y_real] == 'O' and yhat == y_real:
                        continue
                    if yhat == y_real:
                        correct += 1
                    total += 1
            else:
                total += len(y_batch)
                correct += (outputs.argmax(axis=1) == y_batch).sum().item()
        acc_global.append(100*correct/total)
        loss_global.append(sum(loss_list))

    print(f'Accuracy of the network on the {dataset}: {100 * correct / total}% loss:{sum(loss_list)}')


class Parser(object):
    def __init__(self,word_id_file=None,word_dict_vec=None):
        if word_id_file is not None and \
                word_dict_vec is not None:
            self.word_dict = {k: list(map(float, v.split()))
                              for k, v in zip(open(word_id_file).read().split('\n'),
                                              open(word_dict_vec).read().split('\n')) if len(v.split())==50}
            self.word_id = {item: id_item for id_item, item in
                            enumerate(open('vocab.txt').read().split('\n'))
                            if len(item) != 0}
        else:
            self.word_id = {'UUUNKKK': 0}
        self.index_2_label = self.label_2_index = None
        self.uniq_labels = 0

    def load_input_file(self, input_file, to_train=False):
        data = []
        label = []
        labels_uniq = set()
        index = max(self.word_id.values())+1
        if to_train and len(self.word_id) < 2:
            self.word_id['<s>'] = 1
            self.word_id['</s>'] = 2
            index = 3
        if features:
            start_id = [self.word_id['<s>'], self.word_id['<s>'], self.word_id['<s>']]
            end_id = [self.word_id['</s>'], self.word_id['</s>'], self.word_id['</s>']]
        else:
            start_id = self.word_id['<s>']
            end_id = self.word_id['</s>']
        start = [start_id, start_id]
        end = [end_id, end_id]

        with open(input_file) as f:
            for line in f:
                sentence = []
                for word in line.split():
                    word_old, tag = word.rsplit('/', 1)
                    word, index = self.check_word_insert(word_old, index, to_train)

                    if features:
                        pref, index = self.check_word_insert('$'+word_old[:3], index, to_train)
                        suff, index = self.check_word_insert(word_old[-3:]+'$', index, to_train)
                        sentence.append([self.word_id[word],
                                         self.word_id[pref],
                                         self.word_id[suff]])
                    else:
                        sentence.append(self.word_id[word])
                    label.append(tag)
                    if to_train:
                        labels_uniq.add(tag)

                lst_tmp = start + sentence + end
                rng = zip(lst_tmp, lst_tmp[1:], lst_tmp[2:], lst_tmp[3:], lst_tmp[4:])
                data.append([item for item in rng])
        if to_train:
            self.index_2_label = {i: lbl for i, lbl in enumerate(labels_uniq)}
            self.label_2_index = {lbl: i for i, lbl in self.index_2_label.items()}
            self.uniq_labels = len(self.index_2_label)
        transform = np.array([ngram for sen in data for ngram in sen])
        data = LongTensor(transform)
        label = [float(self.label_2_index[lbl]) for lbl in label]
        label = LongTensor(label)
        return data, label

    def check_word_insert(self, word,index=0,to_train=False):
        if word not in self.word_id and to_train:
            self.word_id[word] = index
            if isEmbedded:
                self.word_dict[word] = np.random.uniform(-1, 1, 50)
            index += 1
        elif word not in self.word_id:
            word = 'UUUNKKK'
        return word, index

    def load_blind_file(self,input_file):
        data = []
        text = []
        if features:
            start_id = [self.word_id['<s>'], self.word_id['<s>'], self.word_id['<s>']]
            end_id = [self.word_id['</s>'], self.word_id['</s>'], self.word_id['</s>']]
        else:
            start_id = self.word_id['<s>']
            end_id = self.word_id['</s>']
        start = [start_id, start_id]
        end = [end_id, end_id]
        with open(input_file) as f:
            for line in f:
                sentence = []
                for word in line.split():
                    word_old = word
                    if word not in self.word_id:
                        word = 'UUUNKKK'
                    if features:
                        pref, _ = self.check_word_insert('$' + word_old[:3])
                        suff, _ = self.check_word_insert(word_old[-3:] + '$')
                        sentence.append([self.word_id[word],
                                         self.word_id[pref],
                                         self.word_id[suff]
                                         ])
                    else:
                        sentence.append(self.word_id[word])
                lst_tmp = start + sentence + end
                rng = zip(lst_tmp, lst_tmp[1:], lst_tmp[2:], lst_tmp[3:], lst_tmp[4:])
                data.append([item for item in rng])
                text.append(line.split())
        transform = np.array([ngram for sen in data for ngram in sen])
        data = LongTensor(transform)
        return data, text


def predict_test(file_name,file_out_name):
    test, text = parser.load_blind_file(file_name)
    results = predict(network, test)

    i = 0
    with open(file_out_name, 'w') as f:
        for line in text:
            res = ""
            for word in line:
                res += word + "/" + parser.index_2_label[results[i]] + ' '
                i += 1
            f.write(res + '\n')


var_param = {
    'tagger1': {
        'pos': {
            'hidden': 300,
            'isNer': False,
            'epoch_nbr': 25,
            'batch_size': 1024,
            'test': 'data/ass1-tagger-test-input',
            'lr': 0.01
        },
        'ner': {
            'hidden': 100,
            'isNer': True,
            'epoch_nbr': 25,
            'batch_size': 512,
            'test': 'ner/test.blind',
            'lr': 0.001
        }
    },
    'tagger2': {
        'pos': {
            'hidden': 300,
            'isNer': False,
            'epoch_nbr': 25,
            'batch_size': 64,
            'test': 'data/ass1-tagger-test-input',
            'lr': 0.1
        },
        'ner': {
            'hidden': 100,
            'isNer': True,
            'epoch_nbr': 25,
            'batch_size': 1024,
            'test': 'ner/test.blind',
            'lr': 0.01
        }
    },
    'tagger4': {
        'pos': {
            'hidden': 500,
            'isNer': False,
            'epoch_nbr': 25,
            'batch_size': 1024,
            'test': 'data/ass1-tagger-test-input',
            'lr': 0.01
        },
        'ner': {
            'hidden': 128,
            'isNer': True,
            'epoch_nbr': 40,
            'batch_size': 128,
            'test': 'ner/test.blind',
            'lr': 0.01
        }
    }
}

if __name__ == '__main__':

    train_file = argv[2]
    dev_fie = argv[3]
    if len(argv) == 7:
        emb_voc = argv[5]
        emb_voc_vec = argv[6]
        parser = Parser(word_id_file=emb_voc, word_dict_vec=emb_voc_vec)
        isEmbedded = True
    else:
        parser = Parser()
        emb_voc_vec = None
        isEmbedded = False


    print("Parsing file")

    data, label = parser.load_input_file(train_file, True)
    val_data, val_label = parser.load_input_file(dev_fie)

    print("Creating Neural network")
    embedded = np.array(list(parser.word_dict.values())) if isEmbedded else None

    network = NeuralNetwork(output_shape=parser.uniq_labels,
                            voc_emb_size=len(parser.word_id),
                            word_embedding=embedded,
                            hidden=var_param[variation][corpus]['hidden']).to(device)

    print("Start training phase")
    network = train(network, data, label, batch_size=var_param[variation][corpus]['batch_size'],
                    val_data=val_data,
                    val_label=val_label,
                    lr=0.01,
                    isNer=var_param[variation][corpus]['isNer'],
                    epoch=var_param[variation][corpus]['epoch_nbr'])

    predict_test(var_param[variation][corpus]['test'], 'test2.' + corpus)


