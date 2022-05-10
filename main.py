from operator import ne
import os
import torch
import argparse
import random
import numpy as np
from model import SASRec
from SASRec_utils import *
from UFN_utils import *
from torch_ema import ExponentialMovingAverage

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="Beauty")
parser.add_argument('--name', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--neg_nums', default=100, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--reverse',default=5,type=int)
parser.add_argument('--lbd',default=0.3,type=float)
parser.add_argument('--decay',default=0.999,type=float)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.name = "./" + args.name
if not os.path.isdir(args.dataset + '_' + args.name):
    os.makedirs(args.dataset + '_' + args.name)
with open(os.path.join(args.dataset + '_' + args.name, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.name, 'log.txt'), 'w')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3,
                          neg_nums=args.neg_nums)
    model = SASRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.train()  # enable model training
    epoch_start_idx = 1

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    loss_lst = [] 
    
    flag = False # flag used to decide whether to do negative sampling

    hard_items = {} # record the user's fixed ordinal position's hard negs in in the last training. 
    # key:(user_id, position) values:[hard_negs,....]

    cnt_items = {} # record frequency of the user's fixed ordinal position's hard negs
    # key:(user_id, position, hard_neg_id) values:The number of occurrences of hard_neg_id, int
    
    ema = None

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        sum_loss = 0  
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg) 
          
            reverse = hard_neg_sample(u, seq, pos, neg, hard_items, cnt_items, args)

            pos_ = np.expand_dims(pos, 2)
            all_samples = np.concatenate((pos_, neg), axis=2)
            all_logits = model(u, seq, all_samples)

            teacher_logits = None
            if flag:
                if ema == None:
                    ema = ExponentialMovingAverage(model.parameters(), decay=args.decay)
                    ema.to(device = args.device)
                else:
                    with ema.average_parameters():
                        teacher_logits = model(u, seq, all_samples)
                        teacher_logits = teacher_logits.detach()

            pos_labels, neg_labels = torch.ones(all_logits.shape[0], all_logits.shape[1], 1), torch.zeros(
                all_logits.shape[0], all_logits.shape[1], all_logits.shape[2] - 1)
            
            dim0, dim1, dim2 = [],[],[]
            cnt = 0
            for i in reverse: 
                neg_labels[i[0]][i[1]][i[2]] = 1
                dim0.append(i[0])
                dim1.append(i[1])
                dim2.append(i[2] + 1)
                cnt += 1
            reverse_indices = (np.array(dim0), np.array(dim1), np.array(dim2))
                
            if (step == 0):
                print("num of labels reversed: ",cnt)
            
            all_labels = torch.cat((pos_labels, neg_labels), dim=2)
            all_labels = all_labels.to(device = args.device)

            indices = np.where(all_samples != 0)
            
            bce_criterion = torch.nn.BCEWithLogitsLoss()
            loss = bce_criterion(all_logits[indices], all_labels[indices])
            
            loss = loss * 2
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            
            if teacher_logits != None and len(reverse_indices[0]):
                loss += args.lbd * bce_criterion(all_logits[reverse_indices], torch.sigmoid(teacher_logits[reverse_indices]))
                
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            sum_loss += loss.item()
            if flag:
                ema.update()

            if ((not flag) and len(loss_lst) > 7 and loss_lst[epoch - 3] - loss_lst[epoch - 2] < (loss_lst[0] - loss_lst[5])/200):
                flag = True

            if flag:
                update_hard_negs(u, pos, hard_items, all_logits, all_labels, all_samples, args)

            # print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
        
        loss_lst.append(sum_loss)
        
        print("loss in epoch {} : {}".format(epoch + 1, sum_loss / num_batch))

        if (epoch % 4 == 0):
            f.write(str(sum_loss) + ' ')
            model.eval()
            
            if flag:
                with ema.average_parameters():
                    print('Evaluating', end='')
                    t_test = evaluate(model, dataset, args)
                    t_valid = evaluate_valid(model, dataset, args)
            else:
                print('Evaluating', end='')
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
            
            print('\n epoch:%d, valid (NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f), test (NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f)'
                  % (epoch, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_valid[4], t_valid[5], t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()           
            model.train()

    f.close()
    sampler.close()
    print("Done")
