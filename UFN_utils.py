import numpy as np
import copy

def hard_neg_sample(u, seq, pos, neg, hard_items, cnt_items, args):
    reverse = [] # Record the location of neg_item needs to be reversed this step
    for uid, user in enumerate(np.nditer(u)):
        user = int(user)
        for pid in reversed(range(args.maxlen)):
            if (pos[uid][pid] == 0): break
            if (user,pid) not in hard_items: continue
            for neg_id in range(len(hard_items[(user,pid)])):
                hard = hard_items[(user,pid)][neg_id]
                neg[uid][pid][neg_id] = hard
                if (user,pid,hard) not in cnt_items: 
                    cnt_items[(user,pid,hard)] = 0
                cnt_items[(user,pid,hard)] += 1
                    
                if cnt_items[(user, pid, hard)] > args.reverse:
                    reverse.append((uid, pid, neg_id))
    return reverse

def update_hard_negs(u, pos, hard_items, all_logits, all_labels, all_samples, args):
    all_logits = all_logits.detach().cpu().numpy()
    all_labels = all_labels.detach().cpu().numpy()
    for uid,user in enumerate(np.nditer(u)):
        user = int(user)
        for pid in reversed(range(args.maxlen)):
            if (pos[uid][pid] == 0): break
            hard_items[(user, pid)] = []
            for neg_id in range(args.neg_nums):
                if (all_logits[uid][pid][neg_id + 1] > all_logits[uid][pid][0] and all_labels[uid][pid][neg_id + 1] != 1):
                    hard_items[(user,pid)].append(copy.deepcopy(int(all_samples[uid][pid][neg_id + 1])))
    return
